"""Channel adapter for proactive MCP notifications."""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.types import JSONRPCNotification
from mcp.shared.message import SessionMessage


@dataclass
class ChannelSessionState:
    """Tracks channel push state for a single session."""

    cooldown_seconds: int = 60
    max_per_session: int = 20

    pushed_point_ids: set[str] = field(default_factory=set)
    retrieved_point_ids: set[str] = field(default_factory=set)
    push_count: int = 0
    consecutive_ignored: int = 0
    last_push_time: float = 0.0
    session_context_sent: bool = False

    def can_push(self) -> bool:
        if self.push_count >= self.max_per_session:
            return False
        if self.is_backed_off():
            return False
        elapsed = time.time() - self.last_push_time
        return elapsed >= self.cooldown_seconds

    def is_already_pushed(self, point_id: str) -> bool:
        return point_id in self.pushed_point_ids

    def is_backed_off(self) -> bool:
        return self.consecutive_ignored >= 3

    def record_push(self, point_id: str) -> None:
        self.pushed_point_ids.add(point_id)
        self.push_count += 1
        self.consecutive_ignored += 1
        self.last_push_time = time.time()

    def record_retrieval(self, point_id: str) -> None:
        self.retrieved_point_ids.add(point_id)
        if point_id in self.pushed_point_ids:
            self.consecutive_ignored = 0

    def mark_session_context_sent(self) -> None:
        self.session_context_sent = True


class ChannelNotifier:
    """Builds and sends channel notifications via MCP protocol."""

    CHANNEL_METHOD = "notifications/claude/channel"

    def build_notification(
        self, content: str, meta: dict[str, str] | None = None
    ) -> JSONRPCNotification:
        return JSONRPCNotification(
            jsonrpc="2.0",
            method=self.CHANNEL_METHOD,
            params={"content": content, "meta": meta or {}},
        )

    async def send(
        self,
        session: object,
        content: str,
        meta: dict[str, str] | None = None,
    ) -> None:
        """Send via private _write_stream bypass (no public API for custom notifications)."""
        notification = self.build_notification(content, meta)
        await session._write_stream.send(SessionMessage(message=notification))


class JournalWatcher:
    """Reads new entries from the context journal file."""

    def __init__(self, journal_path: str):
        self.journal_path = journal_path
        self._file_offset: int = 0

    def read_new_entries(self) -> list[dict]:
        if not os.path.exists(self.journal_path):
            return []
        try:
            with open(self.journal_path, 'r') as f:
                f.seek(self._file_offset)
                new_data = f.read()
                self._file_offset = f.tell()
            if not new_data.strip():
                return []
            return self.parse_entries(new_data)
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def parse_entries(raw: str) -> list[dict]:
        entries = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    @staticmethod
    def extract_search_text(entries: list[dict]) -> str:
        parts = []
        for entry in entries:
            if 'file' in entry:
                p = Path(entry['file'])
                parts.extend(p.parts[-3:])
                parts.append(p.stem)
            if 'pattern' in entry:
                parts.append(entry['pattern'])
            if 'command' in entry:
                parts.append(entry['command'])
            if 'description' in entry:
                parts.append(entry['description'])
        return ' '.join(parts)


STALE_MEMORY_DAYS = 30


class ChannelOrchestrator:
    """Coordinates journal watching, similarity search, rate limiting, and notification sending."""

    def __init__(
        self,
        connector: Any,
        collections: dict[str, str],
        session: Any,
        journal_path: str = "",
        similarity_threshold: float = 0.80,
        cooldown_seconds: int = 60,
        max_per_session: int = 20,
        project_filter: str | None = None,
    ) -> None:
        self._connector = connector
        self._collections = collections
        self._project_filter = project_filter
        self._session = session
        self._similarity_threshold = similarity_threshold
        self._state = ChannelSessionState(
            cooldown_seconds=cooldown_seconds,
            max_per_session=max_per_session,
        )
        self._notifier = ChannelNotifier()
        self._journal_watcher = JournalWatcher(journal_path)

    async def check_journal(self) -> None:
        """Read new journal entries, search for relevant memory, and push if a match is found."""
        entries = self._journal_watcher.read_new_entries()
        if not entries:
            return

        search_text = self._journal_watcher.extract_search_text(entries)
        if not search_text.strip():
            return

        match = await self._find_best_match(search_text)
        if match is None:
            return

        best_entry, collection_label = match

        if not self._state.can_push():
            return

        point_id = self._extract_point_id(best_entry)
        if point_id and self._state.is_already_pushed(point_id):
            return

        # Truncate to ~200 chars with retrieval hint
        summary = best_entry.content[:180].rsplit(" ", 1)[0]
        tool_name = f"qdrant-find-{collection_label}"
        content = f"{summary}... Use {tool_name} for full context."
        meta = {"type": "memory_match", "point_id": point_id or ""}
        await self._notifier.send(self._session, content, meta)
        if point_id:
            self._state.record_push(point_id)

    def _build_project_filter(self):
        """Build a Qdrant filter to match only the configured project tag."""
        if not self._project_filter:
            return None
        from qdrant_client import models
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.project",
                    match=models.MatchValue(value=self._project_filter),
                )
            ]
        )

    async def _find_best_match(self, search_text: str):
        """Search all collections and return (entry, label) for the first unpushed result, or None."""
        query_filter = self._build_project_filter()
        for label, collection_name in self._collections.items():
            results = await self._connector.search(
                search_text,
                collection_name=collection_name,
                limit=5,
                query_filter=query_filter,
            )
            for result in results:
                point_id = self._extract_point_id(result)
                if point_id and self._state.is_already_pushed(point_id):
                    continue
                return result, label
        return None

    async def check_stale_memory(self, entries: list, collection_name: str) -> None:
        """Send a warning notification for any entry whose stored_at is older than STALE_MEMORY_DAYS."""
        now = datetime.now(timezone.utc)
        for entry in entries:
            stored_at = self._parse_stored_at(entry)
            if stored_at is None:
                continue
            age_days = (now - stored_at).days
            if age_days < STALE_MEMORY_DAYS:
                continue
            meta = {
                "type": "stale_memory",
                "collection": collection_name,
                "age_days": str(age_days),
            }
            await self._notifier.send(self._session, entry.content, meta)

    def record_retrieval(self, point_id: str) -> None:
        """Record that the agent retrieved a pushed point, resetting backoff."""
        self._state.record_retrieval(point_id)

    @staticmethod
    def _extract_point_id(entry: Any) -> str | None:
        if entry.metadata and "id" in entry.metadata:
            return str(entry.metadata["id"])
        return None

    @staticmethod
    def _parse_stored_at(entry: Any) -> "datetime | None":
        if not entry.metadata:
            return None
        raw = entry.metadata.get("stored_at")
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            return None
