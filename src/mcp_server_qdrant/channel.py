"""Channel adapter for proactive MCP notifications."""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

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
