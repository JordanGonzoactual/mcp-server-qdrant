"""Channel adapter for proactive MCP notifications."""

import time
from dataclasses import dataclass, field

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
