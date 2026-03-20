import pytest
import tempfile
import os
import time
from unittest.mock import AsyncMock, MagicMock
from mcp_server_qdrant.channel import ChannelSessionState, ChannelNotifier, JournalWatcher


class TestChannelSessionState:
    def test_initial_state(self):
        state = ChannelSessionState()
        assert state.push_count == 0
        assert state.session_context_sent is False
        assert len(state.pushed_point_ids) == 0

    def test_can_push_respects_cooldown(self):
        state = ChannelSessionState(cooldown_seconds=60)
        assert state.can_push() is True
        state.record_push("point1")
        assert state.can_push() is False  # within cooldown

    def test_can_push_after_cooldown(self):
        state = ChannelSessionState(cooldown_seconds=0)
        state.record_push("point1")
        assert state.can_push() is True

    def test_dedup_prevents_repeat(self):
        state = ChannelSessionState(cooldown_seconds=0)
        assert state.is_already_pushed("point1") is False
        state.record_push("point1")
        assert state.is_already_pushed("point1") is True

    def test_backoff_after_ignored(self):
        state = ChannelSessionState(cooldown_seconds=0)
        state.record_push("p1")
        state.record_push("p2")
        state.record_push("p3")
        assert state.is_backed_off() is True

    def test_backoff_resets_on_retrieval(self):
        state = ChannelSessionState(cooldown_seconds=0)
        state.record_push("p1")
        state.record_push("p2")
        state.record_push("p3")
        assert state.is_backed_off() is True
        state.record_retrieval("p1")
        assert state.is_backed_off() is False

    def test_max_per_session(self):
        state = ChannelSessionState(cooldown_seconds=0, max_per_session=2)
        state.record_push("p1")
        state.record_push("p2")
        assert state.can_push() is False

    def test_mark_session_context_sent(self):
        state = ChannelSessionState()
        assert state.session_context_sent is False
        state.mark_session_context_sent()
        assert state.session_context_sent is True


class TestChannelNotifier:
    @pytest.mark.asyncio
    async def test_build_notification(self):
        notifier = ChannelNotifier()
        notif = notifier.build_notification(
            content="Decision exists: use async hooks.",
            meta={"type": "memory_match", "score": "0.91", "point_id": "abc123"},
        )
        assert notif.method == "notifications/claude/channel"
        assert notif.params["content"] == "Decision exists: use async hooks."
        assert notif.params["meta"]["type"] == "memory_match"

    @pytest.mark.asyncio
    async def test_send_notification(self):
        mock_write_stream = AsyncMock()
        mock_session = MagicMock()
        mock_session._write_stream = mock_write_stream

        notifier = ChannelNotifier()
        await notifier.send(
            session=mock_session,
            content="Test message",
            meta={"type": "memory_match", "point_id": "abc"},
        )
        mock_write_stream.send.assert_called_once()


class TestJournalWatcher:
    def test_parse_journal_entries(self):
        entries = JournalWatcher.parse_entries(
            '{"tool":"Read","file":"src/hooks/test.sh","ts":"2026-03-20T01:30:00Z"}\n'
            '{"tool":"Grep","pattern":"channel","ts":"2026-03-20T01:30:05Z"}\n'
        )
        assert len(entries) == 2
        assert entries[0]["file"] == "src/hooks/test.sh"
        assert entries[1]["pattern"] == "channel"

    def test_extract_search_text(self):
        entries = [
            {"tool": "Read", "file": "src/hooks/session-start.sh"},
            {"tool": "Grep", "pattern": "channelMessage"},
            {"tool": "Edit", "file": "src/server.ts"},
        ]
        text = JournalWatcher.extract_search_text(entries)
        assert "hooks" in text
        assert "session-start" in text
        assert "channelMessage" in text
        assert "server" in text

    def test_read_new_entries(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"tool":"Read","file":"a.py","ts":"2026-03-20T01:00:00Z"}\n')
            f.write('{"tool":"Edit","file":"b.py","ts":"2026-03-20T01:00:01Z"}\n')
            path = f.name
        try:
            watcher = JournalWatcher(path)
            entries = watcher.read_new_entries()
            assert len(entries) == 2
            entries2 = watcher.read_new_entries()
            assert len(entries2) == 0
        finally:
            os.unlink(path)

    def test_read_new_entries_appended(self):
        """Test that watcher picks up entries appended after first read."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"tool":"Read","file":"a.py","ts":"2026-03-20T01:00:00Z"}\n')
            path = f.name
        try:
            watcher = JournalWatcher(path)
            entries = watcher.read_new_entries()
            assert len(entries) == 1
            # Append more
            with open(path, 'a') as f:
                f.write('{"tool":"Edit","file":"c.py","ts":"2026-03-20T01:00:02Z"}\n')
            entries2 = watcher.read_new_entries()
            assert len(entries2) == 1
            assert entries2[0]["file"] == "c.py"
        finally:
            os.unlink(path)

    def test_read_nonexistent_file(self):
        watcher = JournalWatcher("/tmp/nonexistent-journal-12345.jsonl")
        entries = watcher.read_new_entries()
        assert len(entries) == 0
