import pytest
import tempfile
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_server_qdrant.channel import ChannelSessionState, ChannelNotifier, JournalWatcher, ChannelOrchestrator
from mcp_server_qdrant.qdrant import Entry


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


class TestChannelOrchestrator:
    @pytest.mark.asyncio
    async def test_check_journal_pushes_on_match(self):
        mock_connector = AsyncMock()
        mock_connector.search.return_value = [
            Entry(content="hooks use async event-driven pattern", metadata={"id": "abc123"})
        ]
        mock_session = MagicMock()
        mock_session._write_stream = AsyncMock()

        orchestrator = ChannelOrchestrator(
            connector=mock_connector,
            collections={"memory": "mem-collection", "research": "res-collection"},
            session=mock_session,
            similarity_threshold=0.85,
            cooldown_seconds=0,
        )
        orchestrator._journal_watcher = MagicMock()
        orchestrator._journal_watcher.read_new_entries.return_value = [
            {"tool": "Read", "file": "src/hooks/session-start.sh"}
        ]
        orchestrator._journal_watcher.extract_search_text.return_value = "hooks session-start"

        await orchestrator.check_journal()
        mock_session._write_stream.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_journal_skips_when_no_match(self):
        mock_connector = AsyncMock()
        mock_connector.search.return_value = []
        mock_session = MagicMock()
        mock_session._write_stream = AsyncMock()

        orchestrator = ChannelOrchestrator(
            connector=mock_connector,
            collections={"memory": "mem-collection"},
            session=mock_session,
            cooldown_seconds=0,
        )
        orchestrator._journal_watcher = MagicMock()
        orchestrator._journal_watcher.read_new_entries.return_value = [
            {"tool": "Read", "file": "src/unrelated.py"}
        ]
        orchestrator._journal_watcher.extract_search_text.return_value = "unrelated"

        await orchestrator.check_journal()
        mock_session._write_stream.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_rate_limits(self):
        mock_connector = AsyncMock()
        mock_connector.search.return_value = [
            Entry(content="some pattern", metadata={"id": "xyz"})
        ]
        mock_session = MagicMock()
        mock_session._write_stream = AsyncMock()

        orchestrator = ChannelOrchestrator(
            connector=mock_connector,
            collections={"memory": "mem-collection"},
            session=mock_session,
            cooldown_seconds=9999,
        )
        orchestrator._journal_watcher = MagicMock()
        orchestrator._journal_watcher.read_new_entries.return_value = [
            {"tool": "Read", "file": "src/a.py"}
        ]
        orchestrator._journal_watcher.extract_search_text.return_value = "a"

        await orchestrator.check_journal()
        assert mock_session._write_stream.send.call_count == 1

        # Second push within cooldown should not fire
        mock_connector.search.return_value = [
            Entry(content="another pattern", metadata={"id": "xyz2"})
        ]
        orchestrator._journal_watcher.read_new_entries.return_value = [
            {"tool": "Read", "file": "src/b.py"}
        ]
        orchestrator._journal_watcher.extract_search_text.return_value = "b"

        await orchestrator.check_journal()
        assert mock_session._write_stream.send.call_count == 1  # unchanged

    @pytest.mark.asyncio
    async def test_dedup_same_point(self):
        mock_connector = AsyncMock()
        mock_connector.search.return_value = [
            Entry(content="same pattern", metadata={"id": "same123"})
        ]
        mock_session = MagicMock()
        mock_session._write_stream = AsyncMock()

        orchestrator = ChannelOrchestrator(
            connector=mock_connector,
            collections={"memory": "mem-collection"},
            session=mock_session,
            cooldown_seconds=0,
        )
        orchestrator._journal_watcher = MagicMock()
        orchestrator._journal_watcher.read_new_entries.return_value = [
            {"tool": "Read", "file": "src/a.py"}
        ]
        orchestrator._journal_watcher.extract_search_text.return_value = "a"

        await orchestrator.check_journal()
        assert mock_session._write_stream.send.call_count == 1

        # Same point_id again — should be deduped
        await orchestrator.check_journal()
        assert mock_session._write_stream.send.call_count == 1  # unchanged

    @pytest.mark.asyncio
    async def test_check_stale_memory(self):
        mock_session = MagicMock()
        mock_session._write_stream = AsyncMock()

        orchestrator = ChannelOrchestrator(
            connector=AsyncMock(),
            collections={"memory": "mem-collection"},
            session=mock_session,
            cooldown_seconds=0,
        )
        # Entry with old stored_at
        entries = [
            Entry(content="Old build config", metadata={"id": "old1", "stored_at": "2025-01-01T00:00:00Z"})
        ]
        await orchestrator.check_stale_memory(entries, "mem-collection")
        mock_session._write_stream.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_retrieval_resets_backoff(self):
        orchestrator = ChannelOrchestrator(
            connector=AsyncMock(),
            collections={},
            session=MagicMock(),
            cooldown_seconds=0,
        )
        orchestrator._state.record_push("p1")
        orchestrator._state.record_push("p2")
        orchestrator._state.record_push("p3")
        assert orchestrator._state.is_backed_off() is True
        orchestrator.record_retrieval("p1")
        assert orchestrator._state.is_backed_off() is False
