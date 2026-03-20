import pytest
import time
from mcp_server_qdrant.channel import ChannelSessionState


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
