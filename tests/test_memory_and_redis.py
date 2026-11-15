"""
Unit and Integration Tests for Short-Term Memory and Distributed Redis Session Store.
Verifies sliding-window buffer eviction, dialogue formatting, Redis state serialization,
TTL expiry handling, and seamless in-memory fallback.
"""

import pytest
import uuid
from unittest.mock import MagicMock

from src.memory import (
    ShortTermMemoryManager,
    RedisSessionStore,
    ShortTermSessionManager,
)
from src.models import ConversationState, FlowStep, QualificationStatus
from src.agent import LeadQualificationAgent


class TestShortTermMemoryManager:
    def test_sliding_window_buffer_eviction(self):
        manager = ShortTermMemoryManager(window_size=5)
        session_id = "test-session-1"

        # Add 8 turns (exceeding window_size=5)
        for i in range(1, 9):
            role = "user" if i % 2 != 0 else "assistant"
            manager.add_turn(session_id, role, f"Message {i}")

        window = manager.get_sliding_window(session_id)
        assert len(window) == 5
        # Oldest turns (1, 2, 3) should be evicted; turns 4, 5, 6, 7, 8 should remain
        assert window[0]["content"] == "Message 4"
        assert window[-1]["content"] == "Message 8"

    def test_formatted_dialogue_transcript(self):
        manager = ShortTermMemoryManager(window_size=5)
        session_id = "test-session-2"

        manager.add_turn(session_id, "user", "I want to buy a house")
        manager.add_turn(session_id, "assistant", "Do you have a specific property in mind?")
        manager.add_turn(session_id, "user", "No")

        dialogue = manager.get_formatted_dialogue(session_id)
        assert "Visitor: I want to buy a house" in dialogue
        assert "Assistant: Do you have a specific property in mind?" in dialogue
        assert "Visitor: No" in dialogue

    def test_clear_session_memory(self):
        manager = ShortTermMemoryManager(window_size=5)
        session_id = "test-session-3"
        manager.add_turn(session_id, "user", "Hello")
        assert len(manager.get_sliding_window(session_id)) == 1

        manager.clear(session_id)
        assert len(manager.get_sliding_window(session_id)) == 0


class TestRedisSessionStore:
    def test_in_memory_fallback_when_redis_offline(self):
        # Initialize with dummy unreachable port to force in-memory fallback
        store = RedisSessionStore(redis_url="redis://127.0.0.1:9999/0", enabled=True)
        assert store.is_connected is False

        session_id = str(uuid.uuid4())
        state = ConversationState(session_id=session_id)
        state.lead.visitor_name = "Sophia"
        state.qualification_status = QualificationStatus.QUALIFIED

        # Save and load should work seamlessly via local fallback
        store.save_session(session_id, state)
        loaded = store.load_session(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id
        assert loaded.lead.visitor_name == "Sophia"
        assert loaded.qualification_status == QualificationStatus.QUALIFIED

    def test_redis_mock_persistence_and_ttl(self):
        store = RedisSessionStore(enabled=False)
        mock_client = MagicMock()
        store.client = mock_client
        store.is_connected = True
        store.ttl_seconds = 3600

        session_id = "mock-redis-session"
        state = ConversationState(session_id=session_id)
        state.lead.visitor_name = "Marcus"

        store.save_session(session_id, state)
        # Verify set was called on Redis with TTL expiry
        mock_client.set.assert_called_once()
        args, kwargs = mock_client.set.call_args
        assert args[0] == f"ashcombe:session:{session_id}"
        assert kwargs["ex"] == 3600

    def test_telemetry_status(self):
        store = RedisSessionStore(enabled=False)
        status = store.get_status()
        assert "provider" in status
        assert "ttl_seconds" in status
        assert status["connected"] is False


class TestAgentMemoryIntegration:
    def test_agent_chat_updates_sliding_memory_and_persists_state(self):
        agent = LeadQualificationAgent()
        session_id = str(uuid.uuid4())

        # First turn
        agent.chat(session_id, "Hello")
        agent.chat(session_id, "Arthur")

        # Verify short term memory tracked the turns
        window = agent.memory.get_sliding_window(session_id)
        assert len(window) >= 2

        # Verify session can be loaded back from memory store
        loaded = agent.memory.load_session(session_id)
        assert loaded is not None
        assert loaded.lead.visitor_name == "Arthur"

        # Telemetry reports memory status
        diag = agent.get_system_diagnostics()
        assert "short_term_memory" in diag
        assert diag["short_term_memory"]["window_size"] == 5
