"""
Local Short-Term Memory and Session Store for Ashcombe & Rowe Estates.
Maintains strictly the last 5 turns of conversation in local memory.
Automatically evicts and clears all dialogue turns older than 5 turns.
100% self-contained locally without any external workflows or daemon dependencies.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.config import settings
from src.models import ConversationState

logger = logging.getLogger("ashcombe_rowe.memory")


@dataclass
class ShortTermTurn:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)


class ShortTermMemoryManager:
    """
    Manages local in-memory short-term conversational context strictly capped at the last 5 turns.
    Automatically evicts and clears all dialogue turns older than the last 5 turns.
    Runs 100% locally with zero external services or network dependencies.
    """

    DEFAULT_WINDOW_SIZE = 5

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size or settings.SHORT_TERM_MEMORY_WINDOW or self.DEFAULT_WINDOW_SIZE
        self._buffers: Dict[str, List[ShortTermTurn]] = {}

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Appends a dialogue turn and automatically clears everything before the last 5 turns."""
        if not session_id or not content:
            return
        if session_id not in self._buffers:
            self._buffers[session_id] = []
        
        self._buffers[session_id].append(ShortTermTurn(role=role, content=content))
        # Strictly keep the last 5 turns; automatically prune any turn before the 5th
        if len(self._buffers[session_id]) > self.window_size:
            self._buffers[session_id] = self._buffers[session_id][-self.window_size:]

    def get_sliding_window(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Returns strictly the last 5 turns (or fewer if conversation just began)."""
        turns = self._buffers.get(session_id, [])
        effective_limit = min(limit or self.window_size, self.window_size)
        recent = turns[-effective_limit:]
        return [{"role": t.role, "content": t.content} for t in recent]

    def get_formatted_dialogue(self, session_id: str, limit: Optional[int] = 5) -> str:
        """Renders the last 5 turns into a compact dialogue transcript for prompt context."""
        window = self.get_sliding_window(session_id, limit=limit or 5)
        if not window:
            return "No prior dialogue in active window."
        lines = []
        for t in window:
            speaker = "Visitor" if t["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {t['content']}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        """Clears sliding window memory for a session."""
        self._buffers.pop(session_id, None)


class RedisSessionStore:
    """
    Distributed Redis session storage and checkpointing.
    Provides fast, persistent state storage with automatic TTL expiry.
    If Redis is unavailable or offline, gracefully falls back to in-memory state.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
    ):
        self.redis_url = redis_url or settings.REDIS_URL
        self.enabled = settings.REDIS_ENABLED if enabled is None else enabled
        self.ttl_seconds = ttl_seconds or settings.REDIS_TTL_SECONDS
        self.client = None
        self.is_connected = False
        self._local_memory: Dict[str, Dict[str, Any]] = {}

        if self.enabled:
            self._init_redis_client()

    def _init_redis_client(self) -> None:
        try:
            import redis
            self.client = redis.from_url(
                self.redis_url,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
                decode_responses=True,
            )
            # Ping test to verify active socket connection
            self.client.ping()
            self.is_connected = True
            logger.info(f"Connected to Redis session store at {self.redis_url} (TTL: {self.ttl_seconds}s)")
        except Exception as e:
            self.is_connected = False
            self.client = None
            logger.warning(
                f"Redis unavailable at {self.redis_url} ({e}). "
                "Operating with in-memory short-term session fallback (zero disruption)."
            )

    def save_session(self, session_id: str, state: ConversationState) -> None:
        """Persists ConversationState to Redis with TTL, falling back to local memory."""
        if not session_id or not state:
            return
        
        state_dict = state.model_dump()
        # Always maintain in local memory as ultra-fast tier
        self._local_memory[session_id] = state_dict

        if self.is_connected and self.client:
            try:
                key = f"ashcombe:session:{session_id}"
                raw_json = json.dumps(state_dict)
                self.client.set(key, raw_json, ex=self.ttl_seconds)
            except Exception as e:
                logger.error(f"Failed to persist session {session_id} to Redis: {e}. Kept in local memory.")

    def load_session(self, session_id: str) -> Optional[ConversationState]:
        """Loads ConversationState from Redis or local memory fallback."""
        if not session_id:
            return None

        # Try Redis first if connected
        if self.is_connected and self.client:
            try:
                key = f"ashcombe:session:{session_id}"
                raw = self.client.get(key)
                if raw:
                    data = json.loads(raw)
                    self._local_memory[session_id] = data
                    return ConversationState.model_validate(data)
            except Exception as e:
                logger.error(f"Failed to load session {session_id} from Redis: {e}. Falling back to local memory.")

        # Fallback to local memory
        local_data = self._local_memory.get(session_id)
        if local_data:
            return ConversationState.model_validate(local_data)

        return None

    def delete_session(self, session_id: str) -> None:
        """Deletes session from Redis and local memory."""
        self._local_memory.pop(session_id, None)
        if self.is_connected and self.client:
            try:
                self.client.delete(f"ashcombe:session:{session_id}")
            except Exception as e:
                logger.error(f"Failed to delete session {session_id} from Redis: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Telemetry diagnostics for system observability."""
        return {
            "provider": "redis" if self.is_connected else "in_memory_fallback",
            "connected": self.is_connected,
            "redis_url": self.redis_url,
            "ttl_seconds": self.ttl_seconds,
            "active_local_sessions": len(self._local_memory),
        }


class ShortTermSessionManager:
    """
    Unified manager combining sliding-window short-term memory and distributed Redis session store.
    """

    def __init__(self):
        self.memory = ShortTermMemoryManager()
        self.store = RedisSessionStore()

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self.memory.add_turn(session_id, role, content)

    def get_sliding_window(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        return self.memory.get_sliding_window(session_id, limit)

    def get_formatted_dialogue(self, session_id: str, limit: Optional[int] = 6) -> str:
        return self.memory.get_formatted_dialogue(session_id, limit)

    def save_session(self, session_id: str, state: ConversationState) -> None:
        self.store.save_session(session_id, state)

    def load_session(self, session_id: str) -> Optional[ConversationState]:
        return self.store.load_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self.memory.clear(session_id)
        self.store.delete_session(session_id)

    def get_status(self) -> Dict[str, Any]:
        status = self.store.get_status()
        status["window_size"] = self.memory.window_size
        return status


# Global singleton instance
short_term_memory = ShortTermSessionManager()
