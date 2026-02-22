from typing import Optional
from datetime import datetime

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.config import settings


class ShortTermMemory(BaseChatMessageHistory):
    """In-memory sliding-window chat history for a single session."""

    def __init__(self, session_id: str, max_messages: int = 0):
        self.session_id = session_id
        self.max_messages = max_messages or settings.SHORT_TERM_MAX_MESSAGES
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def clear(self) -> None:
        self._messages = []


# Global store keyed by session_id
_session_store: dict[str, ShortTermMemory] = {}


def get_session_history(session_id: str) -> ShortTermMemory:
    """Get or create a ShortTermMemory for the given session."""
    if session_id not in _session_store:
        _session_store[session_id] = ShortTermMemory(session_id)
    return _session_store[session_id]


def clear_session(session_id: str) -> bool:
    """Clear and remove a session's memory. Returns True if session existed."""
    if session_id in _session_store:
        _session_store[session_id].clear()
        del _session_store[session_id]
        return True
    return False


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_session_store.keys())
