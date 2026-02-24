"""
Message Queue - Async message handling with pause/resume.
Inspired by Claude Code's h2A dual-buffer queue.
Allows user interjections mid-task without full restart.
"""

import threading
import queue
from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum


class MessageType(Enum):
    USER_INPUT = "user_input"
    USER_INTERRUPT = "user_interrupt"  # Mid-task interjection
    TOOL_RESULT = "tool_result"
    PLAN_UPDATE = "plan_update"
    SYSTEM = "system"
    STOP = "stop"


@dataclass
class QueueMessage:
    """A message in the queue."""
    type: MessageType
    content: str
    priority: int = 0  # Higher = process first

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class MessageQueue:
    """
    Async message queue with pause/resume support.
    
    Allows the user to inject new instructions mid-task.
    The agentic loop checks for new messages between tool calls.
    """

    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._paused = False
        self._lock = threading.Lock()
        self._callbacks = []

    def put(self, message: QueueMessage):
        """Add a message to the queue."""
        self._queue.put(message)

    def put_user_input(self, content: str):
        """Add user input message."""
        self.put(QueueMessage(MessageType.USER_INPUT, content, priority=1))

    def put_interrupt(self, content: str):
        """Add a user interrupt (high priority)."""
        self.put(QueueMessage(MessageType.USER_INTERRUPT, content, priority=10))

    def put_tool_result(self, content: str):
        """Add a tool result."""
        self.put(QueueMessage(MessageType.TOOL_RESULT, content, priority=0))

    def get(self, timeout: float = 0.1) -> Optional[QueueMessage]:
        """Get next message from queue."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_interrupt(self) -> bool:
        """Check if there's a pending user interrupt (without consuming it)."""
        # This is a simplified check - in production, use peek
        return not self._queue.empty()

    def pause(self):
        """Pause processing."""
        with self._lock:
            self._paused = True

    def resume(self):
        """Resume processing."""
        with self._lock:
            self._paused = False

    def is_paused(self) -> bool:
        """Check if queue is paused."""
        with self._lock:
            return self._paused

    def stop(self):
        """Signal the loop to stop."""
        self.put(QueueMessage(MessageType.STOP, "", priority=100))

    def clear(self):
        """Clear all pending messages."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
