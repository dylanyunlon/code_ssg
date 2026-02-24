"""
Context Manager - Manages message history and context compression.
Inspired by Claude Code's Compressor wU2 that triggers at ~92% context usage.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import time


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    PLAN_UPDATE = "plan_update"


@dataclass
class Message:
    """A single message in the conversation history."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    def token_estimate(self) -> int:
        """Rough token count estimate (4 chars ≈ 1 token)."""
        return len(self.content) // 4


class ContextManager:
    """
    Manages the flat message history for the agentic loop.
    
    Design principles (from Claude Code):
    - Flat message history (no complex threading)
    - Auto-compression at ~92% context window
    - Tool results fed back as messages
    - Plan state injected after tool calls
    """

    DEFAULT_MAX_TOKENS = 100000  # ~100K token context

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._system_prompt: Optional[str] = None
        self._compression_threshold = 0.92

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self._system_prompt = prompt

    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None):
        """Add a message to history."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)

        # Check if compression needed
        if self._should_compress():
            self._compress()

    def add_user_message(self, content: str):
        self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str):
        self.add_message(MessageRole.ASSISTANT, content)

    def add_tool_result(self, tool_name: str, result: str):
        self.add_message(MessageRole.TOOL_RESULT, result, {"tool": tool_name})

    def add_plan_update(self, plan_display: str):
        self.add_message(MessageRole.PLAN_UPDATE, plan_display)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages formatted for LLM API."""
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        for msg in self._messages:
            # Map roles to standard API roles
            if msg.role in (MessageRole.TOOL_RESULT, MessageRole.PLAN_UPDATE):
                role = "user"
            else:
                role = msg.role.value
            messages.append({"role": role, "content": msg.content})

        return messages

    def get_token_usage(self) -> int:
        """Estimate current token usage."""
        total = len(self._system_prompt or "") // 4
        total += sum(msg.token_estimate() for msg in self._messages)
        return total

    def get_usage_ratio(self) -> float:
        """Get ratio of used/max tokens."""
        return self.get_token_usage() / self.max_tokens

    def _should_compress(self) -> bool:
        """Check if we're approaching context limit."""
        return self.get_usage_ratio() >= self._compression_threshold

    def _compress(self):
        """
        Compress conversation history.
        Strategy: Keep system prompt, first message, last N messages,
        and summarize the middle.
        """
        if len(self._messages) <= 4:
            return

        # Keep first 2 and last 10 messages
        keep_start = 2
        keep_end = 10

        if len(self._messages) <= keep_start + keep_end:
            return

        middle = self._messages[keep_start:-keep_end]

        # Summarize middle section
        summary_parts = []
        tool_calls = 0
        for msg in middle:
            if msg.role == MessageRole.TOOL_RESULT:
                tool_calls += 1
            elif msg.role == MessageRole.ASSISTANT:
                # Keep first line of assistant messages
                first_line = msg.content.split('\n')[0][:200]
                summary_parts.append(f"- {first_line}")

        summary = (
            f"[Context compressed: {len(middle)} messages summarized]\n"
            f"[{tool_calls} tool calls were made]\n"
            f"Key actions:\n" + "\n".join(summary_parts[:10])
        )

        compressed_msg = Message(
            role=MessageRole.SYSTEM,
            content=summary,
        )

        self._messages = (
            self._messages[:keep_start] +
            [compressed_msg] +
            self._messages[-keep_end:]
        )

    def get_history_display(self) -> str:
        """Get a human-readable conversation summary."""
        total_msgs = len(self._messages)
        tokens = self.get_token_usage()
        ratio = self.get_usage_ratio()
        return (
            f"Context: {total_msgs} messages, ~{tokens} tokens "
            f"({ratio*100:.1f}% of {self.max_tokens})"
        )

    def clear(self):
        """Clear all messages."""
        self._messages = []
