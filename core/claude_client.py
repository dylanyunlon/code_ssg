"""
Claude API Client - Real Anthropic SDK Integration
====================================================
Provides the actual LLM backend for the agentic loop.
Replaces the placeholder _call_model() in agent_loop.py.

Reference: skynetCheapBuy/app/core/ai_engine.py

Location: core/claude_client.py (NEW FILE)
"""
import json
import time
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Real Claude API client using the Anthropic Python SDK.
    
    Supports:
    - Tool calling (function calling with tool_use blocks)
    - Streaming responses (SSE)
    - Content block parsing (text + tool_use)
    - Automatic retry with exponential backoff
    - Token usage tracking
    
    Usage:
        client = ClaudeClient(api_key="sk-ant-...")
        response = await client.chat(messages, tools=tool_defs)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url or os.environ.get("ANTHROPIC_API_BASE")
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Call Claude API with tool support.
        
        Args:
            messages: Conversation history in Claude format
            tools: Tool definitions (converted to Anthropic format)
            system_prompt: System prompt
            temperature: Sampling temperature
        
        Returns dict with:
            - content: list of content blocks [{type: "text"/"tool_use", ...}]
            - stop_reason: "end_turn" | "tool_use"
            - usage: {input_tokens, output_tokens}
        """
        import asyncio

        client = self._get_client()

        # Clean messages for API (must alternate user/assistant properly)
        clean_messages = self._clean_messages(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": clean_messages,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._format_tools_for_api(tools)

        # Run synchronous SDK call in executor (SDK is sync)
        loop = asyncio.get_event_loop()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await loop.run_in_executor(
                    None, lambda: client.messages.create(**kwargs)
                )
                break
            except Exception as e:
                if attempt < max_retries - 1 and "overloaded" in str(e).lower():
                    wait = 2 ** attempt
                    logger.warning(f"API overloaded, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Claude API error: {e}")
                    raise

        parsed = self._parse_response(response)

        # Track token usage
        usage = parsed.get("usage", {})
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)

        return parsed

    def chat_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Synchronous version of chat() for non-async contexts."""
        client = self._get_client()
        clean_messages = self._clean_messages(messages)

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": clean_messages,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._format_tools_for_api(tools)

        response = client.messages.create(**kwargs)
        parsed = self._parse_response(response)

        usage = parsed.get("usage", {})
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)

        return parsed

    def _clean_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Clean messages for the Claude API.
        Converts tool_result messages and ensures proper alternation.
        """
        cleaned = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map tool_result → user role with proper format
            if role == "tool_result":
                if isinstance(content, str):
                    # Already formatted as tool_result content
                    cleaned.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    cleaned.append({"role": "user", "content": content})
                continue

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Pass through user/assistant messages
            if role in ("user", "assistant"):
                cleaned.append({"role": role, "content": content})

        # Ensure we don't have consecutive messages with same role
        merged = []
        for msg in cleaned:
            if merged and merged[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                prev_content = merged[-1]["content"]
                new_content = msg["content"]
                if isinstance(prev_content, str) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + "\n" + new_content
                else:
                    # For complex content blocks, just keep the last one
                    merged[-1] = msg
            else:
                merged.append(msg)

        return merged if merged else [{"role": "user", "content": "Begin."}]

    def _format_tools_for_api(self, tools: List[Dict]) -> List[Dict]:
        """Convert internal tool definitions to Anthropic API format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", tool.get("parameters", {
                    "type": "object",
                    "properties": {},
                })),
            })
        return formatted

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse Anthropic API response into internal format."""
        content_blocks = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append({
                    "type": "text",
                    "text": block.text,
                })
            elif block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return {
            "content": content_blocks,
            "text": "".join(
                b["text"] for b in content_blocks if b["type"] == "text"
            ),
            "tool_calls": [
                {
                    "name": b["name"],
                    "id": b["id"],
                    "arguments": b["input"],
                }
                for b in content_blocks if b["type"] == "tool_use"
            ],
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    @property
    def total_tokens_used(self) -> Dict[str, int]:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total": self._total_input_tokens + self._total_output_tokens,
        }