"""
Claude API Client - httpx-based (matches skynetCheapBuy pattern)
=================================================================
Uses httpx to call /v1/messages endpoint directly, NOT the anthropic SDK.
This matches how skynetCheapBuy/app/core/ai_engine.py ClaudeCompatibleProvider works.

Loads API config from .env file via env_loader.

Key difference from previous version:
  - Uses httpx.AsyncClient instead of anthropic.Anthropic
  - Calls /v1/messages endpoint with Bearer token
  - Parses raw JSON response (not SDK objects)

Location: core/claude_client.py (REWRITTEN)
"""

import json
import time
import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Model pricing ($ per 1M tokens)
MODEL_PRICING = {
    "claude-opus-4-6":            {"input": 15.0,  "output": 75.0},
    "claude-opus-4-5-20251101":   {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.0},
    "_default":                   {"input": 3.0,   "output": 15.0},
}


class ClaudeClient:
    """
    Real Claude API client using httpx (NOT anthropic SDK).
    Matches skynetCheapBuy/app/core/ai_engine.py ClaudeCompatibleProvider pattern.
    
    Usage:
        client = ClaudeClient()  # reads from .env
        response = await client.chat(messages, tools=tool_defs)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = None,
        max_tokens: int = 8192,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        
        # Base URL handling - same as skynetCheapBuy
        raw_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("ANTHROPIC_API_BASE", "https://api.tryallai.com/v1")
        if raw_base.endswith("/v1"):
            raw_base = raw_base[:-3]
        elif raw_base.endswith("/v1/"):
            raw_base = raw_base[:-4]
        self._api_base = raw_base
        self._messages_endpoint = f"{self._api_base}/v1/messages"
        
        self.model = model or os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")
        self.max_tokens = max_tokens
        
        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
    
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call Claude API with tool support via httpx.
        
        Returns dict with:
            - content: list of content blocks [{type: "text"/"tool_use", ...}]
            - text: concatenated text content
            - tool_calls: extracted tool calls [{name, id, arguments}]
            - stop_reason: "end_turn" | "tool_use"
            - usage: {input_tokens, output_tokens}
        """
        import httpx
        
        used_model = model or self.model
        
        # Separate system messages from conversation
        system_content = system_prompt
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                if system_content:
                    system_content = system_content + "\n\n" + msg["content"]
                else:
                    system_content = msg["content"]
            elif isinstance(msg.get("content"), list):
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Build request body (native Claude format, same as skynetCheapBuy)
        request_body = {
            "model": used_model,
            "messages": claude_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }
        
        if system_content:
            request_body["system"] = system_content
        if tools:
            request_body["tools"] = self._format_tools(tools)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "anthropic-version": "2023-06-01",
        }
        
        timeout = 180.0 if tools else 60.0
        
        logger.info(f"Calling Claude Messages API: model={used_model}, endpoint={self._messages_endpoint}, tools={bool(tools)}")
        
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        self._messages_endpoint,
                        json=request_body,
                        headers=headers,
                    )
                    
                    if response.status_code != 200:
                        error_text = response.text
                        logger.error(f"Claude API error: {response.status_code} - {error_text}")
                        raise Exception(f"Claude API error: {response.status_code} - {error_text}")
                    
                    data = response.json()
                    parsed = self._parse_response(data)
                    
                    usage = parsed.get("usage", {})
                    self._total_input_tokens += usage.get("input_tokens", 0)
                    self._total_output_tokens += usage.get("output_tokens", 0)
                    self._total_cost += self._estimate_cost(
                        used_model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                    )
                    
                    return parsed
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1 and ("overloaded" in str(e).lower() or "timeout" in str(e).lower()):
                    wait = 2 ** attempt
                    logger.warning(f"API error (attempt {attempt+1}), retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise
        
        raise last_error
    
    def _format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool definitions to Anthropic API format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", tool.get("parameters", {
                    "type": "object", "properties": {},
                })),
            })
        return formatted
    
    def _parse_response(self, data: Dict) -> Dict[str, Any]:
        """Parse raw JSON response from Claude API."""
        content_blocks = data.get("content", [])
        
        text_parts = []
        tool_calls = []
        
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "name": block["name"],
                    "id": block.get("id", str(uuid.uuid4())),
                    "arguments": block.get("input", {}),
                })
        
        usage = {}
        if data.get("usage"):
            usage = {
                "input_tokens": data["usage"].get("input_tokens", 0),
                "output_tokens": data["usage"].get("output_tokens", 0),
            }
        
        return {
            "content": content_blocks,
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "stop_reason": data.get("stop_reason", "end_turn"),
            "usage": usage,
        }
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["_default"])
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    
    @property
    def total_tokens(self) -> Dict[str, Any]:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total": self._total_input_tokens + self._total_output_tokens,
            "estimated_cost": round(self._total_cost, 6),
        }