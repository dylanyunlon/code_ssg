"""
Fetch Tool — HTTP page fetching with proper error handling.
=============================================================
Fetches web pages and returns their content. Supports:
  - HTML content extraction (strips tags for readability)
  - Timeout handling
  - Content truncation for large pages
  - Response header capture

Location: tools/fetch_tool.py (NEW FILE — plan item 2.7)
"""

import logging
import urllib.request
import urllib.error
import re
from typing import Any, Dict, Optional

from .base_tool import BaseTool, ToolResult, ToolRiskLevel

logger = logging.getLogger(__name__)


class FetchTool(BaseTool):
    """
    Fetch a URL and return its content.
    
    Claude Code equivalent: the `fetch` tool that shows
    "Fetched: <page title>" in step titles.
    """
    
    name = "fetch"
    description = "Fetch a web page and return its content"
    risk_level = ToolRiskLevel.LOW
    
    MAX_CONTENT_LENGTH = 50000  # chars
    DEFAULT_TIMEOUT = 30        # seconds
    
    def __init__(self, max_content_length: int = None, timeout: int = None):
        super().__init__()
        self.max_content_length = max_content_length or self.MAX_CONTENT_LENGTH
        self.timeout = timeout or self.DEFAULT_TIMEOUT
    
    @property
    def input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "strip_html": {
                    "type": "boolean",
                    "description": "Strip HTML tags for readability (default: True)",
                    "default": True,
                },
                "max_length": {
                    "type": "integer",
                    "description": f"Maximum content length (default: {MAX_CONTENT_LENGTH})",
                    "default": MAX_CONTENT_LENGTH,
                },
            },
            "required": ["url"],
        }
    
    def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        strip_html = kwargs.get("strip_html", True)
        max_length = kwargs.get("max_length", self.max_content_length)
        
        if not url:
            return ToolResult(
                success=False,
                output={"error": "URL is required"},
            )
        
        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Code-SSG/1.0 (Scientific Statement Generator)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                
                raw = response.read()
                content = raw.decode(charset, errors="replace")
                status = response.status
                
                # Extract title
                title = self._extract_title(content)
                
                # Strip HTML if requested
                if strip_html and ("text/html" in content_type or "<html" in content[:500].lower()):
                    content = self._strip_html(content)
                
                # Truncate
                truncated = False
                if len(content) > max_length:
                    content = content[:max_length]
                    truncated = True
                
                return ToolResult(
                    success=True,
                    output={
                        "url": url,
                        "title": title,
                        "content": content,
                        "status": status,
                        "content_type": content_type,
                        "truncated": truncated,
                        "content_length": len(content),
                    },
                )
        
        except urllib.error.HTTPError as e:
            return ToolResult(
                success=False,
                output={
                    "url": url,
                    "error": f"HTTP {e.code}: {e.reason}",
                    "status": e.code,
                },
            )
        except urllib.error.URLError as e:
            return ToolResult(
                success=False,
                output={
                    "url": url,
                    "error": f"URL error: {e.reason}",
                },
            )
        except TimeoutError:
            return ToolResult(
                success=False,
                output={
                    "url": url,
                    "error": f"Request timed out after {self.timeout}s",
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output={
                    "url": url,
                    "error": str(e),
                },
            )
    
    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract <title> from HTML."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up whitespace
            title = re.sub(r"\s+", " ", title)
            return title[:200]
        return ""
    
    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags and return readable text."""
        # Remove script and style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&nbsp;", " ")
        # Collapse whitespace
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


class WebSearchTool(BaseTool):
    """
    Web search tool stub. 
    
    In production, this would integrate with a search API.
    For now returns a structured response indicating search capability.
    """
    
    name = "web_search"
    description = "Search the web for information"
    risk_level = ToolRiskLevel.LOW
    
    @property
    def input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        }
    
    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 10)
        
        return ToolResult(
            success=True,
            output={
                "query": query,
                "max_results": max_results,
                "results": [],
                "note": "Web search requires API integration. Configure SEARCH_API_KEY in .env",
            },
        )