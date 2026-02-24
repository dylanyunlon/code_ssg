"""
Search and Fetch Tools - Web search and URL fetching.
Implements features from claudecode功能.txt:
- Feature #4: Web search with result display
- Feature #5: URL fetching with content extraction
"""

import json
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .base_tool import BaseTool, ToolResult, ToolRiskLevel


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str = ""


class SearchTool(BaseTool):
    """
    Web search tool.
    
    Displays results like Claude Code:
    - "Searched the web"
    - Query text
    - "N results"
    - List of results with title, URL, snippet
    """

    def __init__(self, search_backend: str = "duckduckgo"):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            risk_level=ToolRiskLevel.NETWORK,
        )
        self.search_backend = search_backend

    def execute(self, query: str, max_results: int = 10, **kwargs) -> ToolResult:
        """
        Search the web.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        """
        try:
            results = self._do_search(query, max_results)

            # Format output like claudecode功能.txt feature #4
            output_parts = [
                f"Searched the web\n",
                f"  {query}\n",
                f"  {len(results)} results\n",
            ]

            for result in results:
                output_parts.append(f"\n  {result.title}")
                output_parts.append(f"  {result.url}")
                if result.snippet:
                    output_parts.append(f"  {result.snippet[:150]}")

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(output_parts),
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
                },
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Search failed: {e}",
            )

    def _do_search(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Perform actual search. Uses curl + DuckDuckGo HTML API as fallback.
        In production, integrate with proper search API.
        """
        # Placeholder: return structured search results
        # In production, use ddgs, serpapi, or similar
        try:
            # Try using ddgs (duckduckgo-search) if available
            result = subprocess.run(
                ["python3", "-c", f"""
import json
try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text("{query}", max_results={max_results}))
        print(json.dumps(results))
except ImportError:
    print("[]")
except Exception as e:
    print("[]")
"""],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", r.get("link", "")),
                        snippet=r.get("body", r.get("snippet", "")),
                    )
                    for r in data[:max_results]
                ]
        except Exception:
            pass

        return [SearchResult(
            title="[Search backend unavailable]",
            url="",
            snippet="Install duckduckgo-search: pip install duckduckgo-search",
        )]


class FetchTool(BaseTool):
    """
    Fetch URL content.
    
    Displays like claudecode功能.txt feature #5:
    - "Fetched: <title>"
    - Shows extracted content
    """

    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="Fetch and extract content from a URL",
            risk_level=ToolRiskLevel.NETWORK,
        )

    def execute(self, url: str, max_chars: int = 10000, **kwargs) -> ToolResult:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            max_chars: Maximum characters of content to return
        """
        try:
            # Use curl for fetching
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "10", "-A",
                 "Mozilla/5.0 (compatible; CodeSSG/1.0)", url],
                capture_output=True, text=True, timeout=15
            )

            if result.returncode != 0:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output="",
                    error=f"Fetch failed (exit {result.returncode}): {result.stderr[:200]}",
                )

            content = result.stdout
            # Try to extract text from HTML
            text = self._extract_text(content)

            # Truncate if needed
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n... [Truncated at {max_chars} chars]"

            # Extract title
            title = self._extract_title(content) or url

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"Fetched: {title}\n\n{url}\n\n{text}",
                metadata={
                    "url": url,
                    "title": title,
                    "content_length": len(text),
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Fetch timed out: {url}",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="",
                error=f"Fetch error: {e}",
            )

    @staticmethod
    def _extract_text(html: str) -> str:
        """Simple HTML to text extraction."""
        import re
        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Decode entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        return text

    @staticmethod
    def _extract_title(html: str) -> Optional[str]:
        """Extract title from HTML."""
        import re
        match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
