import re
from typing import Dict, List

import requests


class WebSearchClient:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            return self._search_with_ddgs(query=query, max_results=max_results)
        except Exception:
            return self._search_with_duckduckgo_html(query=query, max_results=max_results)

    def _search_with_ddgs(self, query: str, max_results: int) -> List[Dict[str, str]]:
        from ddgs import DDGS

        results: List[Dict[str, str]] = []
        with DDGS(timeout=self.timeout) as ddgs:
            for item in ddgs.text(query, max_results=max_results, safesearch="moderate", region="vn-vi"):
                title = (item.get("title") or "").strip()
                body = (item.get("body") or "").strip()
                href = (item.get("href") or "").strip()
                if not (title or body):
                    continue
                results.append({"title": title, "snippet": body, "url": href})
        return results

    def _search_with_duckduckgo_html(self, query: str, max_results: int) -> List[Dict[str, str]]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        html = response.text

        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html,
            flags=re.DOTALL,
        )

        cleaned: List[Dict[str, str]] = []
        for href, raw_title, raw_snippet in blocks[:max_results]:
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
            if not (title or snippet):
                continue
            cleaned.append({"title": title, "snippet": snippet, "url": href.strip()})

        return cleaned
