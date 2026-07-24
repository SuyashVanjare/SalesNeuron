"""
SalesNeuron — Browser Scraper
Playwright async scraper that:
  1. Navigates to a URL with a real browser (bypasses basic bot detection)
  2. Waits for JS to render
  3. Extracts clean text (strips nav/footer/ads/scripts)
  4. Returns page title + clean body text

Set HEADLESS_BROWSER=false in .env to watch it work in real time.
"""

import os
import asyncio
import logging
import re
from urllib.parse import urlparse
from typing import Optional
from playwright.async_api import async_playwright, Page, Browser

logger = logging.getLogger(__name__)

# Tags stripped when extracting page TEXT (not applied when extracting links)
# Note: nav/header are intentionally NOT in this list — we need their links.
# They are stripped only after link extraction is complete.
NOISE_TAGS = [
    "script", "style", "footer",
    "noscript", "iframe", "svg",
    "aside",
]

# Keywords that indicate a high-value page for sales research.
# Links matching these will be ranked to the top of the returned list.
_HIGH_VALUE_KEYWORDS = [
    "job", "jobs", "career", "careers", "hiring", "internship", "internships",
    "opening", "vacancy", "vacancies", "apply", "work-with-us", "work-at",
    "about", "team", "leadership", "founders", "people",
    "product", "products", "features", "solutions", "platform",
    "pricing", "plans",
    "news", "press", "blog", "announce", "launch",
    "customer", "case-study", "success", "partner",
    "contact", "demo", "trial",
]


class BrowserScraper:
    """
    Async context manager. Use it like:
        async with BrowserScraper() as scraper:
            result = await scraper.scrape("https://example.com")
    """

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._headless = os.getenv("HEADLESS_BROWSER", "true").lower() == "true"
        self._delay = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        logger.info(f"🌐 Browser started (headless={self._headless})")
        return self

    async def __aexit__(self, *args):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("🌐 Browser closed")

    async def scrape(self, url: str, timeout_ms: int = 15000) -> dict:
        """
        Scrape a single URL.
        Returns: {url, title, text, links, success, error}
        """
        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        # Block images, fonts, media — we only need text
        await page.route(
            "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3}",
            lambda route: route.abort(),
        )

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await asyncio.sleep(1.5)  # Let dynamic content render

            title = await page.title()

            # ── IMPORTANT: extract links FIRST on the raw DOM ──────────
            # nav/header tags are stripped later for text extraction.
            # If we stripped them first, all navbar links (jobs, pricing,
            # about, etc.) would be permanently lost.
            links = await self._extract_links(page, url)

            # Now strip noise tags and extract clean text
            text = await self._extract_clean_text(page)

            await context.close()
            await asyncio.sleep(self._delay)

            logger.info(f"  ✅ Scraped: {url[:60]} ({len(text)} chars, {len(links)} links)")
            return {
                "url": url,
                "title": title,
                "text": text[:8000],   # Cap per page — enough for LLM analysis
                "links": links[:40],   # More links so LLM has real choice
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.warning(f"  ❌ Failed to scrape {url}: {e}")
            await context.close()
            return {
                "url": url,
                "title": "",
                "text": "",
                "links": [],
                "success": False,
                "error": str(e),
            }

    async def scrape_multiple(self, urls: list[str]) -> list[dict]:
        """Scrape a list of URLs sequentially (polite, avoids bans)."""
        results = []
        for url in urls:
            result = await self.scrape(url)
            results.append(result)
        return results

    async def _extract_clean_text(self, page: Page) -> str:
        """
        Remove noise elements, then extract visible text.
        Called AFTER _extract_links so nav/header links are already captured.
        """
        # Strip nav and header for clean text (already extracted links from them)
        all_noise = NOISE_TAGS + ["nav", "header", "button", "form"]
        noise_selector = ", ".join(all_noise)
        await page.evaluate(f"""
            document.querySelectorAll('{noise_selector}').forEach(el => el.remove());
        """)

        # Get remaining text
        text = await page.evaluate("""
            () => {
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null
                );
                const chunks = [];
                let node;
                while (node = walker.nextNode()) {
                    const t = node.textContent.trim();
                    if (t.length > 20) chunks.push(t);
                }
                return chunks.join(' ');
            }
        """)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def _extract_links(self, page: Page, base_url: str) -> list[str]:
        """
        Extract internal links from the raw DOM (before any tags are stripped).
        Links are deduplicated and ranked — high-value pages surface to the top.
        """
        base_domain = urlparse(base_url).netloc

        hrefs = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                       .map(a => a.href)
        """)

        # Skip patterns — noise links that are never useful
        _SKIP = [
            ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
            ".pdf", ".zip", "javascript:", "mailto:", "tel:",
            "privacy", "terms", "cookie", "legal", "refund", "dmca",
            "help/", "gp/help", "cdn-cgi", "#", "void(0",
        ]

        internal = []
        for href in hrefs:
            if not href:
                continue
            parsed = urlparse(href)
            # Only same-domain HTTP/S links
            if parsed.netloc != base_domain or parsed.scheme not in ("http", "https"):
                continue
            path = parsed.path.lower()
            if len(path) <= 1:
                continue
            if any(skip in path for skip in _SKIP):
                continue
            clean = href.split("#")[0]  # strip anchors
            if clean:
                internal.append(clean)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for link in internal:
            if link not in seen:
                seen.add(link)
                unique.append(link)

        # ── Smart ranking ─────────────────────────────────────────────
        # Links whose path contains a high-value keyword go to the top.
        # This ensures the LLM sees /jobs, /about, /pricing first —
        # not the 30th marketing page.
        def _rank(url: str) -> int:
            path = urlparse(url).path.lower()
            for i, kw in enumerate(_HIGH_VALUE_KEYWORDS):
                if kw in path:
                    return i  # lower = higher priority
            return len(_HIGH_VALUE_KEYWORDS)  # ungrouped go to end

        unique.sort(key=_rank)
        return unique