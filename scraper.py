"""
SalesNeuron — Browser Scraper (Stealth Mode)
=============================================
Playwright async scraper with playwright-stealth applied.
Bypasses bot detection on Amazon, LinkedIn, and most major sites.

Changes from v1:
  - playwright-stealth applied to every new page
  - Human-like behavior: random delays, realistic viewport, mouse movement
  - Longer wait for networkidle on heavy JS sites
  - CAPTCHA detection with clear error message
  - Retry logic (up to 2 retries on failure)
"""

import os
import asyncio
import logging
import random
import re
from typing import Optional
from playwright.async_api import async_playwright, Page, Browser

logger = logging.getLogger(__name__)

NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "noscript", "iframe", "svg", "form", "button",
    "aside", "cookie", "advertisement",
]

# Sites that need extra wait time for JS rendering
HEAVY_JS_DOMAINS = ["amazon", "linkedin", "twitter", "instagram", "facebook"]


class BrowserScraper:

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
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--window-size=1280,800",
            ],
        )
        logger.info(f"🌐 Browser started (headless={self._headless}, stealth=ON)")
        return self

    async def __aexit__(self, *args):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("🌐 Browser closed")

    async def scrape(self, url: str, timeout_ms: int = 20000, retries: int = 2) -> dict:
        """
        Scrape a single URL with stealth mode and retry logic.
        Returns: {url, title, text, links, success, error}
        """
        last_error = None

        for attempt in range(retries + 1):
            if attempt > 0:
                wait = random.uniform(3, 6)
                logger.info(f"  🔄 Retry {attempt}/{retries} for {url[:50]} (waiting {wait:.1f}s)")
                await asyncio.sleep(wait)

            try:
                result = await self._scrape_once(url, timeout_ms)
                if result["success"]:
                    return result
                last_error = result["error"]
            except Exception as e:
                last_error = str(e)

        return {
            "url": url,
            "title": "",
            "text": "",
            "links": [],
            "success": False,
            "error": last_error,
        }

    async def _scrape_once(self, url: str, timeout_ms: int) -> dict:
        """Single scrape attempt with full stealth setup."""

        # Randomize viewport slightly — real users don't all use 1280x800
        width = random.randint(1240, 1400)
        height = random.randint(760, 900)

        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": width, "height": height},
            locale="en-US",
            timezone_id="America/New_York",
            # Realistic browser headers
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        )

        page = await context.new_page()

        # ── Apply stealth patches ──────────────────────────────────
        await self._apply_stealth(page)

        # Block only media files — keep CSS/JS for proper rendering
        await page.route(
            "**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf,mp4,mp3,avi}",
            lambda route: route.abort(),
        )

        try:
            # Use networkidle for heavy JS sites, domcontentloaded for others
            domain = url.split("/")[2] if "/" in url else url
            is_heavy = any(d in domain for d in HEAVY_JS_DOMAINS)
            wait_until = "networkidle" if is_heavy else "domcontentloaded"

            await page.goto(url, wait_until=wait_until, timeout=timeout_ms)

            # Human-like random delay
            await asyncio.sleep(random.uniform(1.5, 3.0))

            # Simulate human scrolling — triggers lazy-load content
            await self._human_scroll(page)

            title = await page.title()
            text = await self._extract_clean_text(page)
            links = await self._extract_links(page, url)

            # Detect if we hit a CAPTCHA or bot wall
            bot_detected = await self._detect_bot_wall(page, text)
            if bot_detected:
                logger.warning(f"  🤖 Bot detection triggered on {url[:50]}")
                await context.close()
                await asyncio.sleep(self._delay)
                return {
                    "url": url,
                    "title": title,
                    "text": "",
                    "links": links,
                    "success": False,
                    "error": "bot_detection",
                }

            await context.close()
            await asyncio.sleep(self._delay)

            logger.info(f"  ✅ Scraped: {url[:60]} ({len(text)} chars)")
            return {
                "url": url,
                "title": title,
                "text": text[:8000],
                "links": links[:20],
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.warning(f"  ❌ Failed to scrape {url}: {e}")
            try:
                await context.close()
            except Exception:
                pass
            return {
                "url": url,
                "title": "",
                "text": "",
                "links": [],
                "success": False,
                "error": str(e),
            }

    async def scrape_multiple(self, urls: list[str]) -> list[dict]:
        """Scrape a list of URLs sequentially with polite delays."""
        results = []
        for url in urls:
            result = await self.scrape(url)
            results.append(result)
        return results

    # ──────────────────────────────────────────────────────────────
    # Stealth patches
    # ──────────────────────────────────────────────────────────────

    async def _apply_stealth(self, page: Page):
        """
        Apply all stealth patches to hide Playwright's automation fingerprint.
        Combines playwright-stealth library + manual JS patches.
        """
        # playwright-stealth library
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
            logger.debug("  🥷 playwright-stealth applied")
        except ImportError:
            logger.warning("  ⚠️  playwright-stealth not installed — running without it")

        # Additional manual patches on top of stealth library
        await page.add_init_script("""
            // Remove webdriver flag — the #1 bot detection signal
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });

            // Fake realistic plugin list
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin' },
                    { name: 'Chrome PDF Viewer' },
                    { name: 'Native Client' },
                ],
            });

            // Fake language
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });

            // Fix chrome object — headless doesn't have it
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {},
            };

            // Realistic screen dimensions
            Object.defineProperty(screen, 'availWidth',  { get: () => 1280 });
            Object.defineProperty(screen, 'availHeight', { get: () => 800 });

            // Prevent notification permission detection
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters)
            );
        """)

    async def _human_scroll(self, page: Page):
        """Simulate human scrolling behavior."""
        try:
            await page.evaluate("""
                () => new Promise(resolve => {
                    let scrolled = 0;
                    const total = Math.min(document.body.scrollHeight, 1500);
                    const step = () => {
                        scrolled += Math.random() * 150 + 50;
                        window.scrollTo(0, Math.min(scrolled, total));
                        if (scrolled < total) {
                            setTimeout(step, Math.random() * 100 + 50);
                        } else {
                            window.scrollTo(0, 0);
                            resolve();
                        }
                    };
                    step();
                })
            """)
            await asyncio.sleep(0.5)
        except Exception:
            pass  # Non-critical

    async def _detect_bot_wall(self, page: Page, text: str) -> bool:
        """
        Detect common bot detection / CAPTCHA patterns.
        Returns True if we're blocked.
        """
        # Check page text for bot wall signals
        bot_signals = [
            "captcha",
            "robot",
            "are you human",
            "unusual traffic",
            "verify you are human",
            "i'm not a robot",
            "enable javascript and cookies",
            "access denied",
            "403 forbidden",
            "please enable cookies",
            "checking your browser",
            "ddos protection",
            "cloudflare",
            "just a moment",  # Cloudflare challenge
        ]

        text_lower = text.lower()
        title_lower = (await page.title()).lower()

        for signal in bot_signals:
            if signal in text_lower or signal in title_lower:
                return True

        # Check if page has almost no content (bot walls return minimal text)
        if len(text) < 200:
            # Could be bot wall or just an empty page — check URL
            current_url = page.url
            if "challenge" in current_url or "captcha" in current_url:
                return True

        return False

    # ──────────────────────────────────────────────────────────────
    # Text extraction
    # ──────────────────────────────────────────────────────────────

    async def _extract_clean_text(self, page: Page) -> str:
        noise_selector = ", ".join(NOISE_TAGS)
        try:
            await page.evaluate(f"""
                document.querySelectorAll('{noise_selector}').forEach(el => el.remove());
            """)
        except Exception:
            pass

        try:
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
            return re.sub(r"\s+", " ", text).strip()
        except Exception:
            return ""

    async def _extract_links(self, page: Page, base_url: str) -> list[str]:
        from urllib.parse import urlparse
        base_domain = urlparse(base_url).netloc

        try:
            hrefs = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                           .map(a => a.href)
            """)
        except Exception:
            return []

        internal = []
        for href in hrefs:
            if not href:
                continue
            from urllib.parse import urlparse
            parsed = urlparse(href)
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                path = parsed.path.lower()
                if (
    len(path) > 1  # anything that's not just "/"
    and not any(skip in path for skip in [
        ".js", ".css", ".png", ".jpg", ".gif",
        "javascript:", "mailto:", "#",
        "affiliate", "privacy", "terms", "cookie",
        "help/", "gp/help",
    ])
):
                    internal.append(href.split("#")[0])

        seen = set()
        unique = []
        for link in internal:
            if link not in seen:
                seen.add(link)
                unique.append(link)
        return unique