"""
SalesNeuron — Browser Scraper (Playwright + curl-cffi Cloudflare Bypass)
=========================================================================
Three-layer scraping strategy — tries each in order, stops on first success:

Layer 1 — Playwright + Stealth
  Full headless browser. Works on 70% of sites. Renders JavaScript so
  dynamic content (team pages, job listings) loads fully.

Layer 2 — curl-cffi (Cloudflare Bypass)
  Mimics Chrome's TLS fingerprint at the network level. Cloudflare and
  similar WAFs check this fingerprint — curl-cffi passes while Playwright
  fails. No JS rendering but gets past the wall.
  Triggered when: (a) bot signals detected, OR (b) content is empty/thin.

Layer 3 — URL-guessing fallback
  ResearcherAgent falls back to guessing common URL paths when both layers
  fail (/about, /careers, etc.)

CRITICAL — scrape() trigger logic:
  curl-cffi must trigger on BOTH:
    (a) bot signals in response text (Cloudflare challenge page detected)
    (b) empty or thin content (Playwright "succeeded" but returned nothing)
  The previous bug was returning early on success=True without checking
  whether we actually GOT any content — so consciousengines.com returned
  success=True with 0 chars and curl-cffi was never attempted.

CRITICAL — extraction order:
  Links MUST be extracted before nav/header/form/button are stripped for
  text cleaning. Getting this backwards silently deletes every navbar link
  (Jobs, Careers, Pricing, About). Keep _extract_links() BEFORE
  _extract_clean_text() inside _scrape_playwright().

CRITICAL — email extraction (added after finding two real data-loss bugs
that affected EVERY site scraped so far):
  1. mailto: links were silently filtered out of _extract_links() —
     that method only keeps http/https links matching the base domain,
     and a mailto: URI has no http scheme and no netloc, so it always
     failed that check. There was no separate path that ever captured it.
  2. _extract_clean_text()'s TreeWalker only kept text nodes with
     t.length > 20 characters. "info@blockxint.com" is 19 characters —
     a bare email sitting in its own short <span> (a very common
     contact-page pattern: icon + short text) was silently dropped
     before the LLM or any regex ever saw it. Not email-specific either
     — any short standalone text (phone numbers, taglines) was lost
     the same way.
  Both fixed via a dedicated _extract_emails() that (a) collects
  mailto: hrefs directly, unfiltered by domain/scheme, and (b) regexes
  the full raw page text for email patterns, independent of the 20-char
  node filter (which itself was also lowered). Same two fixes applied
  to the curl-cffi/_parse_html fallback path.
"""

import os
import asyncio
import logging
import random
import re
from typing import Optional
from urllib.parse import urlparse, unquote
from playwright.async_api import async_playwright, Page, Browser

logger = logging.getLogger(__name__)

# Matches any email address in text or a mailto: href.
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "noscript", "iframe", "svg", "form", "button", "aside",
]

THIN_CONTENT_THRESHOLD = int(os.getenv("THIN_CONTENT_THRESHOLD", "400"))
THIN_CONTENT_EXTRA_WAIT = float(os.getenv("THIN_CONTENT_EXTRA_WAIT", "3.5"))

# Minimum chars to consider a scrape "successful enough" without trying curl-cffi.
# If Playwright returns fewer chars than this, we always try curl-cffi regardless
# of whether success=True (this was the exact consciousengines.com bug).
MIN_USEFUL_CONTENT = int(os.getenv("MIN_USEFUL_CONTENT", "150"))

BOT_SIGNALS = [
    "captcha", "robot", "are you human", "unusual traffic",
    "verify you are human", "i'm not a robot",
    "enable javascript and cookies", "access denied",
    "please enable cookies", "checking your browser",
    "ddos protection", "just a moment", "attention required",
    "ray id",  # Cloudflare specific
]


class BrowserScraper:
    """
    Async context manager — always use as:
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
                "--disable-web-security",
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

    async def _restart_browser(self):
        """
        Restart the Playwright browser after a crash.
        Called automatically when 'Connection closed' is detected.
        """
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--window-size=1280,800",
                ],
            )
            logger.info("🌐 Browser restarted after crash")
        except Exception as e:
            logger.error(f"🌐 Browser restart failed: {e}")
            self._browser = None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def scrape(self, url: str, timeout_ms: int = 20000) -> dict:
        """
        Scrape a URL — Layer 1 (Playwright) first, then Layer 2 (curl-cffi).

        curl-cffi is triggered when ANY of these are true:
          (a) Playwright failed outright (exception, timeout)
          (b) Bot signals detected in response text (Cloudflare challenge)
          (c) Content is thin/empty (< MIN_USEFUL_CONTENT chars) —
              this was the consciousengines.com bug: success=True but 0 chars

        Browser disconnect / crash recovery:
          If the Playwright browser crashes mid-scrape
          ("Connection closed while reading from the driver"), we catch it,
          restart the browser automatically, and fall through to curl-cffi
          so we still return something instead of raising.
        """
        # Layer 1 — Playwright
        try:
            result = await self._scrape_playwright(url, timeout_ms)
        except Exception as e:
            error_str = str(e).lower()
            if "connection closed" in error_str or "browser" in error_str:
                logger.warning(
                    f"  💥 Browser crashed on {url[:50]} — restarting browser..."
                )
                await self._restart_browser()
            result = {
                "url": url, "title": "", "text": "", "links": [], "emails": [],
                "success": False, "error": str(e),
            }

        playwright_chars = len(result.get("text", ""))
        bot_blocked = self._is_bot_blocked(result.get("text", ""))

        # Decide whether curl-cffi is worth trying
        should_try_curl = (
            not result["success"]
            or bot_blocked
            or playwright_chars < MIN_USEFUL_CONTENT
        )

        if should_try_curl:
            reason = (
                "crashed" if not result["success"] and "crash" in result.get("error", "")
                else "failed" if not result["success"]
                else "bot-blocked" if bot_blocked
                else f"thin content ({playwright_chars} chars < {MIN_USEFUL_CONTENT})"
            )
            logger.info(f"  🔄 curl-cffi fallback for {url[:55]} — reason: {reason}")
            curl_result = await self._scrape_curl_cffi(url)
            curl_chars = len(curl_result.get("text", ""))

            if curl_result["success"] and curl_chars > playwright_chars:
                logger.info(
                    f"  ✅ curl-cffi won: {playwright_chars} → {curl_chars} chars"
                )
                # Merge emails from both — cheap to keep, and one layer
                # (JS-rendered mailto vs static-HTML mailto) may catch
                # something the other missed regardless of which text won.
                curl_result["emails"] = list(dict.fromkeys(
                    result.get("emails", []) + curl_result.get("emails", [])
                ))
                return curl_result
            elif curl_result["success"] and curl_chars > 0:
                curl_result["links"] = list(dict.fromkeys(
                    result.get("links", []) + curl_result["links"]
                ))
                merged_emails = list(dict.fromkeys(
                    result.get("emails", []) + curl_result.get("emails", [])
                ))
                if curl_chars >= playwright_chars:
                    curl_result["emails"] = merged_emails
                    return curl_result
                else:
                    result["emails"] = merged_emails
                    return result

        return result

    async def scrape_multiple(self, urls: list[str]) -> list[dict]:
        """
        Scrape a list of URLs sequentially with polite delays.
        Per-URL crashes (browser disconnect, timeout) are caught and
        returned as failed results — one bad URL cannot kill the session.
        """
        results = []
        for url in urls:
            try:
                result = await self.scrape(url)
            except Exception as e:
                logger.warning(f"  💥 Unhandled crash scraping {url}: {e}")
                result = {
                    "url": url, "title": "", "text": "", "links": [], "emails": [],
                    "success": False, "error": f"crash: {e}",
                }
            results.append(result)
        return results

    # ──────────────────────────────────────────────────────────────
    # Layer 1 — Playwright
    # ──────────────────────────────────────────────────────────────

    async def _scrape_playwright(self, url: str, timeout_ms: int) -> dict:
        """Full browser scrape with stealth and thin-content retry."""
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
        await self._apply_stealth(page)

        # Block media only — keep CSS/JS so frameworks render
        await page.route(
            "**/*.{png,jpg,jpeg,gif,webp,woff,woff2,ttf,mp4,mp3,avi}",
            lambda route: route.abort(),
        )

        try:
            # networkidle waits for JS frameworks to finish rendering
            try:
                await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            except Exception:
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            await asyncio.sleep(random.uniform(1.5, 3.0))
            await self._human_scroll(page)

            title = await page.title()

            # Links FIRST on untouched DOM — nav/header gets stripped
            # in _extract_clean_text. Must run before that.
            links = await self._extract_links(page, url)
            emails = await self._extract_emails(page, url)
            text = await self._extract_clean_text(page)

            # Thin-content retry — JS framework still rendering
            if len(text) < THIN_CONTENT_THRESHOLD:
                logger.debug(
                    f"  ⏳ Thin content ({len(text)} chars) — "
                    f"waiting {THIN_CONTENT_EXTRA_WAIT}s for JS..."
                )
                await asyncio.sleep(THIN_CONTENT_EXTRA_WAIT)
                await self._human_scroll(page)
                retried = await self._extract_clean_text(page)
                if len(retried) > len(text):
                    logger.debug(f"  ✅ JS retry: {len(text)} → {len(retried)} chars")
                    text = retried
                retried_links = await self._extract_links(page, url)
                if len(retried_links) > len(links):
                    links = retried_links
                retried_emails = await self._extract_emails(page, url)
                if retried_emails:
                    emails = list(dict.fromkeys(emails + retried_emails))

            await context.close()
            await asyncio.sleep(self._delay)

            logger.info(
                f"  {'✅' if len(text) >= MIN_USEFUL_CONTENT else '⚠️ '} "
                f"Playwright: {url[:55]} ({len(text)} chars, {len(links)} links, "
                f"{len(emails)} emails)"
            )
            return {
                "url": url, "title": title,
                "text": text[:8000], "links": links[:40], "emails": emails[:10],
                "success": True, "error": None,
            }

        except Exception as e:
            logger.warning(f"  ❌ Playwright failed for {url}: {e}")
            try:
                await context.close()
            except Exception:
                pass
            return {
                "url": url, "title": "", "text": "", "links": [], "emails": [],
                "success": False, "error": str(e),
            }

    # ──────────────────────────────────────────────────────────────
    # Layer 2 — curl-cffi Cloudflare bypass
    # ──────────────────────────────────────────────────────────────

    async def _scrape_curl_cffi(self, url: str) -> dict:
        """
        Use curl-cffi to mimic Chrome's TLS fingerprint.
        Bypasses Cloudflare, DataDome, and similar WAFs.
        No JavaScript rendering — gets raw HTML only.

        Install: pip install curl-cffi
        """
        try:
            from curl_cffi.requests import AsyncSession
        except ImportError:
            logger.warning("  ⚠️  curl-cffi not installed — run: pip install curl-cffi")
            return {
                "url": url, "title": "", "text": "", "links": [], "emails": [],
                "success": False, "error": "curl-cffi not installed",
            }

        try:
            async with AsyncSession(impersonate="chrome124") as session:
                response = await session.get(
                    url,
                    timeout=15,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                    },
                )

            if response.status_code not in (200, 201, 203):
                return {
                    "url": url, "title": "", "text": "", "links": [], "emails": [],
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                }

            html = response.text

            if self._is_bot_blocked(html):
                return {
                    "url": url, "title": "", "text": "", "links": [], "emails": [],
                    "success": False, "error": "Still bot-blocked after curl-cffi",
                }

            title, text, links, emails = self._parse_html(html, url)

            logger.info(
                f"  ✅ curl-cffi: {url[:55]} ({len(text)} chars, {len(links)} links, "
                f"{len(emails)} emails)"
            )
            return {
                "url": url, "title": title,
                "text": text[:8000], "links": links[:40], "emails": emails[:10],
                "success": True, "error": None,
            }

        except Exception as e:
            logger.warning(f"  ❌ curl-cffi failed for {url}: {e}")
            return {
                "url": url, "title": "", "text": "", "links": [], "emails": [],
                "success": False, "error": str(e),
            }

    def _parse_html(self, html: str, base_url: str) -> tuple:
        """
        Parse raw HTML without a browser.
        Returns (title, clean_text, links, emails).
        Used by curl-cffi since there's no Playwright page object.

        Same two fixes as the Playwright path: mailto: hrefs are
        captured separately (they fail the domain-match check that
        filters `links`), and the per-chunk length filter was lowered
        so short standalone text like a bare email isn't dropped.
        """
        try:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.title = ""
                    self.text_chunks = []
                    self.links = []
                    self.mailto_emails = []
                    self._in_title = False
                    self._skip = False
                    self._skip_tags = set(NOISE_TAGS)
                    self._skip_depth = 0

                def handle_starttag(self, tag, attrs):
                    if tag == "title":
                        self._in_title = True
                    if tag in self._skip_tags:
                        self._skip = True
                        self._skip_depth += 1
                    if tag == "a":
                        href = dict(attrs).get("href", "")
                        if href:
                            self.links.append(href)
                            if href.lower().startswith("mailto:"):
                                addr = href[7:].split("?")[0].strip()
                                addr = unquote(addr)
                                if EMAIL_REGEX.fullmatch(addr):
                                    self.mailto_emails.append(addr.lower())

                def handle_endtag(self, tag):
                    if tag == "title":
                        self._in_title = False
                    if tag in self._skip_tags:
                        self._skip_depth -= 1
                        if self._skip_depth <= 0:
                            self._skip = False
                            self._skip_depth = 0

                def handle_data(self, data):
                    if self._in_title:
                        self.title += data
                    elif not self._skip:
                        chunk = data.strip()
                        # Was `> 20` — dropped short standalone text like a
                        # bare email address in its own element. See the
                        # module docstring for the full explanation.
                        if len(chunk) > 1:
                            self.text_chunks.append(chunk)

            parser = TextExtractor()
            parser.feed(html)

            title = parser.title.strip()
            text = re.sub(r"\s+", " ", " ".join(parser.text_chunks)).strip()

            # Resolve and filter links
            base_domain = urlparse(base_url).netloc
            clean_links = []
            seen = set()
            for href in parser.links:
                if href.startswith("http"):
                    parsed = urlparse(href)
                elif href.startswith("/"):
                    parsed = urlparse(
                        f"{urlparse(base_url).scheme}://{base_domain}{href}"
                    )
                else:
                    continue

                if parsed.netloc == base_domain:
                    full = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    path = parsed.path.lower()
                    if (
                        full not in seen
                        and len(path) > 1
                        and not any(s in path for s in [
                            ".js", ".css", ".png", ".jpg", ".gif",
                            "privacy", "terms", "cookie",
                        ])
                    ):
                        seen.add(full)
                        clean_links.append(full)

            # Emails — mailto hrefs first (highest confidence), then
            # regex over the full raw HTML as a second independent pass
            # (catches bare emails with no mailto: link at all).
            base_domain_short = base_domain.replace("www.", "")
            regex_emails = [m.lower() for m in EMAIL_REGEX.findall(html)]
            all_emails = parser.mailto_emails + regex_emails
            email_seen = set()
            unique_emails = []
            for e in all_emails:
                if e not in email_seen:
                    email_seen.add(e)
                    unique_emails.append(e)
            domain_matches = [e for e in unique_emails if base_domain_short in e]
            other_matches = [e for e in unique_emails if base_domain_short not in e]
            emails = domain_matches + other_matches

            return title, text, clean_links, emails

        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return "", text[:8000], [], []

    # ──────────────────────────────────────────────────────────────
    # Bot detection
    # ──────────────────────────────────────────────────────────────

    def _is_bot_blocked(self, text: str) -> bool:
        """True if response contains Cloudflare or bot-wall signals."""
        if not text:
            return False
        text_lower = text.lower()
        return any(signal in text_lower for signal in BOT_SIGNALS)

    # ──────────────────────────────────────────────────────────────
    # Stealth helpers (Playwright only)
    # ──────────────────────────────────────────────────────────────

    async def _apply_stealth(self, page: Page):
        """Apply stealth patches to hide Playwright's automation fingerprint."""
        try:
            from playwright_stealth import Stealth
            await Stealth().apply_stealth_async(page)
            logger.debug("  🥷 playwright-stealth applied")
        except ImportError:
            pass
        except Exception:
            pass

        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin' },
                    { name: 'Chrome PDF Viewer' },
                    { name: 'Native Client' },
                ],
            });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = {
                runtime: {}, loadTimes: function() {},
                csi: function() {}, app: {}
            };
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters)
            );
        """)

    async def _human_scroll(self, page: Page):
        """Simulate human scrolling to trigger lazy-loaded content."""
        try:
            await page.evaluate("""
                () => new Promise(resolve => {
                    let scrolled = 0;
                    const total = Math.min(document.body.scrollHeight, 2000);
                    const step = () => {
                        scrolled += Math.random() * 150 + 50;
                        window.scrollTo(0, Math.min(scrolled, total));
                        if (scrolled < total) setTimeout(step, Math.random() * 100 + 50);
                        else { window.scrollTo(0, 0); resolve(); }
                    };
                    step();
                })
            """)
            await asyncio.sleep(0.5)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Text + link extraction (Playwright only)
    # ──────────────────────────────────────────────────────────────

    async def _extract_links(self, page: Page, base_url: str) -> list[str]:
        """
        Extract internal links from unmodified DOM.
        MUST run BEFORE _extract_clean_text() — that method strips
        nav/header/button from the DOM permanently.

        Note: mailto: hrefs are intentionally excluded here (they're not
        navigable pages) but are captured separately by _extract_emails().
        """
        base_domain = urlparse(base_url).netloc
        try:
            hrefs = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]'))
                           .map(a => a.href)
            """)
        except Exception:
            return []

        seen = set()
        unique = []
        for href in hrefs:
            if not href:
                continue
            parsed = urlparse(href)
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                path = parsed.path.lower()
                if (
                    len(path) > 1
                    and not any(s in path for s in [
                        ".js", ".css", ".png", ".jpg", ".gif",
                        "javascript:", "mailto:", "#",
                        "privacy", "terms", "cookie", "help/",
                    ])
                ):
                    clean = href.split("#")[0]
                    if clean not in seen:
                        seen.add(clean)
                        unique.append(clean)
        return unique

    async def _extract_emails(self, page: Page, base_url: str) -> list[str]:
        """
        Extract real email addresses from the page via two independent
        methods, since either one alone misses cases the other catches:

        1. mailto: hrefs — the most reliable source when present, but
           _extract_links() throws these away (they're not http/https
           pages). Collected here directly instead.
        2. Regex over the FULL raw page text — catches bare emails
           displayed as plain text with no mailto: link at all. Runs
           independently of _extract_clean_text()'s node-length filter,
           so a short standalone "info@company.com" span isn't lost.

        Domain-matching emails are returned first (highest confidence
        they're a real company inbox), followed by any other emails
        found on the page (lower confidence — could be a third party).
        """
        base_domain = urlparse(base_url).netloc.replace("www.", "")
        found: list[str] = []

        # Method 1 — mailto: hrefs directly
        try:
            mailto_hrefs = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href^="mailto:"]'))
                           .map(a => a.getAttribute('href'))
            """)
            for href in mailto_hrefs or []:
                addr = href.replace("mailto:", "").split("?")[0].strip()
                addr = unquote(addr)
                if EMAIL_REGEX.fullmatch(addr):
                    found.append(addr.lower())
        except Exception:
            pass

        # Method 2 — regex over full raw text (unfiltered by node length)
        try:
            raw_text = await page.evaluate("() => document.body.innerText || ''")
            found.extend(m.lower() for m in EMAIL_REGEX.findall(raw_text or ""))
        except Exception:
            pass

        # Dedupe, then rank domain-matching addresses first
        seen = set()
        unique = []
        for email in found:
            if email not in seen:
                seen.add(email)
                unique.append(email)

        domain_matches = [e for e in unique if base_domain in e]
        other_matches = [e for e in unique if base_domain not in e]
        return domain_matches + other_matches

    async def _extract_clean_text(self, page: Page) -> str:
        """
        Strip noise tags then extract visible text.
        Call AFTER _extract_links() — strips nav/header permanently.
        """
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
                        document.body, NodeFilter.SHOW_TEXT, null
                    );
                    const chunks = [];
                    let node;
                    while (node = walker.nextNode()) {
                        const t = node.textContent.trim();
                        // Was `t.length > 20` — that silently dropped short
                        // but meaningful standalone text (a bare email in
                        // its own <span>, a phone number, a short tagline).
                        // "info@blockxint.com" is 19 chars and was being
                        // lost entirely. Lowered to only skip truly empty
                        // or single-character noise nodes.
                        if (t.length > 1) chunks.push(t);
                    }
                    return chunks.join(' ');
                }
            """)
            return re.sub(r"\s+", " ", text).strip()
        except Exception:
            return ""