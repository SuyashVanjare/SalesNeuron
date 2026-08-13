"""
SalesNeuron — LinkedIn People Finder (Apify + Hardened Fallback)
====================================================================
Finds founder/CEO names for companies using LinkedIn data.
Four strategies, tried in order — stops at the first that returns results:

Strategy 0 — Apify Google Search Scraper (most reliable, paid)
  Runs Apify's maintained `apify/google-search-scraper` actor instead of
  us hand-scraping Google's raw HTML. Apify handles proxy rotation and
  CAPTCHA-solving on their end, so results come back as clean structured
  JSON ({title, url, description}) rather than something we have to
  regex out of a raw HTML blob. Only runs if APIFY_API_TOKEN is set —
  skips gracefully otherwise, no hard dependency.

Strategy 1 — Web search → LinkedIn (safe, free, no LinkedIn login)
  Tries DuckDuckGo HTML first (never CAPTCHAs at our volume), then
  Google as a secondary pass via curl-cffi (Chrome TLS fingerprint —
  plain httpx's handshake is a known automation signature). Tries
  multiple name variants per company (legal name, name with
  Ltd/Inc/Limited stripped, domain-derived name) since a company's
  LinkedIn display name rarely matches its legal/website name exactly.

Strategy 2 — LinkedIn Company Page (public, no login)
  Scrapes linkedin.com/company/slug/people using curl-cffi.
  LinkedIn's company pages are partially public. Human-like delays
  (3-8s) + randomized behavior to avoid detection.

Strategy 3 — Session-based (if cookies saved)
  Uses saved LinkedIn session cookies from core/session_store.py.
  Login once manually via run_navigator.py, cookies reused for
  SESSION_MAX_AGE_HOURS. Most powerful — accesses full profile data.

Why Apify goes first despite being paid:
  It's the only strategy that doesn't depend on us successfully evading
  detection — Apify's actors are built and maintained specifically to
  handle that, so the success rate is materially higher than anything
  we can reliably do with raw HTTP requests. Everything below it is
  free fallback for when there's no Apify token configured, or Apify's
  own credits/quota run out for the month.

Setup for Strategy 0:
  1. Sign up at apify.com (free tier includes some monthly credit)
  2. Get your API token from Settings → Integrations
  3. Add to .env:  APIFY_API_TOKEN=apify_api_xxxxxxxx
  If this isn't set, Strategy 0 is skipped silently and Strategy 1 runs.
"""

import asyncio
import html as html_module
import json
import logging
import os
import random
import re
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
APIFY_GOOGLE_SEARCH_ACTOR = os.getenv(
    "APIFY_GOOGLE_SEARCH_ACTOR", "apify~google-search-scraper"
)
APIFY_TIMEOUT = float(os.getenv("APIFY_TIMEOUT_SECONDS", "60"))

# How long to wait between LinkedIn requests (randomized)
MIN_DELAY = float(os.getenv("LINKEDIN_MIN_DELAY", "3.0"))
MAX_DELAY = float(os.getenv("LINKEDIN_MAX_DELAY", "8.0"))

# Shorter delay for search-engine requests (Google/DDG, not LinkedIn itself)
SEARCH_MIN_DELAY = float(os.getenv("LINKEDIN_SEARCH_MIN_DELAY", "1.0"))
SEARCH_MAX_DELAY = float(os.getenv("LINKEDIN_SEARCH_MAX_DELAY", "2.5"))

# Priority titles to look for
CEO_TITLES = [
    "ceo", "co-founder", "founder", "cofounder",
    "chief executive", "president", "cto", "coo",
    "managing director", "head of", "vp", "vice president",
]

# Legal-entity suffixes stripped when building name search variants —
# LinkedIn's display name is almost always the short form
# ("BlockX AI" not "BlockX AI Private Limited").
_LEGAL_SUFFIXES = r'\b(Limited|Ltd|Inc|LLC|Corp|Corporation|Pvt|Private)\b'

# Signals that a search request got blocked/CAPTCHA'd rather than
# genuinely returning zero results.
_SEARCH_BLOCK_SIGNALS = [
    "detected unusual traffic", "captcha", "recaptcha",
    "our systems have detected", "sorry, we're having trouble",
    "unusual traffic from your computer network",
    "solve this puzzle", "verify you're a human",
]


class LinkedInFinder:
    """
    Finds people (name + title) at a company using LinkedIn data.
    Returns list of dicts: [{name, title, linkedin_url, source}]
    """

    async def find_people(
        self,
        company_name: str,
        domain: str,
        max_results: int = 3,
    ) -> list[dict]:
        """
        Find key people at a company. Tries all strategies in order,
        starting with Apify (if configured), falling through to the
        free strategies. Returns as soon as any strategy finds results.
        """
        domain = domain.replace("www.", "").split("/")[0]
        company_slug = self._guess_linkedin_slug(company_name, domain)
        name_variants = self._build_name_variants(company_name, domain)

        logger.info(f"🔗 LinkedIn finder: {company_name} ({domain})")

        # Strategy 0 — Apify (most reliable, needs APIFY_API_TOKEN)
        if APIFY_API_TOKEN:
            people = await self._apify_google_search(name_variants, domain)
            if people:
                logger.info(f"🔗 Apify found {len(people)} people")
                return people[:max_results]
        else:
            logger.debug("🔗 APIFY_API_TOKEN not set — skipping Apify strategy")

        # Strategy 1 — free web search (never touches LinkedIn directly)
        people = await self._web_search(name_variants, domain)
        if people:
            logger.info(f"🔗 Web search found {len(people)} people")
            return people[:max_results]

        # Strategy 2 — LinkedIn company page (public)
        people = await self._scrape_company_page(company_slug, domain)
        if people:
            logger.info(f"🔗 Company page found {len(people)} people")
            return people[:max_results]

        # Strategy 3 — Session cookies
        people = await self._session_search(company_name, domain)
        if people:
            logger.info(f"🔗 Session search found {len(people)} people")
            return people[:max_results]

        logger.info(f"🔗 No people found for {company_name}")
        return []

    # ──────────────────────────────────────────────────────────────
    # Strategy 0 — Apify Google Search Scraper
    # ──────────────────────────────────────────────────────────────

    async def _apify_google_search(
        self, name_variants: list[str], domain: str
    ) -> list[dict]:
        """
        Run Apify's apify/google-search-scraper actor synchronously and
        parse LinkedIn profile links out of its structured organic
        results. This replaces hand-scraping Google's HTML — Apify
        handles the anti-bot fight on their infrastructure, and we get
        back clean {title, url, description} objects instead of regex
        matches on a raw page dump.

        Uses the run-sync-get-dataset-items endpoint, which blocks until
        the actor finishes (or times out) and returns the dataset
        directly — no separate polling step needed for a single query.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("🔗 httpx not installed — skipping Apify strategy")
            return []

        queries = [
            f'site:linkedin.com/in "{v}" CEO OR founder OR "co-founder"'
            for v in name_variants[:2]
        ]
        queries.append(f'site:linkedin.com/in "{domain}" founder OR CEO')

        url = (
            f"https://api.apify.com/v2/acts/{APIFY_GOOGLE_SEARCH_ACTOR}"
            f"/run-sync-get-dataset-items"
        )

        try:
            async with httpx.AsyncClient(timeout=APIFY_TIMEOUT) as client:
                for query in queries[:2]:  # cap Apify credit usage per company
                    resp = await client.post(
                        url,
                        params={"token": APIFY_API_TOKEN},
                        json={
                            "queries": query,
                            "resultsPerPage": 10,
                            "maxPagesPerQuery": 1,
                            "countryCode": "us",
                            "languageCode": "en",
                        },
                    )

                    if resp.status_code not in (200, 201):
                        logger.debug(
                            f"🔗 Apify returned HTTP {resp.status_code}: "
                            f"{resp.text[:200]}"
                        )
                        continue

                    dataset = resp.json()
                    people = self._parse_apify_results(dataset)
                    if people:
                        return people

            return []

        except httpx.TimeoutException:
            logger.warning(
                f"🔗 Apify actor timed out after {APIFY_TIMEOUT}s — "
                f"falling back to free strategies"
            )
            return []
        except Exception as e:
            logger.warning(f"🔗 Apify search failed: {e}")
            return []

    def _parse_apify_results(self, dataset: list) -> list[dict]:
        """
        Parse Apify's google-search-scraper dataset items into people.
        Each dataset item has an "organicResults" list of
        {title, url, description}. Much cleaner than regex on raw HTML —
        the title field alone usually contains "Name - Title - LinkedIn".
        """
        people = []
        seen_names = set()

        for item in dataset:
            if not isinstance(item, dict):
                continue
            for result in item.get("organicResults", []):
                result_url = result.get("url", "")
                if "linkedin.com/in/" not in result_url:
                    continue

                slug_match = re.search(r'linkedin\.com/in/([\w\-]+)', result_url)
                if not slug_match:
                    continue
                slug = slug_match.group(1)

                # Apify's title field is typically "First Last - Title | LinkedIn"
                # — parse that directly instead of guessing from the slug when
                # possible, it's far more accurate than slug-derived names.
                title_field = result.get("title", "")
                name, title = self._parse_apify_title(title_field, slug)

                if not name or name.lower() in seen_names:
                    continue

                people.append({
                    "name": name,
                    "title": title or "Founder",
                    "linkedin_url": f"https://www.linkedin.com/in/{slug}",
                    "source": "apify_google_search",
                })
                seen_names.add(name.lower())

        people.sort(key=lambda p: self._title_priority(p.get("title", "")))
        return people

    def _parse_apify_title(self, title_field: str, slug: str) -> tuple[str, str]:
        """
        Google SERP titles for LinkedIn profiles are usually formatted
        "Name - Title - Company | LinkedIn" or "Name | LinkedIn". Split
        on common separators to pull out name + title cleanly; fall back
        to slug-derived name if the title field doesn't parse.
        """
        title_field = html_module.unescape(title_field or "")
        title_field = re.sub(r'\s*\|\s*LinkedIn\s*$', '', title_field).strip()

        parts = re.split(r'\s+[-–]\s+', title_field)
        if len(parts) >= 2:
            name = parts[0].strip()
            role = parts[1].strip()
            if name and len(name.split()) <= 4:
                return name, role

        # Fall back to slug-derived name
        return self._name_from_slug(slug), ""

    # ──────────────────────────────────────────────────────────────
    # Name-variant building (shared by Apify + free web search)
    # ──────────────────────────────────────────────────────────────

    def _build_name_variants(self, company_name: str, domain: str) -> list[str]:
        """
        Build multiple name variations to search — a company's LinkedIn
        display name rarely matches its legal/website name exactly
        (e.g. website says "BlockX AI Private Limited" but the LinkedIn
        page is under "BlockX AI" or just "BlockX").
        """
        short_name = re.sub(
            _LEGAL_SUFFIXES, '', company_name, flags=re.IGNORECASE
        ).strip()
        domain_word = domain.split(".")[0]

        return list(dict.fromkeys([
            company_name,   # exact as scraped
            short_name,     # without Ltd/Inc/Limited suffix
            domain_word,    # derived from the domain itself
        ]))

    # ──────────────────────────────────────────────────────────────
    # Strategy 1 — Free web search (DuckDuckGo, then Google)
    # ──────────────────────────────────────────────────────────────

    async def _web_search(
        self, name_variants: list[str], domain: str
    ) -> list[dict]:
        """
        Search for LinkedIn profiles via DuckDuckGo first, Google second.
        Both go through curl-cffi for a proper Chrome TLS fingerprint.
        Tries each name variant in turn, one retry with jittered backoff
        if the first full pass is blocked or comes back completely empty.
        """
        queries = [
            f'site:linkedin.com/in "{v}" CEO OR founder OR "co-founder"'
            for v in name_variants
        ]
        queries.append(f'site:linkedin.com/in "{domain}" founder OR CEO')

        for attempt in range(2):  # original try + 1 retry
            if attempt > 0:
                backoff = random.uniform(3.0, 6.0)
                logger.debug(f"🔗 Retrying web search after {backoff:.1f}s backoff...")
                await asyncio.sleep(backoff)

            for query in queries[:4]:  # cap total calls per company
                people = await self._search_duckduckgo(query)
                if people:
                    return people

                await asyncio.sleep(random.uniform(SEARCH_MIN_DELAY, SEARCH_MAX_DELAY))

                people = await self._search_google(query)
                if people:
                    return people

        return []

    async def _search_duckduckgo(self, query: str) -> list[dict]:
        """DuckDuckGo's HTML endpoint — static markup, no JS challenge."""
        try:
            from curl_cffi.requests import AsyncSession

            async with AsyncSession(impersonate="chrome124") as session:
                resp = await session.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    timeout=10,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )

            if resp.status_code != 200:
                logger.debug(f"🔗 DuckDuckGo returned HTTP {resp.status_code}")
                return []

            html = html_module.unescape(resp.text)
            if self._is_search_blocked(html):
                logger.debug("🔗 DuckDuckGo response looks blocked/CAPTCHA'd")
                return []

            return self._extract_people_from_html(html)

        except ImportError:
            logger.debug("🔗 curl-cffi not installed — skipping DuckDuckGo search")
            return []
        except Exception as e:
            logger.debug(f"🔗 DuckDuckGo search failed: {e}")
            return []

    async def _search_google(self, query: str) -> list[dict]:
        """Google search fallback via curl-cffi, with block detection."""
        try:
            from curl_cffi.requests import AsyncSession

            async with AsyncSession(impersonate="chrome124") as session:
                resp = await session.get(
                    "https://www.google.com/search",
                    params={"q": query, "num": 5, "hl": "en"},
                    timeout=10,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )

            if resp.status_code != 200:
                return []

            html = html_module.unescape(resp.text)
            if self._is_search_blocked(html):
                logger.debug(f"🔗 Google search appears blocked/CAPTCHA'd")
                return []

            return self._extract_people_from_html(html)

        except ImportError:
            logger.debug("🔗 curl-cffi not installed — skipping Google search")
            return []
        except Exception as e:
            logger.debug(f"🔗 Google search failed: {e}")
            return []

    def _extract_people_from_html(self, html: str) -> list[dict]:
        """
        Parse LinkedIn profile links + nearby titles out of search-result
        HTML (works for both DuckDuckGo and Google markup). Unwraps
        Google's /url?q=... redirect format in addition to direct links.
        """
        people = []
        seen_names = set()

        direct_slugs = re.findall(r'linkedin\.com/in/([\w\-]+)', html, re.IGNORECASE)

        redirect_matches = re.findall(
            r'/url\?q=([^"&]+linkedin\.com%2Fin%2F[\w\-%]+)', html
        )
        redirect_slugs = []
        for m in redirect_matches:
            slug_match = re.search(r'/in/([\w\-]+)', m)
            if slug_match:
                redirect_slugs.append(slug_match.group(1))

        all_slugs = list(dict.fromkeys(direct_slugs + redirect_slugs))

        for slug in all_slugs[:6]:
            name = self._name_from_slug(slug)
            if not name or name.lower() in seen_names:
                continue

            title = self._extract_title_near_url(html, slug)

            people.append({
                "name": name,
                "title": title or "Founder",
                "linkedin_url": f"https://www.linkedin.com/in/{slug}",
                "source": "web_search",
            })
            seen_names.add(name.lower())

        people.sort(key=lambda p: self._title_priority(p.get("title", "")))
        return people

    def _is_search_blocked(self, html: str) -> bool:
        """Detect CAPTCHA / block pages from a search engine response."""
        if not html:
            return False
        lower = html.lower()
        return any(signal in lower for signal in _SEARCH_BLOCK_SIGNALS)

    def _name_from_slug(self, slug: str) -> str:
        """Convert linkedin slug like 'john-doe-123' to 'John Doe'."""
        slug = re.sub(r'-\d+$', '', slug)
        parts = slug.split("-")
        parts = [p.title() for p in parts if len(p) > 1 and not p.isdigit()]
        if len(parts) >= 2:
            return " ".join(parts[:3])
        return ""

    def _extract_title_near_url(self, html: str, slug: str) -> str:
        """Find job title in the text surrounding a LinkedIn URL in search results."""
        pos = html.lower().find(slug.lower())
        if pos == -1:
            return ""

        context = html[max(0, pos - 200):pos + 300]
        context = re.sub(r'<[^>]+>', ' ', context)
        context = re.sub(r'\s+', ' ', context).strip()

        for title_kw in CEO_TITLES:
            if title_kw in context.lower():
                match = re.search(
                    rf'([\w\s]{{0,20}}{title_kw}[\w\s]{{0,20}})',
                    context, re.IGNORECASE
                )
                if match:
                    return match.group(1).strip()[:50]
        return ""

    # ──────────────────────────────────────────────────────────────
    # Strategy 2 — LinkedIn Company Page (public, no login)
    # ──────────────────────────────────────────────────────────────

    async def _scrape_company_page(
        self, company_slug: str, domain: str
    ) -> list[dict]:
        """
        Scrape LinkedIn's public company/people page using curl-cffi.
        LinkedIn partially shows employees on public company pages.
        Uses human-like delays to avoid rate limiting.
        """
        if not company_slug:
            return []

        try:
            from curl_cffi.requests import AsyncSession

            url = f"https://www.linkedin.com/company/{company_slug}/people/"

            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            async with AsyncSession(impersonate="chrome124") as session:
                resp = await session.get(
                    url,
                    timeout=15,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Referer": "https://www.google.com/",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "cross-site",
                    },
                )

            if resp.status_code == 999:
                logger.debug("🔗 LinkedIn returned 999 (rate limited)")
                return []

            if resp.status_code not in (200, 302):
                return []

            html = resp.text
            people = self._parse_linkedin_people(html, domain)
            return people

        except ImportError:
            logger.debug("🔗 curl-cffi not installed — skipping LinkedIn company page")
            return []
        except Exception as e:
            logger.debug(f"🔗 LinkedIn company page failed: {e}")
            return []

    def _parse_linkedin_people(self, html: str, domain: str) -> list[dict]:
        """Parse people from LinkedIn company page HTML."""
        people = []
        seen = set()

        json_blocks = re.findall(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            html, re.DOTALL
        )
        for block in json_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict) and data.get("@type") == "Person":
                    name = data.get("name", "")
                    title = data.get("jobTitle", "")
                    li_url = data.get("url", "")
                    if name and name not in seen:
                        people.append({
                            "name": name, "title": title,
                            "linkedin_url": li_url, "source": "linkedin_company",
                        })
                        seen.add(name)
            except Exception:
                continue

        name_matches = re.findall(
            r'"(?:full[-_]?name|name)"\s*:\s*"([A-Z][a-z]+ [A-Z][a-z]+)"', html
        )
        title_matches = re.findall(
            r'"(?:title|headline|occupation)"\s*:\s*"([^"]{5,80})"', html
        )

        for i, name in enumerate(name_matches[:5]):
            if name not in seen:
                title = title_matches[i] if i < len(title_matches) else ""
                people.append({
                    "name": name, "title": title,
                    "linkedin_url": "", "source": "linkedin_company",
                })
                seen.add(name)

        people.sort(key=lambda p: self._title_priority(p.get("title", "")))
        return people

    # ──────────────────────────────────────────────────────────────
    # Strategy 3 — Session cookies (if user has logged in via Navigator)
    # ──────────────────────────────────────────────────────────────

    async def _session_search(
        self, company_name: str, domain: str
    ) -> list[dict]:
        """
        Use saved LinkedIn session cookies for authenticated search.
        Cookies are saved via core/session_store.py's load(), populated
        when the user logs in through run_navigator.py against
        linkedin.com — same mechanism used for Internshala/YC auth.
        """
        try:
            from core.session_store import session_store
            cookies = await session_store.load("linkedin.com")
            if not cookies:
                logger.debug("🔗 No LinkedIn session — skipping session search")
                return []

            from curl_cffi.requests import AsyncSession

            query = quote_plus(f"{company_name} founder CEO")
            url = (
                f"https://www.linkedin.com/search/results/people/"
                f"?keywords={query}&origin=GLOBAL_SEARCH_HEADER"
            )

            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            async with AsyncSession(impersonate="chrome124") as session:
                for cookie in cookies:
                    session.cookies.set(
                        cookie.get("name", ""),
                        cookie.get("value", ""),
                        domain=".linkedin.com",
                    )

                resp = await session.get(url, timeout=15, headers={
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.linkedin.com/feed/",
                })

            if resp.status_code == 999:
                logger.debug("🔗 LinkedIn session search rate-limited (999)")
                return []
            if resp.status_code != 200:
                return []

            return self._parse_linkedin_people(resp.text, domain)

        except Exception as e:
            logger.debug(f"🔗 Session search failed: {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _guess_linkedin_slug(self, company_name: str, domain: str) -> str:
        """
        Guess the LinkedIn company slug from company name or domain.
        e.g. "ArmorIQ" → "armoriq", "Conscious Engines" → "conscious-engines"
        """
        slug = domain.split(".")[0].lower()
        slug = re.sub(r'[^a-z0-9\-]', '-', slug)
        slug = re.sub(r'-+', '-', slug).strip('-')
        return slug

    def _title_priority(self, title: str) -> int:
        """Lower = higher priority. CEO/Founder first."""
        title_lower = (title or "").lower()
        for i, kw in enumerate(CEO_TITLES):
            if kw in title_lower:
                return i
        return 99


# Singleton
linkedin_finder = LinkedInFinder()