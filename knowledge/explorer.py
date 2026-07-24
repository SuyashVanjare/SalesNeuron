"""
SalesNeuron — Site Explorer Agent
===================================
Visits a website and builds a complete knowledge graph of how it works.

How it works:
  1. SEED     — start from the homepage
  2. ANALYZE  — LLM looks at the page, identifies every interactive element
  3. TRAVERSE — follow important links to discover more page types
  4. SYNTHESIZE — LLM builds named flows (search, login, fill form, etc.)
  5. STORE    — save graph to SQLite

The result: any future agent can navigate this site without re-exploring.
It looks up "how do I search on amazon.com?" and gets exact CSS selectors
and step-by-step actions — no LLM needed for navigation itself.

Usage:
    explorer = SiteExplorer()
    graph = await explorer.learn("https://amazon.com")
"""

import asyncio
import json
import logging
import re
from urllib.parse import urlparse, urljoin

from core.llm import llm
from core.scraper import BrowserScraper
from knowledge.models import (
    SiteGraph, PageNode, NavigationEdge, NavigationFlow, InteractiveElement
)

logger = logging.getLogger(__name__)

# How many pages to explore per site (more = better map, more time)
MAX_EXPLORE_PAGES = int(__import__("os").getenv("MAX_EXPLORE_PAGES", "6"))


class SiteExplorer:
    """
    Learns a website's structure by browsing it and asking the LLM
    to interpret what it sees.
    """

    async def learn(self, url: str) -> SiteGraph:
        """
        Full site learning pipeline.
        Phase 1: SEED        — start from the homepage
        Phase 2: TRAVERSE    — BFS that prioritises high-value sections
                               then drills into individual items
        Phase 3: ANALYZE     — LLM interprets each page
        Phase 4: SYNTHESIZE  — LLM builds flows from the full page set
        """
        url = url.rstrip("/")
        domain = urlparse(url).netloc.replace("www.", "")

        logger.info(f"\n{'='*55}")
        logger.info(f"🗺️  SiteExplorer: learning {domain}")
        logger.info(f"{'='*55}")

        async with BrowserScraper() as browser:

            # ── Phase 1: SEED ─────────────────────────────────────
            logger.info("🌱 Phase 1/4 — Seeding from homepage...")
            homepage = await browser.scrape(url)
            if not homepage["success"]:
                raise RuntimeError(f"Could not load {url} — check the URL")

            # ── Phase 2: TRAVERSE ────────────────────────────────
            logger.info("📋 Phase 2/4 — Traversing site graph...")
            explore_urls = await self._traverse(url, homepage, browser)
            logger.info(f"   Will analyze {len(explore_urls)} pages")
            for u in explore_urls:
                logger.info(f"   → {u}")

            # ── Phase 3: ANALYZE each page ────────────────────────
            logger.info("🔬 Phase 3/4 — Analyzing pages...")
            pages: list[PageNode] = []
            all_scraped = []

            for page_url in explore_urls:
                logger.info(f"   Analyzing: {page_url[:70]}")
                scraped = await browser.scrape(page_url)
                if not scraped["success"]:
                    logger.warning(f"   ❌ Failed to scrape {page_url[:60]}")
                    continue

                all_scraped.append(scraped)

                # Get raw HTML elements for this page
                elements_raw = await self._extract_elements_from_browser(browser, page_url)

                # LLM interprets the page
                page_node = await self._analyze_page(page_url, scraped, elements_raw)
                pages.append(page_node)
                logger.info(
                    f"   ✅ {page_node.page_type}: {page_node.purpose[:60]}"
                )

        # ── Phase 4: SYNTHESIZE flows ───────────────────────────
        logger.info("🧠 Phase 4/4 — Synthesizing navigation flows...")
        graph = await self._synthesize_graph(url, domain, pages, all_scraped)

        logger.info(f"✅ Site graph complete:")
        logger.info(f"   Pages mapped:  {len(graph.pages)}")
        logger.info(f"   Edges found:   {len(graph.edges)}")
        logger.info(f"   Flows built:   {len(graph.flows)}")

        return graph

    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────
    # Phase 2 — BFS Traversal
    # ──────────────────────────────────────────────────────────────

    # Keywords that indicate a high-value section page.
    # Lower index = higher priority in traversal.
    _SECTION_KEYWORDS: list[str] = [
        # Hiring (highest value for SalesNeuron)
        "job", "jobs", "career", "careers", "hiring", "internship", "internships",
        "opening", "openings", "vacancy", "vacancies", "work-with-us", "work-at",
        # Team & people
        "team", "people", "leadership", "founders", "about",
        # Product & commercial
        "product", "products", "features", "solutions", "platform", "pricing", "plans",
        # Social proof
        "customer", "customers", "case-study", "case_study", "success", "partner",
        # News & content
        "news", "press", "blog", "announce", "launch",
        # Contact
        "contact", "demo", "trial",
    ]

    # Keywords that indicate this is a LEAF/individual page (not a section)
    # — worth scraping as a child of a section.
    _LEAF_INDICATORS: list[str] = [
        "job", "career", "internship", "opening",   # individual job postings
        "post", "article", "story", "press-release",  # individual blog/news
        "product",                                    # individual product page
    ]

    def _link_priority(self, url: str) -> int:
        """
        Score a URL by how valuable it is to explore.
        Lower number = explore sooner.
        Returns len(_SECTION_KEYWORDS) for unrecognised URLs (lowest priority).
        """
        path = urlparse(url).path.lower()
        for i, kw in enumerate(self._SECTION_KEYWORDS):
            if kw in path:
                return i
        return len(self._SECTION_KEYWORDS)

    async def _traverse(
        self, base_url: str, homepage: dict, browser: BrowserScraper
    ) -> list[str]:
        """
        Two-phase BFS traversal:

        Phase A — Section discovery:
          Start from all homepage links. Pick the highest-priority sections
          (Careers, Products, Team, News, etc.) up to MAX_EXPLORE_PAGES//2.

        Phase B — Deep item discovery:
          For each section page found, scrape it and extract 1-2 sample
          child pages (individual job descriptions, product detail, etc.).
          These are the pages that contain actual intel.

        The result is a flat, prioritised list for Phase 3 analysis.
        """
        max_total = MAX_EXPLORE_PAGES
        visited: set[str] = {base_url}
        result: list[str] = [base_url]

        # ── Phase A: discover section pages from homepage ──────────────
        homepage_links = homepage.get("links", [])

        # If homepage gave no links, fall back to guessed paths
        if not homepage_links:
            logger.info("   No links on homepage, using domain-aware fallback")
            for p in self._guess_key_pages(base_url):
                if len(result) < max_total and p not in visited:
                    result.append(p)
                    visited.add(p)
            return result

        # Sort by priority so highest-value sections come first
        ranked = sorted(
            [l for l in homepage_links if l not in visited],
            key=self._link_priority,
        )

        # Slots: leave half for deep items
        section_budget = max(1, max_total // 2)
        sections: list[str] = []
        for link in ranked:
            if len(sections) >= section_budget:
                break
            # Only take links that match at least one section keyword
            if self._link_priority(link) < len(self._SECTION_KEYWORDS):
                sections.append(link)
                visited.add(link)

        # If we didn't find enough section pages, fill with top-ranked links
        for link in ranked:
            if len(sections) >= section_budget:
                break
            if link not in visited:
                sections.append(link)
                visited.add(link)

        result.extend(sections)
        logger.info(f"   Phase A: {len(sections)} section pages selected")

        # ── Phase B: deep item discovery ──────────────────────────
        # For each section page, scrape it and extract 1-2 child URLs.
        item_budget = max_total - len(result)
        items_found = 0

        for section_url in sections:
            if items_found >= item_budget:
                break

            section_path = urlparse(section_url).path.lower().rstrip("/")

            # Only drill into pages that are likely listing pages
            if not any(kw in section_path for kw in self._LEAF_INDICATORS + [
                "job", "career", "internship", "product", "blog", "news",
                "press", "customer", "success", "announce",
            ]):
                continue

            try:
                section_scraped = await browser.scrape(section_url)
                if not section_scraped["success"]:
                    continue

                child_links = [
                    link for link in section_scraped.get("links", [])
                    if link not in visited
                    and urlparse(link).path.lower().startswith(section_path)
                    and len(urlparse(link).path) > len(section_path) + 2
                ]

                for child in child_links[:2]:
                    if items_found >= item_budget:
                        break
                    result.append(child)
                    visited.add(child)
                    items_found += 1
                    logger.info(f"   Phase B: 📄 child page: {child[:65]}")

            except Exception as e:
                logger.warning(f"   Deep traversal failed for {section_url}: {e}")

        logger.info(f"   Phase B: {items_found} child pages discovered")
        return result

    async def _plan_exploration(
        self, base_url: str, homepage: dict, browser: BrowserScraper
    ) -> list[str]:
        """
        Legacy wrapper — kept for any callers that use the old API.
        Delegates to _traverse().
        """
        return await self._traverse(base_url, homepage, browser)

    # ──────────────────────────────────────────────────────────────
    # Phase 3a — Extract raw elements from live browser page
    # ──────────────────────────────────────────────────────────────

    async def _extract_elements_from_browser(
        self, browser: BrowserScraper, url: str
    ) -> list[dict]:
        """
        Use Playwright to extract all interactive elements with their
        CSS selectors, types, and text — directly from the live DOM.
        This is more reliable than asking LLM to guess selectors.
        """
        async with BrowserScraper() as b:
            context = await b._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(1.5)

                # Extract all interactive elements via JavaScript
                elements = await page.evaluate("""
                    () => {
                        const results = [];
                        const seen = new Set();

                        const getSelector = (el) => {
                            // Build a reliable CSS selector for this element
                            if (el.id) return '#' + el.id;
                            if (el.name) return `[name="${el.name}"]`;
                            if (el.getAttribute('data-testid'))
                                return `[data-testid="${el.getAttribute('data-testid')}"]`;
                            if (el.getAttribute('aria-label'))
                                return `[aria-label="${el.getAttribute('aria-label')}"]`;

                            // Fall back to class-based selector
                            const tag = el.tagName.toLowerCase();
                            const classes = Array.from(el.classList)
                                .filter(c => !c.match(/^(css|sc-|_)/))
                                .slice(0, 2)
                                .join('.');
                            return classes ? `${tag}.${classes}` : tag;
                        };

                        const getLabel = (el) => {
                            return (
                                el.innerText?.trim().slice(0, 60) ||
                                el.placeholder ||
                                el.getAttribute('aria-label') ||
                                el.getAttribute('title') ||
                                el.getAttribute('alt') ||
                                el.name ||
                                el.type ||
                                ''
                            );
                        };

                        // Buttons
                        document.querySelectorAll('button, [role="button"]').forEach(el => {
                            const label = getLabel(el);
                            if (!label || seen.has(label)) return;
                            seen.add(label);
                            results.push({
                                type: 'button',
                                selector: getSelector(el),
                                label: label,
                                visible: el.offsetParent !== null
                            });
                        });

                        // Input fields
                        document.querySelectorAll('input, textarea').forEach(el => {
                            const label = getLabel(el) || el.type;
                            const key = el.type + ':' + label;
                            if (seen.has(key)) return;
                            seen.add(key);
                            results.push({
                                type: 'input',
                                input_type: el.type,
                                selector: getSelector(el),
                                label: label,
                                visible: el.offsetParent !== null
                            });
                        });

                        // Select dropdowns
                        document.querySelectorAll('select').forEach(el => {
                            const label = getLabel(el);
                            results.push({
                                type: 'select',
                                selector: getSelector(el),
                                label: label,
                                options: Array.from(el.options).map(o => o.text).slice(0, 5)
                            });
                        });

                        // Important links (nav, CTA)
                        document.querySelectorAll('nav a, header a, [class*="cta"] a, [class*="nav"] a').forEach(el => {
                            const label = getLabel(el);
                            const href = el.href;
                            if (!label || !href || seen.has(href)) return;
                            seen.add(href);
                            results.push({
                                type: 'link',
                                selector: getSelector(el),
                                label: label,
                                href: href
                            });
                        });

                        return results.slice(0, 40); // cap at 40 elements
                    }
                """)

                await context.close()
                return elements or []

            except Exception as e:
                logger.warning(f"Element extraction failed for {url}: {e}")
                await context.close()
                return []

    # ──────────────────────────────────────────────────────────────
    # Phase 3b — LLM analyzes a page and its elements
    # ──────────────────────────────────────────────────────────────

    async def _analyze_page(
        self, url: str, scraped: dict, elements_raw: list[dict]
    ) -> PageNode:
        """
        LLM reads the page text + extracted elements and builds a PageNode:
        - What type of page is this?
        - What is its purpose?
        - What can an agent DO on this page?
        """
        elements_text = json.dumps(elements_raw, indent=2)[:3000]
        page_text = scraped.get("text", "")[:4000]
        title = scraped.get("title", "")

        # Build url pattern (replace dynamic segments with regex)
        url_pattern = self._url_to_pattern(url)

        prompt = (
            "You are analyzing a web page to teach an AI agent how to navigate it.\n\n"
            f"URL: {url}\n"
            f"Title: {title}\n\n"
            f"PAGE TEXT:\n{page_text}\n\n"
            f"INTERACTIVE ELEMENTS FOUND:\n{elements_text}\n\n"
            "Analyze this page and return ONLY a JSON object:\n"
            "{\n"
            '  "page_type": "one of: homepage/search_results/product_page/cart/'
            'checkout/login/signup/form/listing/dashboard/pricing/article/other",\n'
            '  "purpose": "one sentence: what is this page for?",\n'
            '  "requires_auth": false,\n'
            '  "elements": [\n'
            '    {\n'
            '      "element_type": "button/input/link/select/form",\n'
            '      "selector": "the CSS selector from the elements list above",\n'
            '      "label": "human readable label",\n'
            '      "purpose": "what does interacting with this element do?",\n'
            '      "required": true\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Only include elements that are meaningful for task automation.\n"
            "Use the EXACT selectors from the extracted elements list.\n"
            "No markdown, pure JSON only."
        )

        try:
            result = await llm.generate_structured(prompt, "{}", temperature=0.1)
            return PageNode(
                url=url,
                url_pattern=url_pattern,
                page_type=result.get("page_type", "other"),
                title=title,
                purpose=result.get("purpose", ""),
                requires_auth=result.get("requires_auth", False),
                elements=[
                    InteractiveElement(**e)
                    for e in result.get("elements", [])
                    if isinstance(e, dict) and all(
                        k in e for k in ["element_type", "selector", "label", "purpose"]
                    )
                ],
                outgoing_urls=scraped.get("links", [])[:10],
            )
        except Exception as e:
            logger.warning(f"Page analysis failed for {url}: {e}")
            return PageNode(
                url=url,
                url_pattern=url_pattern,
                page_type="other",
                title=title,
                purpose="Analysis failed",
            )

    # ──────────────────────────────────────────────────────────────
    # Phase 4 — Synthesize full site graph + flows
    # ──────────────────────────────────────────────────────────────

    async def _synthesize_graph(
        self,
        base_url: str,
        domain: str,
        pages: list[PageNode],
        all_scraped: list[dict],
    ) -> SiteGraph:
        """
        Given all analyzed pages, ask the LLM to:
        1. Determine site type and overall description
        2. Build navigation edges between pages
        3. Create named flows (step-by-step task sequences)
        """
        # Summarize pages for the LLM
        pages_summary = []
        for p in pages:
            elements_summary = [
                f"{e.element_type}: '{e.label}' ({e.selector}) → {e.purpose}"
                for e in p.elements[:8]
            ]
            pages_summary.append(
                f"PAGE: {p.url}\n"
                f"TYPE: {p.page_type}\n"
                f"PURPOSE: {p.purpose}\n"
                f"ELEMENTS:\n" + "\n".join(f"  - {e}" for e in elements_summary)
            )

        pages_text = "\n\n".join(pages_summary)

        prompt = (
            "You are building a navigation knowledge graph for an AI agent.\n\n"
            f"WEBSITE: {base_url}\n\n"
            "PAGES ANALYZED:\n"
            + pages_text
            + "\n\n"
            "Build the site knowledge graph. Return ONLY a JSON object:\n"
            "{\n"
            '  "site_type": "ecommerce/saas/news/social/corporate/job_board/directory/other",\n'
            '  "description": "one sentence: what does this site do?",\n'
            '  "edges": [\n'
            '    {\n'
            '      "from_url_pattern": "regex pattern of source page",\n'
            '      "to_url_pattern": "regex pattern of destination page",\n'
            '      "action_type": "click/type/navigate/submit/select",\n'
            '      "selector": "CSS selector to interact with",\n'
            '      "input_value": "value to type, or null",\n'
            '      "description": "what this action does",\n'
            '      "wait_for": "CSS selector to wait for after action, or null",\n'
            '      "confidence": 0.9\n'
            '    }\n'
            '  ],\n'
            '  "flows": [\n'
            '    {\n'
            '      "flow_name": "machine_name_like_this",\n'
            '      "description": "what goal this flow accomplishes",\n'
            '      "variables": ["query"],\n'
            '      "success_indicator": "CSS selector visible on success",\n'
            '      "steps": [\n'
            '        {\n'
            '          "from_url_pattern": "source pattern",\n'
            '          "to_url_pattern": "destination pattern",\n'
            '          "action_type": "click/type/navigate/submit",\n'
            '          "selector": "CSS selector",\n'
            '          "input_value": "{query} or null",\n'
            '          "description": "step description",\n'
            '          "wait_for": "selector or null",\n'
            '          "confidence": 0.9\n'
            '        }\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Create flows for every distinct task a user might do on this site.\n"
            "Use {variable_name} syntax for dynamic values.\n"
            "No markdown, pure JSON only."
        )

        try:
            result = await llm.generate_structured(prompt, "{}", temperature=0.1)

            edges = [
                NavigationEdge(**e)
                for e in result.get("edges", [])
                if isinstance(e, dict)
                and all(k in e for k in [
                    "from_url_pattern", "to_url_pattern",
                    "action_type", "selector", "description"
                ])
            ]

            flows = []
            for f in result.get("flows", []):
                if not isinstance(f, dict):
                    continue
                try:
                    steps = [
                        NavigationEdge(**s)
                        for s in f.get("steps", [])
                        if isinstance(s, dict)
                        and all(k in s for k in [
                            "from_url_pattern", "to_url_pattern",
                            "action_type", "selector", "description"
                        ])
                    ]
                    flows.append(NavigationFlow(
                        flow_name=f.get("flow_name", "unnamed"),
                        description=f.get("description", ""),
                        variables=f.get("variables", []),
                        success_indicator=f.get("success_indicator"),
                        steps=steps,
                    ))
                except Exception:
                    continue

            return SiteGraph(
                domain=domain,
                base_url=base_url,
                site_type=result.get("site_type", "other"),
                description=result.get("description", ""),
                pages=pages,
                edges=edges,
                flows=flows,
                pages_explored=len(pages),
            )

        except Exception as e:
            logger.error(f"Graph synthesis failed: {e}")
            # Return minimal graph so pipeline doesn't crash
            return SiteGraph(
                domain=domain,
                base_url=base_url,
                site_type="other",
                description="Synthesis failed — partial graph",
                pages=pages,
                pages_explored=len(pages),
            )

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _guess_key_pages(self, base_url: str) -> list[str]:
            """Domain-aware fallback pages instead of generic guesses."""
            domain = urlparse(base_url).netloc.lower()
            if "amazon" in domain:
                return [
            f"{base_url}/s?k=laptop",
            f"{base_url}/ap/signin",
            f"{base_url}/cart",
            f"{base_url}/best-sellers",
            f"{base_url}/deals",
        ]
            elif "linkedin" in domain:
                return [
            f"{base_url}/jobs",
            f"{base_url}/login",
            f"{base_url}/signup",
            f"{base_url}/feed",
        ]
            elif "internshala" in domain:
                return [
                    f"{base_url}/internships",
                    f"{base_url}/jobs",
                    f"{base_url}/internships/work-from-home-internships",
                    f"{base_url}/student/login",
                    f"{base_url}/internships/in/computer-science",
                ]
            elif "ycombinator" in domain or "workatastartup" in domain:
                return [
                    f"{base_url}/jobs",
                    f"{base_url}/companies",
                    f"{base_url}/companies?batch=W24",
                    f"{base_url}/jobs?role=Engineer",
                    f"{base_url}/apply",
                ]
            else:
                paths = [
                    "/about", "/team", "/leadership",
                    "/jobs", "/careers", "/hiring", "/internships",
                    "/product", "/products", "/features", "/solutions",
                    "/pricing", "/plans",
                    "/news", "/press", "/blog",
                    "/contact", "/demo",
                    "/login", "/signup", "/search",
                ]
            return [f"{base_url}{p}" for p in paths]
            

    def _url_to_pattern(self, url: str) -> str:
        r"""
        Convert a specific URL to a regex pattern that matches the page type.
        e.g. https://amazon.com/dp/B08N5WRWNW → https://amazon\.com/dp/.+
        """
        parsed = urlparse(url)
        path = parsed.path

        # Replace dynamic segments (looks like IDs) with .+
        path = re.sub(r"/[A-Z0-9]{8,}", r"/.+", path)   # Amazon ASINs
        path = re.sub(r"/\d+", r"/\\d+", path)            # numeric IDs
        path = re.sub(r"/[a-f0-9-]{32,}", r"/.+", path)  # UUIDs

        domain = re.escape(parsed.netloc)
        return f"{parsed.scheme}://{domain}{path}"