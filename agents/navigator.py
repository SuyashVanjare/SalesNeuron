"""
SalesNeuron — Navigator Agent
================================
Executes navigation flows on websites using the Site Knowledge Graph.

Instead of exploring blindly, it:
  1. Loads the stored site graph (already learned)
  2. Finds the right flow for the task
  3. Executes step by step using Playwright
  4. Handles errors by asking LLM to self-heal
  5. Records success/failure back to DB

Real world tasks:
  - Apply for internships on Internshala
  - Easy Apply on LinkedIn
  - Fill contact forms
  - Scrape behind search (search → results → extract)
  - Submit job applications autonomously

Usage:
    agent = NavigatorAgent()
    result = await agent.execute(
        site="internshala.com",
        flow_name="search_internships",
        variables={"query": "Python Developer", "location": "Remote"}
    )
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from core.llm import llm
from core.credentials import credentials
from core.session_store import session_store
from knowledge.graph_store import graph_store
from knowledge.models import NavigationFlow, NavigationEdge

logger = logging.getLogger(__name__)


class NavigationResult:
    """Result of a navigation flow execution."""

    def __init__(self):
        self.success = False
        self.steps_completed = 0
        self.total_steps = 0
        self.final_url = ""
        self.extracted_data = {}
        self.error = None
        self.screenshots = []
        self.execution_log = []

    def log(self, message: str):
        self.execution_log.append(message)
        logger.info(f"   {message}")

    def to_dict(self):
        return {
            "success": self.success,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "final_url": self.final_url,
            "extracted_data": self.extracted_data,
            "error": self.error,
            "execution_log": self.execution_log,
        }


class DeepNavigationResult:
    """Result of a deep navigation: listing page + all scraped item pages."""

    def __init__(self):
        self.success = False
        self.listing_url = ""
        self.items: list[dict] = []
        self.error: Optional[str] = None
        self.execution_log: list[str] = []

    def log(self, message: str):
        self.execution_log.append(message)
        logger.info(f"   {message}")

    def to_dict(self):
        return {
            "success": self.success,
            "listing_url": self.listing_url,
            "item_count": len(self.items),
            "items": self.items,
            "error": self.error,
            "execution_log": self.execution_log,
        }


class NavigatorAgent:
    """
    Executes website navigation flows autonomously.
    Uses the Site Knowledge Graph to know exactly what to click.
    Falls back to LLM self-healing when selectors break.
    """

    def __init__(self):
        self._headless = os.getenv("HEADLESS_BROWSER", "true").lower() == "true"
        self._timeout = int(os.getenv("NAV_TIMEOUT_MS", "15000"))
        self._screenshot_on_error = os.getenv("SCREENSHOT_ERRORS", "true").lower() == "true"

    async def execute(
        self,
        site: str,
        flow_name: str,
        variables: dict = None,
        start_url: str = None,
        extract_after: bool = False,
    ) -> NavigationResult:
        """
        Execute a named flow on a site.

        Args:
            site:         domain e.g. "internshala.com"
            flow_name:    e.g. "search_internships", "easy_apply"
            variables:    e.g. {"query": "Python", "location": "Remote"}
            start_url:    override starting URL (optional)
            extract_after: extract page data after flow completes

        Returns:
            NavigationResult with success status and extracted data
        """
        variables = variables or {}
        result = NavigationResult()

        logger.info(f"\n{'='*55}")
        logger.info(f"🧭 NavigatorAgent: {flow_name} on {site}")
        logger.info(f"{'='*55}")

        # ── Step 1: Load site graph ────────────────────────────────
        await graph_store.init()
        graph = await graph_store.get(site)

        if not graph:
            result.error = (
                f"No site graph for {site}. "
                f"Run: python run_explorer.py https://{site} first"
            )
            logger.error(f"❌ {result.error}")
            return result

        # ── Step 2: Find the flow ──────────────────────────────────
        flow = await graph_store.get_flow(site, flow_name)

        if not flow:
            available = await graph_store.list_flows(site)
            flow_names = [f["flow_name"] for f in available]
            result.error = (
                f"Flow '{flow_name}' not found for {site}. "
                f"Available flows: {flow_names}"
            )
            logger.error(f"❌ {result.error}")
            return result

        result.total_steps = len(flow.steps)
        result.log(f"Found flow: {flow.description}")
        result.log(f"Steps: {len(flow.steps)} | Variables: {list(variables.keys())}")

        # ── Step 3: Execute the flow ───────────────────────────────
        start = start_url or graph.base_url

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            context = await self._create_context(browser, site)

            # Apply stealth
            await self._apply_stealth(context)

            page = await context.new_page()

            try:
                # ── Auto-authenticate if this flow needs a logged-in session ──
                needs_auth = flow.requires_auth or graph.base_url and self._start_page_requires_auth(graph, start)
                if needs_auth:
                    authed = await self._ensure_authenticated(
                        page=page, context=context, site=site, graph=graph, result=result
                    )
                    if not authed:
                        result.error = (
                            f"Flow '{flow_name}' requires authentication but no valid "
                            f"credentials/session were available for {site}. "
                            f"Store credentials first: agent.login(site, email, password)."
                        )
                        result.log(f"❌ {result.error}")
                        return result

                # Navigate to start URL
                result.log(f"Navigating to {start}")
                await page.goto(start, wait_until="domcontentloaded", timeout=self._timeout)
                await asyncio.sleep(2)

                # Execute each step
                for i, step in enumerate(flow.steps, 1):
                    result.log(f"Step {i}/{len(flow.steps)}: {step.description}")

                    step_ok = await self._execute_step(
                        page=page,
                        step=step,
                        variables=variables,
                        result=result,
                    )

                    if step_ok:
                        result.steps_completed += 1
                        await asyncio.sleep(1.5)
                    else:
                        # Try self-healing
                        result.log(f"⚠️  Step {i} failed — attempting self-heal...")
                        healed = await self._self_heal(page, step, variables, result)
                        if healed:
                            result.steps_completed += 1
                            result.log(f"✅ Self-healed step {i}")
                        else:
                            result.error = f"Step {i} failed and could not self-heal: {step.description}"
                            result.log(f"❌ {result.error}")
                            break

                result.final_url = page.url
                result.success = result.steps_completed == result.total_steps

                # Extract data after flow if requested
                if extract_after and result.success:
                    result.log("📊 Extracting page data...")
                    result.extracted_data = await self._extract_page_data(page)

                if result.success:
                    result.log(f"✅ Flow complete — {result.steps_completed}/{result.total_steps} steps")
                    await graph_store.record_flow_use(site, flow_name, success=True)
                else:
                    await graph_store.record_flow_use(site, flow_name, success=False)

            except Exception as e:
                result.error = str(e)
                result.success = False
                logger.error(f"❌ Navigator error: {e}")
                if self._screenshot_on_error:
                    await self._take_screenshot(page, f"error_{flow_name}")

            finally:
                await browser.close()

        return result

    # ──────────────────────────────────────────────────────────────
    # Authentication — generic, works for ANY site
    # ──────────────────────────────────────────────────────────────

    async def _create_context(self, browser: Browser, site: str) -> BrowserContext:
        """
        Create a browser context, reusing a saved session (cookies) for
        this domain if one exists and is still fresh. Saves a full
        round-trip login on every single run.
        """
        kwargs = dict(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        if session_store.is_fresh(site):
            kwargs["storage_state"] = str(session_store.path_for(site))
            logger.info(f"🍪 Reusing saved session for {site}")

        return await browser.new_context(**kwargs)

    def _start_page_requires_auth(self, graph, start_url: str) -> bool:
        """Check the SiteGraph to see if the starting page is marked requires_auth."""
        for page_node in graph.pages:
            if page_node.url == start_url or start_url.startswith(page_node.url):
                return page_node.requires_auth
        return False

    async def _ensure_authenticated(
        self,
        page: Page,
        context: BrowserContext,
        site: str,
        graph,
        result: NavigationResult,
    ) -> bool:
        """
        Make sure this browser context is logged in before continuing.
        1. If session was reused (fresh cookies loaded) — assume OK, skip.
        2. Otherwise, look up stored credentials + the site's login flow
           and execute the login flow now.
        Returns True if authenticated (or auth wasn't actually needed),
        False if credentials/flow were missing.
        """
        if session_store.is_fresh(site):
            result.log(f"🔐 Using existing session for {site} (skipping login)")
            return True

        await credentials.init()
        creds = await credentials.get(site)
        if not creds:
            result.log(f"🔐 No stored credentials for {site}")
            return False

        login_flow = await self._find_auth_flow(site, "login")
        if not login_flow:
            result.log(f"🔐 No login flow learned for {site}")
            return False

        result.log(f"🔐 Logging in to {site}...")
        login_vars = {"email": creds["email"], "password": creds["password"]}

        await page.goto(graph.base_url, wait_until="domcontentloaded", timeout=self._timeout)
        await asyncio.sleep(1.5)

        for i, step in enumerate(login_flow.steps, 1):
            ok = await self._execute_step(page, step, login_vars, result)
            if not ok:
                ok = await self._self_heal(page, step, login_vars, result)
            if not ok:
                result.log(f"🔐 Login step {i} failed")
                return False
            await asyncio.sleep(1.2)

        if login_flow.success_indicator:
            try:
                await page.wait_for_selector(login_flow.success_indicator, timeout=self._timeout)
            except Exception:
                result.log("🔐 Login success indicator not found — login may have failed")
                return False

        result.log(f"✅ Logged in to {site}")
        await session_store.save(site, context)
        await graph_store.record_flow_use(site, login_flow.flow_name, success=True)
        return True

    async def _find_auth_flow(self, site: str, auth_type: str) -> Optional[NavigationFlow]:
        """Find the stored login/signup flow for a domain."""
        available = await graph_store.list_flows(site)
        for f in available:
            flow = await graph_store.get_flow(site, f["flow_name"])
            if flow and flow.is_auth_flow and flow.auth_flow_type == auth_type:
                return flow
        return None

    async def login(self, site: str, email: str, password: str) -> bool:
        """
        Store credentials for a site and immediately try logging in once
        to confirm they work. Call this before running protected flows.
        """
        await credentials.init()
        await credentials.save(site, email, password)

        await graph_store.init()
        graph = await graph_store.get(site)
        if not graph:
            logger.error(f"No site graph for {site} — run explorer first")
            return False

        session_store.clear(site)  # force a fresh login attempt

        result = NavigationResult()
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self._headless)
            context = await self._create_context(browser, site)
            await self._apply_stealth(context)
            page = await context.new_page()
            try:
                ok = await self._ensure_authenticated(page, context, site, graph, result)
            finally:
                await browser.close()
        return ok

    async def signup(self, site: str, email: str, password: str, **extra_fields) -> NavigationResult:
        """
        Execute the site's signup flow (if one has been learned) and, on
        success, store the credentials so future flows can auto-login.
        """
        variables = {"email": email, "password": password, **extra_fields}
        signup_flow = await self._find_auth_flow(site, "signup")
        result = NavigationResult()
        if not signup_flow:
            result.error = f"No signup flow learned for {site}"
            return result

        result = await self.execute(site, signup_flow.flow_name, variables)
        if result.success:
            await credentials.init()
            await credentials.save(site, email, password)
            result.log("🔐 Credentials saved after successful signup")
        return result

    # ──────────────────────────────────────────────────────────────
    # Deep scraping — listing page → individual item pages, w/ pagination
    # ──────────────────────────────────────────────────────────────

    async def execute_deep(
        self,
        site: str,
        flow_name: str,
        variables: dict = None,
        item_link_hint: str = "a",
        max_items: int = 10,
        follow_pagination: bool = False,
        max_pages: int = 3,
    ) -> "DeepNavigationResult":
        """
        Run a listing flow (e.g. search_internships), then visit each
        individual item found on the results page (e.g. each job posting)
        and extract structured data from it. Optionally follows "next
        page" pagination across multiple listing pages first.

        This is generic — item_link_hint lets the caller nudge which
        links are "items" (e.g. 'a.job-title-link'), but by default the
        LLM is used to identify item links vs. navigation/ad links.
        """
        variables = variables or {}
        deep_result = DeepNavigationResult()

        await graph_store.init()
        graph = await graph_store.get(site)
        if not graph:
            deep_result.error = f"No site graph for {site}. Run explorer first."
            return deep_result

        flow = await graph_store.get_flow(site, flow_name)
        if not flow:
            deep_result.error = f"Flow '{flow_name}' not found for {site}"
            return deep_result

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self._headless)
            context = await self._create_context(browser, site)
            await self._apply_stealth(context)
            page = await context.new_page()

            listing_result = NavigationResult()
            try:
                if flow.requires_auth:
                    authed = await self._ensure_authenticated(page, context, site, graph, listing_result)
                    if not authed:
                        deep_result.error = f"Authentication required for {site} but unavailable"
                        await browser.close()
                        return deep_result

                start = graph.base_url
                await page.goto(start, wait_until="domcontentloaded", timeout=self._timeout)
                await asyncio.sleep(1.5)

                for step in flow.steps:
                    ok = await self._execute_step(page, step, variables, listing_result)
                    if not ok:
                        ok = await self._self_heal(page, step, variables, listing_result)
                    await asyncio.sleep(1)

                deep_result.listing_url = page.url

                pages_scraped = 0
                while True:
                    item_links = await self._find_item_links(page, item_link_hint)
                    item_links = item_links[: max(0, max_items - len(deep_result.items))]
                    deep_result.log(f"Found {len(item_links)} item link(s) on {page.url}")

                    for link in item_links:
                        item_data = await self._scrape_item_page(context, link)
                        if item_data:
                            deep_result.items.append(item_data)
                        if len(deep_result.items) >= max_items:
                            break

                    pages_scraped += 1
                    if (
                        not follow_pagination
                        or len(deep_result.items) >= max_items
                        or pages_scraped >= max_pages
                    ):
                        break

                    next_link = await self._find_next_page_link(page)
                    if not next_link:
                        deep_result.log("No further pagination found")
                        break
                    deep_result.log(f"Following pagination → {next_link}")
                    await page.goto(next_link, wait_until="domcontentloaded", timeout=self._timeout)
                    await asyncio.sleep(1.5)

                deep_result.success = len(deep_result.items) > 0
                await graph_store.record_flow_use(site, flow_name, success=deep_result.success)

            except Exception as e:
                deep_result.error = str(e)
                logger.error(f"❌ Deep navigation error: {e}")
            finally:
                await browser.close()

        return deep_result

    async def _find_item_links(self, page: Page, hint: str) -> list[str]:
        """
        Identify links to individual items on a listing page (job cards,
        product cards, etc). Uses the hint selector if it matches enough
        elements; otherwise falls back to an LLM pass over candidate links.
        """
        try:
            hinted = await page.eval_on_selector_all(
                hint, "els => els.map(e => e.href).filter(Boolean)"
            )
        except Exception:
            hinted = []

        # De-dupe, drop obvious nav/social/auth links
        skip_markers = ("login", "signup", "logout", "javascript:", "#", "mailto:", "tel:")
        hinted = [
            u for u in dict.fromkeys(hinted)
            if not any(m in u.lower() for m in skip_markers)
        ]
        if len(hinted) >= 3:
            return hinted

        # Fallback: ask LLM to pick item links from all links on the page
        all_links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({href: e.href, text: e.innerText.trim().slice(0,80)}))"
        )
        all_links = [l for l in all_links if l["href"] and not any(m in l["href"].lower() for m in skip_markers)]
        all_links = all_links[:150]

        prompt = (
            "This is a list of links found on a search-results / listing web page.\n"
            "Return ONLY a JSON array of the href values that point to an INDIVIDUAL "
            "item detail page (e.g. one specific job posting, one specific company, "
            "one specific product) — NOT navigation, filters, ads, login/signup, or "
            "category links.\n\n"
            f"LINKS:\n{json.dumps(all_links, indent=2)}\n\n"
            "Return ONLY a JSON array of strings, nothing else."
        )
        try:
            picked = await llm.generate_structured(prompt, "[]", temperature=0.1)
            if isinstance(picked, list):
                return picked
        except Exception as e:
            logger.warning(f"Item-link LLM selection failed: {e}")

        return hinted

    async def _find_next_page_link(self, page: Page) -> Optional[str]:
        """Find a 'next page' pagination link, if any."""
        try:
            candidates = await page.eval_on_selector_all(
                "a[href]",
                """els => els.map(e => ({
                    href: e.href,
                    text: (e.innerText || '').trim().toLowerCase(),
                    rel: e.getAttribute('rel') || '',
                    aria: (e.getAttribute('aria-label') || '').toLowerCase()
                }))"""
            )
        except Exception:
            return None

        for c in candidates:
            if c["rel"] == "next":
                return c["href"]
        for c in candidates:
            if c["text"] in ("next", "next page", "»", ">") or "next" in c["aria"]:
                return c["href"]
        return None

    async def _scrape_item_page(self, context: BrowserContext, url: str) -> Optional[dict]:
        """Visit one item detail page and extract structured data from it."""
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout)
            await asyncio.sleep(1)
            data = await self._extract_page_data(page)
            data["_source_url"] = url
            return data
        except Exception as e:
            logger.warning(f"Item scrape failed for {url}: {e}")
            return None
        finally:
            await page.close()

    async def execute_task(
        self,
        site: str,
        task_description: str,
        variables: dict = None,
    ) -> NavigationResult:
        """
        High-level task execution — LLM picks the right flow.
        e.g. execute_task("internshala.com", "apply for python internships")
        """
        variables = variables or {}

        # Ask LLM which flow to use
        available_flows = await graph_store.list_flows(site)
        if not available_flows:
            result = NavigationResult()
            result.error = f"No flows learned for {site}. Run explorer first."
            return result

        flows_text = "\n".join([
            f"- {f['flow_name']}: {f['description']}"
            for f in available_flows
        ])

        prompt = (
            f"Task: {task_description}\n\n"
            f"Available flows for {site}:\n{flows_text}\n\n"
            f"Which flow_name best matches the task?\n"
            f"Return ONLY the flow_name string, nothing else."
        )

        try:
            flow_name = await llm.generate(prompt, temperature=0.1)
            flow_name = flow_name.strip().strip('"').strip("'")
            logger.info(f"🧠 LLM selected flow: {flow_name}")
            return await self.execute(site, flow_name, variables)
        except Exception as e:
            result = NavigationResult()
            result.error = f"Flow selection failed: {e}"
            return result

    # ──────────────────────────────────────────────────────────────
    # Step execution
    # ──────────────────────────────────────────────────────────────

    async def _execute_step(
        self,
        page: Page,
        step: NavigationEdge,
        variables: dict,
        result: NavigationResult,
    ) -> bool:
        """Execute a single navigation step."""
        try:
            selector = step.selector
            action = step.action_type
            input_value = self._fill_variables(step.input_value or "", variables)
            wait_for = step.wait_for

            if action == "navigate":
                url = self._fill_variables(selector, variables)
                await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout)

            elif action == "click":
                await page.wait_for_selector(selector, timeout=self._timeout)
                await page.click(selector)

            elif action == "type":
                await page.wait_for_selector(selector, timeout=self._timeout)
                await page.click(selector)
                await page.fill(selector, "")  # clear first
                await page.type(selector, input_value, delay=50)  # human-like typing

            elif action == "submit":
                await page.wait_for_selector(selector, timeout=self._timeout)
                await page.press(selector, "Enter")

            elif action == "select":
                await page.wait_for_selector(selector, timeout=self._timeout)
                await page.select_option(selector, input_value)

            elif action == "hover":
                await page.wait_for_selector(selector, timeout=self._timeout)
                await page.hover(selector)

            elif action == "scroll":
                await page.evaluate("window.scrollBy(0, 500)")

            # Wait for expected element after action
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=self._timeout)

            return True

        except Exception as e:
            logger.warning(f"   Step failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Self-healing — LLM finds new selector when old one breaks
    # ──────────────────────────────────────────────────────────────

    async def _self_heal(
        self,
        page: Page,
        step: NavigationEdge,
        variables: dict,
        result: NavigationResult,
    ) -> bool:
        """
        When a selector breaks (site updated their CSS), ask LLM
        to find the right element from the current page DOM.
        This is what makes the Navigator robust in production.
        """
        try:
            # Get current page elements
            elements = await page.evaluate("""
                () => {
                    const els = [];
                    document.querySelectorAll('button, input, a, [role="button"]').forEach(el => {
                        const text = el.innerText?.trim() || el.placeholder || el.getAttribute('aria-label') || '';
                        if (text && text.length < 100) {
                            els.push({
                                tag: el.tagName.toLowerCase(),
                                id: el.id || null,
                                text: text.slice(0, 60),
                                type: el.type || null,
                            });
                        }
                    });
                    return els.slice(0, 30);
                }
            """)

            elements_text = json.dumps(elements, indent=2)

            prompt = (
                f"A web automation step failed. Find the right element.\n\n"
                f"FAILED STEP: {step.description}\n"
                f"ORIGINAL SELECTOR: {step.selector}\n"
                f"ACTION: {step.action_type}\n\n"
                f"CURRENT PAGE ELEMENTS:\n{elements_text}\n\n"
                f"What CSS selector should be used instead?\n"
                f"If element has an id, use #id. Otherwise use tag + text.\n"
                f"Return ONLY the CSS selector string, nothing else."
            )

            new_selector = await llm.generate(prompt, temperature=0.1)
            new_selector = new_selector.strip().strip('"').strip("'")

            result.log(f"Self-heal: trying new selector '{new_selector}'")

            # Try with new selector
            healed_step = NavigationEdge(
                from_url_pattern=step.from_url_pattern,
                to_url_pattern=step.to_url_pattern,
                action_type=step.action_type,
                selector=new_selector,
                input_value=step.input_value,
                description=step.description,
                wait_for=step.wait_for,
            )

            return await self._execute_step(page, healed_step, variables, result)

        except Exception as e:
            logger.warning(f"Self-heal failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Data extraction after flow
    # ──────────────────────────────────────────────────────────────

    async def _extract_page_data(self, page: Page) -> dict:
        """
        Extract structured data from the current page after flow completes.
        Used to scrape search results, job listings, etc.

        Captures both the visible text AND every link's real href, and
        explicitly forbids the LLM from substituting link/button TEXT
        (e.g. "View details", "Apply now") for an actual URL — that was
        producing garbage like apply_url: "View details".
        """
        try:
            payload = await page.evaluate("""
                () => {
                    document.querySelectorAll('script,style,nav,footer').forEach(e => e.remove());
                    const text = document.body.innerText.slice(0, 6000);
                    const links = Array.from(document.querySelectorAll('a[href]'))
                        .map(a => ({
                            text: (a.innerText || '').trim().slice(0, 100),
                            href: a.href
                        }))
                        .filter(l => l.href && !l.href.startsWith('javascript:') && l.text)
                        .slice(0, 200);
                    return { text, links };
                }
            """)

            prompt = (
                "Extract structured data from this web page.\n\n"
                f"PAGE TEXT:\n{payload['text']}\n\n"
                "LINKS ON THIS PAGE (visible text -> real href). Use ONLY these hrefs "
                "for any url/apply_url field, matched by nearby text. If you cannot "
                "confidently match a real href to an item, set that field to null. "
                "NEVER put link/button caption text (e.g. 'View details', 'Apply now', "
                "'Read more') into a url field — that is always wrong.\n"
                f"{json.dumps(payload['links'], indent=2)}\n\n"
                "Return a JSON object.\n"
                "For job/product listings: return items as a list, each with title, "
                "company/brand, location, price/salary, and url (real href or null).\n"
                "For profiles: extract name, role, company, contact info.\n"
                "Return ONLY valid JSON, no markdown."
            )

            result = await llm.generate_structured(prompt, "{}", temperature=0.1)
            return result

        except Exception as e:
            logger.warning(f"Data extraction failed: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────
    # Stealth + helpers
    # ──────────────────────────────────────────────────────────────

    async def _apply_stealth(self, context: BrowserContext):
        """Apply stealth patches to browser context."""
        try:
            from playwright_stealth import stealth_async
            # Apply to all new pages in this context
            async def stealth_page(page):
                await stealth_async(page)
            context.on("page", lambda page: asyncio.ensure_future(stealth_page(page)))
        except ImportError:
            pass

        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)

    def _fill_variables(self, template: str, variables: dict) -> str:
        """Replace {variable} placeholders with actual values."""
        if not template:
            return ""
        for key, value in variables.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    async def _take_screenshot(self, page: Page, name: str):
        """Save screenshot for debugging."""
        try:
            import os
            os.makedirs("data/screenshots", exist_ok=True)
            path = f"data/screenshots/{name}.png"
            await page.screenshot(path=path)
            logger.info(f"📸 Screenshot saved: {path}")
        except Exception:
            pass