"""
SalesNeuron — Researcher Agent
============================================================
The first agent in the SalesNeuron pipeline.

Workflow:
  1. PLAN    — LLM decides which pages to scrape for this company
  2. SCRAPE  — Playwright scrapes each page in sequence
  3. EXTRACT — LLM analyzes raw text → structured ProspectProfile
  4. ENRICH  — Cross-checks and fills gaps (job listings, news signals)
  5. SCORE   — Assigns buying signal strength and research confidence

Input:  company website URL (str)
Output: ProspectProfile (Pydantic model, JSON-serializable)
"""

import asyncio
import json
import logging
import os
from urllib.parse import urlparse

from core.llm import llm
from core.scraper import BrowserScraper
from core.models import ProspectProfile, BuyingSignal, FundingRound, KeyPerson, TechStack
from core.memory import memory
from core.linkedin_finder import linkedin_finder
from knowledge.graph_store import graph_store

logger = logging.getLogger(__name__)

MAX_PAGES = int(os.getenv("MAX_PAGES_PER_PROSPECT", "8"))


class ResearcherAgent:
    """
    Autonomous agent that researches a company from its website URL.

    Example:
        agent = ResearcherAgent()
        profile = await agent.research("https://stripe.com")
        print(profile.model_dump_json(indent=2))
    """

    async def research(self, company_url: str, force_refresh: bool = False) -> ProspectProfile:
        """
        Full research pipeline for one prospect.
        Checks SQLite cache first — only scrapes if data is missing or stale.
        Uses the SiteGraph (if already learned) to seed the page plan —
        so the Explorer and Researcher share knowledge instead of duplicating work.
        Pass force_refresh=True to always re-scrape regardless of cache.
        """
        company_url = company_url.rstrip("/")
        domain = urlparse(company_url).netloc
        logger.info(f"\n{'='*55}")
        logger.info(f"🔍 ResearcherAgent: starting research on {domain}")
        logger.info(f"{'='*55}")

        # ── Cache check ─────────────────────────────────────────────
        await memory.init()
        if not force_refresh:
            cached = await memory.get(company_url)
            if cached:
                logger.info(f"⚡ CACHE HIT — returning stored profile, no scraping needed")
                return cached
            logger.info(f"💭 No fresh cache — starting full scrape pipeline...")
        else:
            logger.info(f"🔄 Force refresh mode — ignoring cache")

        # ── Check if SiteGraph already exists for this domain ────────────
        # The Explorer may have already mapped this site. If so, we can
        # use its discovered page URLs directly instead of re-scraping
        # the homepage just to get links.
        await graph_store.init()
        clean_domain = domain.replace("www.", "")
        site_graph = await graph_store.get(clean_domain)
        if site_graph:
            logger.info(
                f"🗺️  SiteGraph HIT — {clean_domain} has {len(site_graph.pages)} mapped pages. "
                f"Using graph to seed exploration."
            )
        else:
            logger.info(f"🗺️  SiteGraph MISS — {clean_domain} not learned yet, will discover pages from homepage.")

        async with BrowserScraper() as scraper:
            # ── Step 1: PLAN ────────────────────────────────────────────
            logger.info("📋 Step 1/4 — Planning pages to scrape...")
            pages_to_scrape = await self._plan_pages(company_url, scraper, site_graph)
            logger.info(f"   Will scrape {len(pages_to_scrape)} pages")
            for p in pages_to_scrape:
                logger.info(f"   → {p}")

            # ── Step 2: SCRAPE ──────────────────────────────────────────
            logger.info("🌐 Step 2/4 — Scraping pages...")
            try:
                scraped_pages = await scraper.scrape_multiple(pages_to_scrape)
            except Exception as e:
                # Browser crash (e.g. "Connection closed while reading from driver")
                # — don't kill the pipeline. Work with whatever we have.
                logger.warning(
                    f"  💥 Browser session crashed mid-scrape: {e}\n"
                    f"  ⚠️  Continuing with any pages already scraped..."
                )
                scraped_pages = []

            successful = [p for p in scraped_pages if p["success"]]
            logger.info(f"   {len(successful)}/{len(scraped_pages)} pages scraped successfully")

            if not successful:
                logger.warning(
                    "   ⚠️  No pages scraped successfully — this site may block "
                    "all automated access. Building minimal profile from URL only."
                )

            # ── Step 3: EXTRACT ─────────────────────────────────────────
            logger.info("🧠 Step 3/4 — LLM extracting structured data...")
            combined_text = self._combine_scraped_text(successful)
            raw_profile = await self._extract_profile(company_url, combined_text)

            # ── Step 4: ENRICH ──────────────────────────────────────────
            logger.info("⚡ Step 4/4 — Detecting buying signals...")
            buying_signals = await self._detect_buying_signals(raw_profile, combined_text)
            raw_profile["buying_signals"] = buying_signals
            raw_profile["pages_scraped"] = [p["url"] for p in successful]
            raw_profile["raw_text_summary"] = combined_text[:3000]

        # Build and validate final model
        profile = self._build_profile(raw_profile)
        logger.info(f"✅ Research complete — confidence: {profile.research_confidence}")
        logger.info(f"   Buying signals: {len(profile.buying_signals)} detected")
        logger.info(f"   Open roles found: {len(profile.open_job_roles)}")

        # ── LinkedIn fallback: website had no findable people ──────────
        # A ProspectProfile with zero key_people can't be personalized to
        # a real contact downstream — EmailFinder needs a person, not just
        # a company name. Try LinkedIn before giving up on a real contact.
        if not profile.key_people:
            logger.info(
                "👤 No key people found on website — trying LinkedIn finder..."
            )
            try:
                li_people = await linkedin_finder.find_people(
                    company_name=profile.company_name,
                    domain=urlparse(profile.website).netloc,
                )
                if li_people:
                    profile.key_people = [
                        KeyPerson(
                            name=p["name"],
                            title=p.get("title", ""),
                            linkedin_url=p.get("linkedin_url"),
                        )
                        for p in li_people
                    ]
                    logger.info(
                        f"👤 LinkedIn finder added {len(li_people)} people "
                        f"(source: {li_people[0].get('source', 'unknown')})"
                    )
                else:
                    logger.info("👤 LinkedIn finder found no people either")
            except Exception as e:
                logger.warning(f"👤 LinkedIn finder failed (non-fatal): {e}")

        # ── Persist to memory (prospect cache) ────────────────────────
        await memory.save(profile)

        return profile

    # ────────────────────────────────────────────────────────────────
    # Step 1 — PLAN
    # ────────────────────────────────────────────────────────────────

    async def _plan_pages(
        self,
        url: str,
        scraper: BrowserScraper,
        site_graph=None,
    ) -> list[str]:
        """
        Decide which pages to scrape. Two strategies:

        A) SiteGraph exists — the Explorer already mapped this site.
           Use its known page URLs as the candidate pool. No homepage
           re-scrape needed; we skip straight to LLM selection.

        B) No SiteGraph — scrape homepage to collect links, then LLM picks.
           After LLM selection, do a second-pass deep discovery on
           listing pages (jobs, products, news) to find individual items.
        """
        pages = [url]
        candidate_links: list[str] = []

        # ── Strategy A: use existing site graph ─────────────────────────
        if site_graph and site_graph.pages:
            # Collect all page URLs the Explorer already discovered
            for page_node in site_graph.pages:
                if page_node.url and page_node.url != url:
                    candidate_links.append(page_node.url)
                # Also include outgoing links that may not be full page nodes
                for out_url in page_node.outgoing_urls:
                    if out_url not in candidate_links and out_url != url:
                        candidate_links.append(out_url)
            logger.info(f"   🗺️  Using {len(candidate_links)} URLs from SiteGraph")

        # ── Strategy B: scrape homepage for links ─────────────────────
        if not candidate_links:
            homepage = await scraper.scrape(url)
            if not homepage["success"]:
                logger.warning("Homepage scrape failed — using URL-guessing fallback")
                return self._guess_key_pages(url)[:MAX_PAGES]
            candidate_links = homepage.get("links", [])
            logger.info(f"   🌐 Scraped homepage: {len(candidate_links)} links found")

        if not candidate_links:
            logger.warning("No links found on homepage — using URL-guessing fallback")
            return self._guess_key_pages(url)[:MAX_PAGES]

        # ── LLM selects the best pages from candidates ─────────────────
        links_text = "\n".join(candidate_links[:40])
        prompt = f"""
You are a B2B sales intelligence researcher building a complete company profile.
Company URL: {url}

Available pages (already ranked by relevance):
{links_text}

Select up to {MAX_PAGES - 1} URLs that together give the MOST COMPLETE picture.
Cover as many DIFFERENT categories as possible:

1. Hiring / Jobs / Careers / Internships  — reveals growth and pain points
2. About / Company / Mission / Story       — company background
3. Team / Leadership / Founders            — key people
4. Products / Features / Solutions         — what they sell
5. Pricing / Plans                         — business model
6. Customers / Case Studies / Success      — who they serve
7. News / Press / Blog / Announcements     — recent events
8. Contact / Demo / Trial                  — sales touchpoints

RULES:
- Pick the MOST SPECIFIC URL per category (e.g. /careers over /)
- Do NOT pick two URLs from the same category
- If a jobs/careers page exists, it MUST be included
- If a team/leadership/founders page exists, it MUST be included — we need real people's names to personalize outreach
- Exclude legal, cookie, support, and help pages

Return ONLY a JSON array of full URLs:
["https://...", "https://..."]
"""
        try:
            raw = await llm.generate(prompt, temperature=0.1)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            selected = json.loads(raw)
            if isinstance(selected, list):
                pages.extend([u for u in selected if isinstance(u, str)])
        except Exception as e:
            logger.warning(f"Link planning LLM failed ({e}), using guessed pages")
            pages.extend(self._guess_key_pages(url))

        # Deduplicate — leave 2 slots for deep discovery
        seen = set()
        unique = []
        for p in pages:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        planned = unique[:MAX_PAGES - 2]

        # ── Deep discovery: listing page → individual items ─────────────
        # Scrape listing pages (jobs, products, news) to pull sample
        # child-page URLs (job descriptions, product detail, articles).
        deep = await self._discover_deep_pages(planned, scraper)
        for d in deep:
            if d not in seen:
                seen.add(d)
                planned.append(d)

        return planned[:MAX_PAGES]

    async def _discover_deep_pages(
        self, planned_urls: list[str], scraper: BrowserScraper
    ) -> list[str]:
        """
        For listing-type pages in the plan (jobs, products, news, blog),
        scrape them and extract 1-2 sample individual-item URLs.

        Example:
          /careers  →  /careers/senior-engineer-remote
          /products →  /products/enterprise
          /blog     →  /blog/q1-2024-growth-update

        This gives us the real content — job descriptions, product details, etc.
        """
        LISTING_KEYWORDS = [
            "job", "jobs", "career", "careers", "internship", "internships",
            "opening", "vacancy", "hiring",
            "product", "products", "solutions", "features",
            "blog", "news", "press", "announce",
            "customer", "case-study", "case_study", "success",
        ]

        deep_urls: list[str] = []
        already_planned = set(planned_urls)

        for page_url in planned_urls:
            path = urlparse(page_url).path.lower()
            if not any(kw in path for kw in LISTING_KEYWORDS):
                continue

            try:
                scraped = await scraper.scrape(page_url)
                if not scraped["success"]:
                    continue

                base_path = path.rstrip("/")
                base_domain = urlparse(page_url).netloc

                # FIX: removed startswith(base_path) — too strict.
                # Many sites use /job/123 or /position/engineer
                # which share the same domain but not the same path prefix.
                # Instead we accept any same-domain link that:
                #   (a) is longer than the listing page itself (it's a child)
                #   (b) contains at least one path segment beyond the base
                children = [
                    link for link in scraped["links"]
                    if link not in already_planned
                    and urlparse(link).netloc == base_domain
                    and len(urlparse(link).path) > len(base_path) + 2
                    and any(kw in urlparse(link).path.lower() for kw in LISTING_KEYWORDS + [
                        "detail", "view", "position", "role", "opening",
                        "description", "apply", "post",
                    ])
                ]

                for child in children[:2]:
                    deep_urls.append(child)
                    already_planned.add(child)
                    logger.info(f"   📄 Deep page found: {child[:70]}")

            except Exception as e:
                logger.warning(f"Deep discovery failed for {page_url}: {e}")

        return deep_urls

    def _guess_key_pages(self, base_url: str) -> list[str]:
        """
        Fallback: guess common paths when LLM link planning fails.
        Covers all key categories: hiring, company, product, news.

        Team/leadership paths are listed FIRST — real people's names are
        essential for contact selection downstream (EmailFinder needs a
        person, not just a company name). Includes common variants
        (e.g. /about/leadership, /about/team) generically — not tied to
        any specific company's site structure.
        """
        paths = [
            # Company / leadership (highest priority — we need real names)
            "/about/leadership", "/about/team", "/company/team",
            "/about-us/team", "/team", "/leadership", "/founders", "/about",
            # Hiring
            "/jobs", "/careers", "/hiring", "/internships", "/openings",
            # Product & commercial
            "/product", "/products", "/features", "/solutions", "/pricing", "/plans",
            # Social proof
            "/customers", "/case-studies", "/success-stories",
            # News
            "/newsroom", "/news", "/press", "/blog",
        ]
        return [f"{base_url}{p}" for p in paths]

    # ────────────────────────────────────────────────────────────────
    # Step 2 helper — combine scraped text
    # ────────────────────────────────────────────────────────────────

    def _combine_scraped_text(self, pages: list[dict]) -> str:
        sections = []
        for page in pages:
            if page["text"]:
                sections.append(
                    f"=== PAGE: {page['url']} ===\n"
                    f"TITLE: {page['title']}\n"
                    f"{page['text'][:4000]}"
                )
        return "\n\n".join(sections)

    # ────────────────────────────────────────────────────────────────
    # Step 3 — EXTRACT
    # ────────────────────────────────────────────────────────────────

    async def _extract_profile(self, url: str, combined_text: str) -> dict:
        """
        Send all scraped text to LLM → get structured company profile.
        """
        schema = """
{
  "company_name": "string",
  "website": "string (the URL)",
  "industry": "string (be specific, e.g. 'B2B SaaS - HR Tech')",
  "company_size": "string or null (e.g. '50-200 employees')",
  "founded_year": "string or null",
  "headquarters": "string or null (City, Country)",
  "description": "string (2-3 sentences: what they do, who they serve, key differentiator)",
  "key_people": [
    {"name": "string", "title": "string", "linkedin_url": "string or null"}
  ],
  "recent_news": ["string (specific event with date if available)"],
  "funding_history": [
    {"round_type": "string", "amount_usd": "string or null", "date": "string or null", "investors": ["string"]}
  ],
  "tech_stack": [
    {"category": "string", "tools": ["string"]}
  ],
  "products_services": ["string"],
  "open_job_roles": [
    {
      "title": "string (exact job title)",
      "team": "string or null (e.g. Engineering, Sales, Design)",
      "location": "string or null (e.g. Remote, Bangalore, New York)",
      "type": "string or null (full-time / internship / contract / part-time)",
      "skills": ["string (required skills or technologies mentioned)"],
      "description_snippet": "string or null (1-2 sentence summary of the role)"
    }
  ],
  "hiring_velocity": "string or null (e.g. '15 open roles, heavy engineering hiring')",
  "hiring_locations": ["string (unique cities/countries where they hire)"],
  "research_confidence": "high/medium/low"
}
"""
        prompt = f"""
You are a senior B2B sales intelligence analyst building a prospect profile for sales outreach.

Analyze the following scraped website content and extract structured information.
Be specific and factual. Only include information you found in the text.
If something is not mentioned, use null or an empty array.

SCRAPED CONTENT:
{combined_text[:12000]}

Extract all available information and return it as a JSON object matching this schema:
{schema}
"""
        try:
            result = await llm.generate_structured(prompt, schema, temperature=0.1)
            result["website"] = url
            return result
        except Exception as e:
            logger.error(f"Profile extraction failed: {e}")
            # Return minimal profile to keep pipeline running
            domain = urlparse(url).netloc
            return {
                "company_name": domain.replace("www.", "").split(".")[0].title(),
                "website": url,
                "industry": "Unknown",
                "description": f"Company website at {url}. Extraction failed — manual review needed.",
                "research_confidence": "low",
            }

    # ────────────────────────────────────────────────────────────────
    # Step 4 — ENRICH: detect buying signals
    # ────────────────────────────────────────────────────────────────

    async def _detect_buying_signals(
        self, profile: dict, combined_text: str
    ) -> list[dict]:
        """
        LLM reads the profile + raw text and identifies specific buying signals.
        These are what make the cold email hyper-personalized.
        """
        open_roles = profile.get("open_job_roles", []) or []
        role_titles = [
            str(r) if r and not isinstance(r, dict)
            else r.get("title", "") if isinstance(r, dict)
            else ""
            for r in open_roles
        ]
        role_titles = [r for r in role_titles if r]  # remove empty strings
        roles_summary = f"Open roles ({len(role_titles)} total): " + ", ".join(role_titles) if role_titles else "No hiring data found"
        # Sanitize profile before passing to f-string
        profile_clean = {k: v for k, v in profile.items() if v is not None}
        profile_clean["open_job_roles"] = role_titles
        prompt = f"""
You are a B2B sales signal analyst. Analyze the company information and identify
concrete buying signals — specific, observable facts that suggest this company might
need a new product or service right now.

COMPANY: {profile.get('company_name', 'Unknown')}
INDUSTRY: {profile.get('industry', 'Unknown')}
HIRING: {roles_summary or 'No hiring data found'}
HIRING VELOCITY: {profile.get('hiring_velocity', 'Unknown')}

SCRAPED TEXT SUMMARY:
{combined_text[:5000]}

PROFILE SO FAR:
{json.dumps(profile_clean, indent=2)[:3000]}

Identify up to 6 buying signals. Be VERY SPECIFIC — reference actual facts
(e.g. "12 open engineering roles including 3 ML positions" not just "hiring").

Valid signal types:
- hiring_surge: Rapid hiring in a specific function (name function + count)
- skill_demand: Specific tech/skills listed repeatedly across job postings
- recent_funding: Fresh capital announced (name round and amount)
- tech_migration: Explicitly moving away from a known tool
- product_launch: New product or major feature recently launched
- leadership_change: New C-suite or VP hired recently
- geographic_expansion: Opening offices or expanding to new markets
- pain_point_mention: Direct mention of a challenge, bottleneck, or problem
- rapid_growth: Strong signals of headcount or revenue growth

Return a JSON array:
[
  {{
    "signal_type": "one of the types above",
    "description": "specific factual detail — cite names and numbers",
    "strength": "high/medium/low",
    "source_url": "url where this was found, or null"
  }}
]
Return ONLY the JSON array.
"""
        try:
            raw = await llm.generate(prompt, temperature=0.1)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            signals = json.loads(raw)
            if isinstance(signals, list):
                return signals
        except Exception as e:
            logger.warning(f"Buying signal detection failed: {e}")
        return []

    # ────────────────────────────────────────────────────────────────
    # Final: build validated Pydantic model
    # ────────────────────────────────────────────────────────────────

    def _build_profile(self, raw: dict) -> ProspectProfile:
        """
        Convert the raw LLM dict into a validated ProspectProfile.
        Handles missing fields and both old (list[str]) and new (list[dict]) formats.

        IMPORTANT — why _safe_str exists:
          raw.get("company_name", "Unknown") only falls back to "Unknown"
          when the KEY IS ABSENT. It does NOT catch the key being present
          with value None, or being present with the wrong type (a list,
          a dict, a number). On a near-empty/thin-content page (e.g. a
          site that returns almost no scraped text), the LLM often
          legitimately returns `"company_name": null` rather than
          omitting the key — and .get() lets that None straight through
          to Pydantic, which then rejects it with "Input should be a
          valid string". This was the exact cause of the
          consciousengines.com crash: company_name, industry, and
          description all came back None from the same thin-content
          extraction, and all three failed validation simultaneously.
        """
        def _safe_str(value, default: str) -> str:
            """Coerce any LLM output to a usable string, or fall back."""
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if isinstance(value, list) and value:
                # LLM sometimes wraps a single value in a list
                first = value[0]
                return str(first) if first else default
            return default

        def safe_list(data, model_cls):
            result = []
            for item in (data or []):
                try:
                    result.append(model_cls(**item) if isinstance(item, dict) else item)
                except Exception:
                    pass
            return result

        # Normalise open_job_roles: LLM may return list[dict] or legacy list[str]
        raw_roles = raw.get("open_job_roles", [])
        normalised_roles = []
        for r in raw_roles:
            if isinstance(r, dict):
                normalised_roles.append(r)
            elif isinstance(r, str):
                normalised_roles.append({"title": r})

        domain_fallback = urlparse(raw.get("website", "") or "").netloc \
            .replace("www.", "").split(".")[0].title() or "Unknown"

        return ProspectProfile(
            company_name=_safe_str(raw.get("company_name"), domain_fallback),
            website=raw.get("website") or "",
            industry=_safe_str(raw.get("industry"), "Unknown"),
            company_size=raw.get("company_size") if isinstance(raw.get("company_size"), str) else None,
            founded_year=raw.get("founded_year") if isinstance(raw.get("founded_year"), str) else None,
            headquarters=raw.get("headquarters") if isinstance(raw.get("headquarters"), str) else None,
            description=_safe_str(
                raw.get("description"),
                f"Company at {raw.get('website') or 'unknown URL'}. "
                f"Limited data available — site returned little content."
            ),
            key_people=safe_list(raw.get("key_people", []), KeyPerson),
            recent_news=raw.get("recent_news", []) if isinstance(raw.get("recent_news"), list) else [],
            funding_history=safe_list(raw.get("funding_history", []), FundingRound),
            buying_signals=safe_list(raw.get("buying_signals", []), BuyingSignal),
            tech_stack=safe_list(raw.get("tech_stack", []), TechStack),
            products_services=raw.get("products_services", []) if isinstance(raw.get("products_services"), list) else [],
            open_job_roles=[
                r if isinstance(r, str) else r.get("title", str(r))
                for r in raw.get("open_job_roles", [])
                if isinstance(r, (str, dict))
            ],
            pages_scraped=raw.get("pages_scraped", []) if isinstance(raw.get("pages_scraped"), list) else [],
            research_confidence=_safe_str(raw.get("research_confidence"), "low"),
            raw_text_summary=_safe_str(raw.get("raw_text_summary"), ""),
        )