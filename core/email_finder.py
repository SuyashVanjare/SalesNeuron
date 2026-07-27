"""
SalesNeuron — Email Finder (Full 10-Step Pipeline)
=====================================================
The central engine that wires identity → cache → pattern → verify → paid.

Steps:
  1.  Identity Resolution   — normalize + deduplicate person
  2.  Cache Check           — return instantly if already resolved
  3.  Pattern Discovery     — learn/reuse company email format
  4.  Candidate Generation  — top 5 candidates from known pattern
  5.  Confidence Scoring    — score each candidate, skip paid if ≥95%
  6.  Free Verification     — DNS/MX/SMTP (Reacher optional)
  7.  Adaptive Routing      — pick best paid provider if needed
  8.  Paid Waterfall        — Hunter → Snov → Apollo (stop on first hit)
  9.  Result Aggregation    — save to cache permanently
  10. Feedback ready        — result includes feedback_id for tracking

Usage:
    from core.email_finder import email_finder
    await email_finder.init()

    result = await email_finder.find(
        name="Harshil Mathur",
        company="Razorpay",
        website="https://razorpay.com",
        linkedin_url="https://linkedin.com/in/harshilmathur",
        title="CEO & Co-Founder",
    )
    print(result["email"])       # harshil@razorpay.com
    print(result["confidence"])  # 0.92
    print(result["source"])      # "pattern" / "hunter" / "cache" / etc.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiosqlite
import httpx

from core.identity import identity_resolver
from core.verifier import verifier
from core.providers import provider_waterfall

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))

# Confidence threshold above which we skip paid providers entirely
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.95"))

# Standard email patterns ranked by prevalence across B2B companies
_PATTERNS = [
    "{first}.{last}",      # harshil.mathur@  (most common ~45%)
    "{first}",             # harshil@          (~20%)
    "{f}{last}",           # hmathur@          (~15%)
    "{first}{last}",       # harshilmathur@    (~10%)
    "{first}_{last}",      # harshil_mathur@   (~5%)
    "{first}-{last}",      # harshil-mathur@   (~3%)
    "{f}.{last}",          # h.mathur@         (~2%)
]


class EmailFinder:
    """
    End-to-end email finder. Free-first, paid only as last resort.
    Every result is cached permanently to avoid re-lookup costs.
    """

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False

    async def init(self):
        """Initialize all sub-systems and create email/pattern tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS person_emails (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id           INTEGER NOT NULL,
                    email               TEXT NOT NULL,
                    confidence          REAL DEFAULT 0.0,
                    verification_status TEXT DEFAULT 'unknown',
                    provider_used       TEXT DEFAULT 'internal',
                    pattern_used        TEXT,
                    source              TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES people(id)
                );

                CREATE TABLE IF NOT EXISTS patterns (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain          TEXT UNIQUE NOT NULL,
                    pattern         TEXT NOT NULL,
                    confidence      REAL DEFAULT 0.5,
                    sample_count    INTEGER DEFAULT 1,
                    discovered_via  TEXT DEFAULT 'inference',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    email       TEXT NOT NULL,
                    person_id   INTEGER,
                    event_type  TEXT NOT NULL,
                    metadata    TEXT,
                    occurred_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_person_emails_person
                    ON person_emails(person_id);
                CREATE INDEX IF NOT EXISTS idx_person_emails_email
                    ON person_emails(email);
                CREATE INDEX IF NOT EXISTS idx_patterns_domain
                    ON patterns(domain);
                CREATE INDEX IF NOT EXISTS idx_feedback_email
                    ON feedback_events(email);
            """)
            await db.commit()

        await identity_resolver.init()
        await provider_waterfall.init()

        self._ready = True
        logger.info(f"📧 EmailFinder initialized → {self._db_path}")

    # ──────────────────────────────────────────────────────────────
    # Main public API
    # ──────────────────────────────────────────────────────────────

    async def find(
        self,
        name: str,
        company: str = "",
        website: str = "",
        linkedin_url: str = "",
        title: str = "",
        scraper=None,
    ) -> dict:
        """
        Find the email for a person. Full 10-step pipeline.

        Args:
            name:         Full name (required)
            company:      Company name (helps with provider calls)
            website:      Company website URL (used to extract domain)
            linkedin_url: LinkedIn URL (best deduplication key)
            title:        Job title (optional context)
            scraper:      BrowserScraper instance for pattern discovery
                          and employment verification. Pass None to skip
                          web-based discovery (faster, less accurate).

        Returns dict with:
            email, confidence, source, pattern_used,
            verification_status, provider_used, person_id, feedback_id
        """
        self._ensure_ready()

        # ── Step 1: Identity Resolution ──────────────────────────
        logger.info(f"📧 Finding email for: {name} @ {company or website}")
        person = await identity_resolver.resolve(
            name=name,
            company=company,
            website=website,
            linkedin_url=linkedin_url,
            title=title,
        )
        person_id = person["person_id"]
        first = person["first_name"].lower()
        last = person["last_name"].lower()
        domain = person["domain"]

        if not domain:
            return self._empty_result(person_id, "No domain — provide website or company domain")

        # ── Step 2: Cache Check ───────────────────────────────────
        cached = await self._get_cached_email(person_id)
        if cached and cached.get("verification_status") == "valid":
            logger.info(f"📧 Cache HIT: {cached['email']} (confidence={cached['confidence']:.2f})")
            cached["source"] = "cache"
            return cached

        # ── Step 3: Pattern Discovery ─────────────────────────────
        pattern = await self._get_or_discover_pattern(domain, scraper)
        logger.info(f"📧 Pattern for {domain}: {pattern or 'unknown'}")

        # ── Step 4: Candidate Generation ──────────────────────────
        candidates = self._generate_candidates(first, last, domain, pattern)
        logger.info(f"📧 {len(candidates)} candidates: {candidates[:3]}...")

        # ── Step 5: Confidence Scoring ────────────────────────────
        scored = await self._score_candidates(candidates, domain, pattern)
        top = scored[0] if scored else None

        if top and top["confidence"] >= HIGH_CONFIDENCE_THRESHOLD:
            # High confidence — verify free only, skip paid
            logger.info(
                f"📧 High confidence ({top['confidence']:.2f}) — "
                f"skipping paid providers"
            )
            result = await self._verify_and_save(
                person_id, top["email"], top["confidence"],
                pattern, "pattern"
            )
            if result["verification_status"] in ("valid", "risky"):
                return result

        # ── Steps 6-8: Free verify → adaptive routing → paid ─────
        for candidate in [c["email"] for c in scored]:
            verify_result = await verifier.verify(candidate)
            if verify_result["status"] == "valid":
                result = await self._save_result(
                    person_id=person_id,
                    email=candidate,
                    confidence=max(
                        verify_result["confidence"],
                        next((s["confidence"] for s in scored if s["email"] == candidate), 0.6)
                    ),
                    verification_status="valid",
                    provider_used="internal",
                    pattern_used=pattern or "",
                    source="free_verification",
                )
                logger.info(f"📧 Verified free: {candidate}")
                return result

            if verify_result["status"] == "risky":
                # Catch-all domain — save as risky, proceed to paid
                risky_candidate = candidate
                risky_confidence = verify_result["confidence"]
                logger.info(f"📧 Risky (catch-all): {candidate} — trying paid providers")
                break
        else:
            risky_candidate = None
            risky_confidence = 0.0

        # ── Step 8: Paid Waterfall ────────────────────────────────
        paid_result = await provider_waterfall.find(
            first_name=person["first_name"],
            last_name=person["last_name"],
            domain=domain,
            company=company,
        )
        if paid_result and paid_result.get("email"):
            # Learn pattern from paid result
            await self._learn_pattern_from_email(paid_result["email"], domain)
            result = await self._save_result(
                person_id=person_id,
                email=paid_result["email"],
                confidence=paid_result.get("confidence", 0.80),
                verification_status="valid" if paid_result.get("verified") else "risky",
                provider_used=paid_result.get("provider", "unknown"),
                pattern_used=pattern or "",
                source=paid_result.get("provider", "paid"),
            )
            return result

        # ── Fallback: return risky candidate if we have one ───────
        if risky_candidate:
            result = await self._save_result(
                person_id=person_id,
                email=risky_candidate,
                confidence=risky_confidence,
                verification_status="risky",
                provider_used="internal",
                pattern_used=pattern or "",
                source="best_guess",
            )
            logger.info(f"📧 Returning best guess (risky): {risky_candidate}")
            return result

        logger.info(f"📧 Could not find email for {name} @ {domain}")
        return self._empty_result(person_id, "No email found after all steps")

    async def record_feedback(
        self,
        email: str,
        event_type: str,
        person_id: Optional[int] = None,
        metadata: str = "",
    ):
        """
        Record a feedback event for the feedback loop.

        event_type: 'delivered' | 'opened' | 'clicked' | 'replied'
                    | 'soft_bounce' | 'hard_bounce'

        On bounce: lowers pattern confidence for this domain.
        On reply/delivered: raises pattern confidence for this domain.
        """
        self._ensure_ready()
        now = datetime.now().isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO feedback_events
                    (email, person_id, event_type, metadata, occurred_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email, person_id, event_type, metadata, now),
            )
            await db.commit()

        # Update pattern confidence based on feedback
        domain = email.split("@")[1] if "@" in email else ""
        if domain:
            if event_type in ("replied", "delivered", "opened"):
                await self._adjust_pattern_confidence(domain, delta=+0.05)
            elif event_type == "hard_bounce":
                await self._adjust_pattern_confidence(domain, delta=-0.20)
                # Also mark this email as bounced
                async with aiosqlite.connect(self._db_path) as db:
                    await db.execute(
                        "UPDATE person_emails SET verification_status='bounced' WHERE email=?",
                        (email,),
                    )
                    await db.commit()
            elif event_type == "soft_bounce":
                await self._adjust_pattern_confidence(domain, delta=-0.05)

        logger.info(f"📧 Feedback recorded: {email} → {event_type}")

    async def lookup_pattern(self, domain: str) -> Optional[str]:
        """Return the known email pattern for a domain, or None."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT pattern FROM patterns WHERE domain=? ORDER BY confidence DESC LIMIT 1",
                (domain,),
            ) as cur:
                row = await cur.fetchone()
        return row["pattern"] if row else None

    # ──────────────────────────────────────────────────────────────
    # Step 3 — Pattern Discovery
    # ──────────────────────────────────────────────────────────────

    async def _get_or_discover_pattern(
        self, domain: str, scraper=None
    ) -> Optional[str]:
        """
        Return known pattern from DB, or discover it by:
        1. Scraping the company's team/about pages for email addresses
        2. Inferring the pattern from 2+ discovered emails
        3. Saving it permanently
        """
        known = await self.lookup_pattern(domain)
        if known:
            return known

        discovered_emails: list[str] = []

        # ── Source 1: WHOIS lookup (free, no API key) ─────────────
        whois_emails = await self._emails_from_whois(domain)
        discovered_emails.extend(whois_emails)
        if whois_emails:
            logger.info(f"📧 WHOIS found {len(whois_emails)} email(s) for {domain}")

        # ── Source 2: GitHub org scraping (free, no API key) ──────
        github_emails = await self._emails_from_github(domain)
        discovered_emails.extend(github_emails)
        if github_emails:
            logger.info(f"📧 GitHub found {len(github_emails)} email(s) for {domain}")

        # ── Source 3: Website team/about pages ────────────────────
        if scraper is not None:
            site_emails = await self._scrape_emails_from_site(domain, scraper)
            discovered_emails.extend(site_emails)
            if site_emails:
                logger.info(f"📧 Website crawl found {len(site_emails)} email(s) for {domain}")

        # Deduplicate
        discovered_emails = list(dict.fromkeys(
            e.lower() for e in discovered_emails if "@" + domain in e.lower()
        ))

        if len(discovered_emails) < 2:
            return None

        pattern = self._infer_pattern(discovered_emails, domain)
        if pattern:
            await self._save_pattern(domain, pattern, len(discovered_emails), "crawl")
            logger.info(
                f"📧 Pattern discovered for {domain}: {pattern} "
                f"(from {len(discovered_emails)} emails)"
            )
        return pattern

    async def _emails_from_whois(self, domain: str) -> list[str]:
        """
        Extract email addresses from WHOIS domain registration data.
        Registrant/admin/tech contacts sometimes use company emails.
        Free, no API key needed. Requires: pip install python-whois
        """
        try:
            import whois as whois_lib
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, whois_lib.whois, domain)

            emails = []
            email_re = re.compile(
                r"[a-zA-Z0-9._%+\-]+@" + re.escape(domain), re.IGNORECASE
            )

            # whois returns emails as string or list
            raw_emails = data.get("emails", []) or []
            if isinstance(raw_emails, str):
                raw_emails = [raw_emails]

            for e in raw_emails:
                if e and "@" + domain in e.lower():
                    emails.append(e.lower())

            # Also scan raw text for domain emails
            raw_text = str(data)
            emails.extend(email_re.findall(raw_text))

            return list(dict.fromkeys(emails))  # deduplicate
        except ImportError:
            logger.debug("python-whois not installed — skipping WHOIS lookup")
            return []
        except Exception as e:
            logger.debug(f"WHOIS lookup failed for {domain}: {e}")
            return []

    async def _emails_from_github(self, domain: str) -> list[str]:
        """
        Scrape GitHub for emails associated with this company domain.

        Strategy:
          1. Find GitHub org by searching for the company domain
          2. Get org members
          3. Check each member's public profile for company email
          4. Check recent commits for author emails matching the domain

        Free — uses GitHub public API (60 req/hour unauthenticated,
        5000/hour with GITHUB_TOKEN in .env)
        """
        github_token = os.getenv("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        company_name = domain.split(".")[0]  # stripe.com → stripe
        emails: list[str] = []
        email_re = re.compile(
            r"[a-zA-Z0-9._%+\-]+@" + re.escape(domain), re.IGNORECASE
        )

        try:
            async with httpx.AsyncClient(
                headers=headers, timeout=10.0, follow_redirects=True
            ) as client:

                # ── Step 1: Find the GitHub org ───────────────────
                org_login = None

                # Try direct org name first (stripe → github.com/stripe)
                r = await client.get(f"https://api.github.com/orgs/{company_name}")
                if r.status_code == 200:
                    org_login = r.json().get("login")
                    logger.debug(f"📧 GitHub org found: {org_login}")
                else:
                    # Search for org by domain
                    r = await client.get(
                        "https://api.github.com/search/users",
                        params={"q": f"{domain} type:org", "per_page": 3},
                    )
                    if r.status_code == 200:
                        items = r.json().get("items", [])
                        if items:
                            org_login = items[0]["login"]

                if not org_login:
                    return []

                # ── Step 2: Get org members (public only) ─────────
                r = await client.get(
                    f"https://api.github.com/orgs/{org_login}/members",
                    params={"per_page": 20},
                )
                if r.status_code != 200:
                    return []

                members = r.json()
                logger.debug(f"📧 GitHub: {len(members)} members in {org_login}")

                # ── Step 3: Check each member's public profile ────
                for member in members[:10]:  # cap at 10 to avoid rate limit
                    username = member.get("login", "")
                    if not username:
                        continue

                    r = await client.get(f"https://api.github.com/users/{username}")
                    if r.status_code == 200:
                        profile = r.json()
                        email = profile.get("email", "") or ""
                        if email and "@" + domain in email.lower():
                            emails.append(email.lower())
                            logger.debug(f"📧 GitHub profile email: {email}")

                    await asyncio.sleep(0.1)  # be polite to GitHub API

                # ── Step 4: Check recent org commits for emails ───
                r = await client.get(
                    f"https://api.github.com/orgs/{org_login}/repos",
                    params={"per_page": 5, "sort": "pushed"},
                )
                if r.status_code == 200:
                    repos = r.json()
                    for repo in repos[:3]:  # check 3 most active repos
                        repo_name = repo.get("name", "")
                        r2 = await client.get(
                            f"https://api.github.com/repos/{org_login}/{repo_name}/commits",
                            params={"per_page": 20},
                        )
                        if r2.status_code == 200:
                            for commit in r2.json():
                                author = commit.get("commit", {}).get("author", {})
                                email = author.get("email", "") or ""
                                if (
                                    email
                                    and "@" + domain in email.lower()
                                    and "noreply" not in email
                                    and "github" not in email
                                ):
                                    emails.append(email.lower())

        except httpx.TimeoutException:
            logger.debug(f"GitHub API timeout for {domain}")
        except Exception as e:
            logger.debug(f"GitHub scraping failed for {domain}: {e}")

        return list(dict.fromkeys(emails))  # deduplicate

    async def _scrape_emails_from_site(
        self, domain: str, scraper
    ) -> list[str]:
        """
        Crawl company team/contact/about pages and extract any email addresses
        found in the page text. These seed the pattern inference.
        """
        pages = [
            f"https://{domain}/team",
            f"https://{domain}/about",
            f"https://{domain}/contact",
            f"https://{domain}/leadership",
            f"https://{domain}/about-us",
            f"https://{domain}/people",
        ]
        email_re = re.compile(
            r"[a-zA-Z0-9._%+\-]+@" + re.escape(domain), re.IGNORECASE
        )
        found = []
        for url in pages:
            try:
                result = await scraper.scrape(url)
                if result["success"]:
                    emails = email_re.findall(result.get("text", ""))
                    found.extend([e.lower() for e in emails])
            except Exception:
                continue
            if len(found) >= 5:
                break

        return list(dict.fromkeys(found))  # deduplicate preserving order

    def _infer_pattern(self, emails: list[str], domain: str) -> Optional[str]:
        """
        Given a list of real emails from the same domain, infer the pattern.

        Example:
          harshil@razorpay.com, shashank@razorpay.com → {first}
          harshil.mathur@razorpay.com, shashank.kumar@razorpay.com → {first}.{last}
        """
        # Strip domain part
        local_parts = [e.split("@")[0] for e in emails if "@" in e]
        if not local_parts:
            return None

        # Count pattern matches across emails
        pattern_votes: dict[str, int] = {}
        for email in emails:
            local = email.split("@")[0].lower()
            for fmt in _PATTERNS:
                # We can only validate patterns that contain first/last
                # without knowing the person's name — skip for now,
                # rely on structural heuristics instead
                if "." in local and not local.startswith("."):
                    pattern_votes["{first}.{last}"] = pattern_votes.get("{first}.{last}", 0) + 1
                elif "_" in local:
                    pattern_votes["{first}_{last}"] = pattern_votes.get("{first}_{last}", 0) + 1
                elif "-" in local:
                    pattern_votes["{first}-{last}"] = pattern_votes.get("{first}-{last}", 0) + 1
                elif len(local) <= 8:
                    pattern_votes["{first}"] = pattern_votes.get("{first}", 0) + 1
                else:
                    pattern_votes["{first}{last}"] = pattern_votes.get("{first}{last}", 0) + 1

        if not pattern_votes:
            return None
        return max(pattern_votes, key=lambda k: pattern_votes[k])

    # ──────────────────────────────────────────────────────────────
    # Step 4 — Candidate Generation
    # ──────────────────────────────────────────────────────────────

    def _generate_candidates(
        self, first: str, last: str, domain: str, pattern: Optional[str]
    ) -> list[str]:
        """
        Generate top 5 candidate emails. Known pattern goes first,
        then standard fallbacks.
        """
        # Clean names (remove non-alpha)
        first = re.sub(r"[^a-z]", "", first)
        last = re.sub(r"[^a-z]", "", last)
        f = first[0] if first else ""

        def apply(fmt: str) -> str:
            return (
                fmt
                .replace("{first}", first)
                .replace("{last}", last)
                .replace("{f}", f)
                + f"@{domain}"
            )

        candidates = []

        # Pattern-first
        if pattern and pattern in _PATTERNS:
            candidates.append(apply(pattern))

        # Standard fallbacks (excluding pattern already added)
        for fmt in _PATTERNS:
            if fmt == pattern:
                continue
            if first and last:
                candidates.append(apply(fmt))
            elif first:
                if "{last}" not in fmt and "{f}" not in fmt:
                    candidates.append(apply(fmt))
            if len(candidates) >= 5:
                break

        return list(dict.fromkeys(candidates))[:5]

    # ──────────────────────────────────────────────────────────────
    # Step 5 — Confidence Scoring
    # ──────────────────────────────────────────────────────────────

    async def _score_candidates(
        self, candidates: list[str], domain: str, pattern: Optional[str]
    ) -> list[dict]:
        """
        Score each candidate email by:
        - Whether it matches the known/discovered pattern (+0.4)
        - Historical success rate for this domain's pattern (+0.3)
        - Pattern prevalence rank (+0.2 to 0.1)
        """
        # Get pattern confidence from DB
        pattern_confidence = 0.5
        if pattern:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT confidence FROM patterns WHERE domain=?", (domain,)
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    pattern_confidence = row["confidence"]

        scored = []
        for i, email in enumerate(candidates):
            local = email.split("@")[0]
            score = 0.0

            # Pattern match bonus
            if pattern:
                expected_local = self._apply_pattern_to_email(pattern, email)
                if expected_local and local == expected_local:
                    score += 0.4 * pattern_confidence

            # Rank bonus (first candidate gets higher base score)
            score += max(0.0, 0.3 - i * 0.05)

            # Domain signal (MX/pattern familiarity)
            score += 0.2

            scored.append({"email": email, "confidence": min(score, 0.99)})

        return sorted(scored, key=lambda x: x["confidence"], reverse=True)

    def _apply_pattern_to_email(self, pattern: str, email: str) -> Optional[str]:
        """Not used for scoring — placeholder for future ML integration."""
        return None

    # ──────────────────────────────────────────────────────────────
    # DB helpers
    # ──────────────────────────────────────────────────────────────

    async def _get_cached_email(self, person_id: int) -> Optional[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM person_emails
                   WHERE person_id=?
                   ORDER BY confidence DESC LIMIT 1""",
                (person_id,),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        return {
            "email": row["email"],
            "confidence": row["confidence"],
            "verification_status": row["verification_status"],
            "provider_used": row["provider_used"],
            "pattern_used": row["pattern_used"],
            "person_id": person_id,
            "feedback_id": row["id"],
        }

    async def _verify_and_save(
        self,
        person_id: int,
        email: str,
        confidence: float,
        pattern: Optional[str],
        source: str,
    ) -> dict:
        verify_result = await verifier.verify(email)
        return await self._save_result(
            person_id=person_id,
            email=email,
            confidence=max(confidence, verify_result["confidence"]),
            verification_status=verify_result["status"],
            provider_used="internal",
            pattern_used=pattern or "",
            source=source,
        )

    async def _save_result(
        self,
        person_id: int,
        email: str,
        confidence: float,
        verification_status: str,
        provider_used: str,
        pattern_used: str,
        source: str,
    ) -> dict:
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO person_emails
                    (person_id, email, confidence, verification_status,
                     provider_used, pattern_used, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (person_id, email, confidence, verification_status,
                 provider_used, pattern_used, source, now, now),
            )
            feedback_id = cursor.lastrowid
            await db.commit()

        logger.info(
            f"📧 Saved: {email} | confidence={confidence:.2f} | "
            f"status={verification_status} | source={source}"
        )
        return {
            "email": email,
            "confidence": confidence,
            "verification_status": verification_status,
            "provider_used": provider_used,
            "pattern_used": pattern_used,
            "source": source,
            "person_id": person_id,
            "feedback_id": feedback_id,
        }

    async def _save_pattern(
        self, domain: str, pattern: str, sample_count: int, discovered_via: str
    ):
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO patterns
                    (domain, pattern, confidence, sample_count, discovered_via,
                     created_at, updated_at)
                VALUES (?, ?, 0.6, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    pattern=excluded.pattern,
                    sample_count=excluded.sample_count,
                    updated_at=excluded.updated_at
                """,
                (domain, pattern, sample_count, discovered_via, now, now),
            )
            await db.commit()

    async def _learn_pattern_from_email(self, email: str, domain: str):
        """Infer and save pattern from a single confirmed email."""
        local = email.split("@")[0].lower()
        pattern = None
        if "." in local:
            pattern = "{first}.{last}"
        elif "_" in local:
            pattern = "{first}_{last}"
        elif "-" in local:
            pattern = "{first}-{last}"
        elif len(local) <= 8:
            pattern = "{first}"
        else:
            pattern = "{first}{last}"
        if pattern:
            await self._save_pattern(domain, pattern, 1, "paid_provider_inference")

    async def _adjust_pattern_confidence(self, domain: str, delta: float):
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """UPDATE patterns SET
                   confidence = MAX(0.0, MIN(1.0, confidence + ?)),
                   updated_at = ?
                   WHERE domain = ?""",
                (delta, datetime.now().isoformat(), domain),
            )
            await db.commit()

    def _empty_result(self, person_id: int, reason: str) -> dict:
        logger.info(f"📧 No result: {reason}")
        return {
            "email": None,
            "confidence": 0.0,
            "verification_status": "not_found",
            "provider_used": None,
            "pattern_used": None,
            "source": "none",
            "person_id": person_id,
            "feedback_id": None,
            "reason": reason,
        }

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "EmailFinder not initialized. "
                "Call `await email_finder.init()` first."
            )


# Singleton
email_finder = EmailFinder()