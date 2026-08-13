"""
SalesNeuron — Identity Resolution
====================================
Step 1 of the email finder pipeline.

Takes messy input (LinkedIn URL, full name, company name, website)
and produces a clean, deduplicated PersonID stored in SQLite.

What it does:
  - Normalizes names (strips titles, extra spaces, unicode)
  - Extracts domain from any URL format
  - Deduplicates: same person entered twice = same PersonID
  - Verifies current employment via company team/about page (no LinkedIn scraping)
  - Stores everything in the `people` table in salesneuron.db

Usage:
    from core.identity import identity_resolver

    person = await identity_resolver.resolve(
        name="Harshil Mathur",
        company="Razorpay",
        website="https://razorpay.com",
        linkedin_url="https://linkedin.com/in/harshilmathur",
        title="CEO"
    )
    print(person["person_id"])   # stable int ID
    print(person["first_name"])  # "Harshil"
    print(person["domain"])      # "razorpay.com"
"""

import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))
EMPLOYMENT_STALE_DAYS = int(os.getenv("EMPLOYMENT_STALE_DAYS", "90"))

# Titles/prefixes to strip from names before processing
_NAME_PREFIXES = {
    "mr", "mrs", "ms", "dr", "prof", "sir", "mx",
    "mr.", "mrs.", "ms.", "dr.", "prof.",
}

# Common noise words that appear in "name" fields scraped from web
_NAME_NOISE = {
    "founder", "ceo", "cto", "coo", "vp", "director",
    "head", "manager", "lead", "senior", "junior",
    "linkedin", "profile", "connect",
}


class IdentityResolver:
    """
    Resolves a person to a stable PersonID.
    Deduplicates, normalizes, and optionally verifies employment.
    """

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False

    async def init(self):
        """Create people and employment_history tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name           TEXT NOT NULL,
                    first_name          TEXT,
                    last_name           TEXT,
                    domain              TEXT,
                    company_name        TEXT,
                    linkedin_url        TEXT,
                    title               TEXT,
                    employment_verified INTEGER DEFAULT 0,
                    last_verified_at    TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS employment_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   INTEGER NOT NULL,
                    company     TEXT,
                    domain      TEXT,
                    title       TEXT,
                    start_date  TEXT,
                    end_date    TEXT,
                    is_current  INTEGER DEFAULT 1,
                    source      TEXT,
                    added_at    TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES people(id)
                );

                CREATE INDEX IF NOT EXISTS idx_people_domain
                    ON people(domain);
                CREATE INDEX IF NOT EXISTS idx_people_name_domain
                    ON people(full_name, domain);
                CREATE INDEX IF NOT EXISTS idx_employment_person
                    ON employment_history(person_id);
            """)
            await db.commit()
        self._ready = True
        logger.info(f"🪪  IdentityResolver initialized → {self._db_path}")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def resolve(
        self,
        name: str,
        company: Optional[str] = None,
        website: Optional[str] = None,
        linkedin_url: Optional[str] = None,
        title: Optional[str] = None,
    ) -> dict:
        """
        Main entry point. Returns a dict with:
          person_id, full_name, first_name, last_name,
          domain, company_name, title, employment_verified
        """
        self._ensure_ready()

        # 1. Normalize all inputs
        clean_name = self._normalize_name(name)
        domain = self._extract_domain(website or "")
        clean_linkedin = self._normalize_linkedin(linkedin_url or "")
        clean_company = (company or "").strip()

        if not clean_name:
            logger.warning(f"🪪  Could not normalize name from input: {name!r} — skipping resolution")
            domain = self._extract_domain(website or "")
            parts = name.strip().split()
            return {
                "person_id": None,
                "full_name": name,
                "first_name": parts[0] if parts else "",
                "last_name": parts[-1] if len(parts) > 1 else "",
                "domain": domain,
                "company_name": company or "",
                "title": title or "",
                "linkedin_url": "",
                "employment_verified": False,
            }

        first, last = self._split_name(clean_name)

        # 2. Check for existing person (deduplicate)
        existing = await self._find_existing(clean_name, domain, clean_linkedin)
        if existing:
            logger.info(
                f"🪪  Person found in DB: {existing['full_name']} "
                f"(ID={existing['id']}, domain={existing['domain']})"
            )
            # Update any new fields we got this time
            await self._update_person(existing["id"], {
                "linkedin_url": clean_linkedin or existing["linkedin_url"],
                "title": title or existing["title"],
                "company_name": clean_company or existing["company_name"],
                "domain": domain or existing["domain"],
            })
            return self._format(existing, first, last)

        # 3. Create new person
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO people
                    (full_name, first_name, last_name, domain, company_name,
                     linkedin_url, title, employment_verified,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (clean_name, first, last, domain, clean_company,
                 clean_linkedin, title or "", now, now),
            )
            person_id = cursor.lastrowid

            # Log initial employment
            if domain or clean_company:
                await db.execute(
                    """
                    INSERT INTO employment_history
                        (person_id, company, domain, title, is_current, source, added_at)
                    VALUES (?, ?, ?, ?, 1, 'input', ?)
                    """,
                    (person_id, clean_company, domain, title or "", now),
                )

            await db.commit()

        logger.info(
            f"🪪  New person created: {clean_name} @ {domain} (ID={person_id})"
        )
        return {
            "person_id": person_id,
            "full_name": clean_name,
            "first_name": first,
            "last_name": last,
            "domain": domain,
            "company_name": clean_company,
            "title": title or "",
            "linkedin_url": clean_linkedin,
            "employment_verified": False,
        }

    async def verify_employment(
        self, person_id: int, scraper=None
    ) -> bool:
        """
        Check if person still works at their company by visiting the
        company's team/about/leadership page (NOT LinkedIn — ToS risk).

        Only re-checks if last verification was > EMPLOYMENT_STALE_DAYS ago.
        Returns True if verified (or skipped because still fresh).

        Pass scraper=BrowserScraper instance to enable web verification.
        If scraper=None, marks as unverified and returns False.
        """
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM people WHERE id = ?", (person_id,)
            ) as cur:
                person = await cur.fetchone()

        if not person:
            return False

        # Check freshness
        last_verified = person["last_verified_at"]
        if last_verified:
            age = datetime.now() - datetime.fromisoformat(last_verified)
            if age < timedelta(days=EMPLOYMENT_STALE_DAYS):
                logger.info(
                    f"🪪  Employment fresh for {person['full_name']} "
                    f"(verified {age.days}d ago)"
                )
                return True

        if scraper is None:
            logger.info(
                f"🪪  No scraper — skipping employment verification "
                f"for {person['full_name']}"
            )
            return False

        # Try to find person's name on company team/about pages
        domain = person["domain"]
        if not domain:
            return False

        team_pages = [
            f"https://{domain}/team",
            f"https://{domain}/about",
            f"https://{domain}/leadership",
            f"https://{domain}/people",
            f"https://{domain}/about-us",
        ]

        name = person["full_name"].lower()
        first = (person["first_name"] or "").lower()

        for page_url in team_pages:
            try:
                result = await scraper.scrape(page_url)
                if not result["success"]:
                    continue
                text = result.get("text", "").lower()
                # If we find their name on the team page, they still work there
                if name in text or (first and first in text):
                    logger.info(
                        f"🪪  Employment verified: {person['full_name']} "
                        f"found on {page_url}"
                    )
                    await self._mark_verified(person_id)
                    return True
            except Exception as e:
                logger.debug(f"Team page check failed for {page_url}: {e}")
                continue

        logger.info(
            f"🪪  Could not verify employment for {person['full_name']} "
            f"on any team page — may have left or pages are gated"
        )
        return False

    async def get(self, person_id: int) -> Optional[dict]:
        """Fetch a person by ID."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM people WHERE id = ?", (person_id,)
            ) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        first, last = self._split_name(d.get("full_name", ""))
        return self._format(d, first, last)

    async def search(self, name: str = "", domain: str = "") -> list[dict]:
        """Search people by name fragment or domain."""
        self._ensure_ready()
        conditions, params = [], []
        if name:
            conditions.append("full_name LIKE ?")
            params.append(f"%{name}%")
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM people {where} ORDER BY created_at DESC LIMIT 50",
                params,
            ) as cur:
                rows = await cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            first, last = self._split_name(d.get("full_name", ""))
            result.append(self._format(d, first, last))
        return result

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    async def _find_existing(
        self, name: str, domain: str, linkedin_url: str
    ) -> Optional[dict]:
        """
        Find an existing person by LinkedIn URL (most reliable),
        then by name + domain, then by name alone.
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # 1. LinkedIn URL match (unique identifier)
            if linkedin_url:
                async with db.execute(
                    "SELECT * FROM people WHERE linkedin_url = ?", (linkedin_url,)
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    return dict(row)

            # 2. Name + domain match
            if domain:
                async with db.execute(
                    "SELECT * FROM people WHERE full_name = ? AND domain = ?",
                    (name, domain),
                ) as cur:
                    row = await cur.fetchone()
                if row:
                    return dict(row)

            # 3. Name-only match (same person, different company entry)
            async with db.execute(
                "SELECT * FROM people WHERE full_name = ? LIMIT 1", (name,)
            ) as cur:
                row = await cur.fetchone()
            if row:
                return dict(row)

        return None

    async def _update_person(self, person_id: int, fields: dict):
        """Update non-null fields for an existing person."""
        updates = {k: v for k, v in fields.items() if v}
        if not updates:
            return
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE people SET {set_clause} WHERE id = ?",
                list(updates.values()) + [person_id],
            )
            await db.commit()

    async def _mark_verified(self, person_id: int):
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE people SET employment_verified=1, last_verified_at=? WHERE id=?",
                (now, person_id),
            )
            await db.commit()

    def _normalize_name(self, name: str) -> str:
        """
        Strip titles, normalize unicode, remove noise words.
        'Dr. Harshil Mathur (CEO)' → 'Harshil Mathur'
        """
        # Normalize unicode (é → e)
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))

        # Strip parenthetical content: "Name (CEO)" → "Name"
        name = re.sub(r"\(.*?\)", "", name)

        # Strip LinkedIn-style suffixes: "Name - Company"
        name = re.sub(r"\s*[-|•]\s*.*$", "", name)

        # Split and filter words
        words = name.split()
        cleaned = []
        for word in words:
            w = word.strip(".,;:")
            if w.lower() in _NAME_PREFIXES:
                continue
            if w.lower() in _NAME_NOISE:
                continue
            if re.match(r"^\d+$", w):
                continue
            if len(w) > 1:
                cleaned.append(w)

        return " ".join(cleaned).strip()

    def _split_name(self, full_name: str) -> tuple[str, str]:
        """
        'Harshil Mathur' → ('Harshil', 'Mathur')
        'Shailendra Singh Rao' → ('Shailendra', 'Rao')
        """
        parts = full_name.strip().split()
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[-1]

    def _extract_domain(self, url: str) -> str:
        """
        'https://www.razorpay.com/about' → 'razorpay.com'
        'razorpay.com'                   → 'razorpay.com'
        """
        if not url:
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain.replace("www.", "").strip()

    def _normalize_linkedin(self, url: str) -> str:
        """
        Normalize LinkedIn URLs to canonical form.
        'https://www.linkedin.com/in/harshilmathur/' → 'linkedin.com/in/harshilmathur'
        """
        if not url:
            return ""
        url = url.lower().strip().rstrip("/")
        url = url.replace("https://", "").replace("http://", "").replace("www.", "")
        # Keep only the path part that matters
        if "linkedin.com/in/" in url:
            slug = url.split("linkedin.com/in/")[-1].split("/")[0].split("?")[0]
            return f"linkedin.com/in/{slug}"
        return url

    def _format(self, row: dict, first: str, last: str) -> dict:
        return {
            "person_id": row["id"],
            "full_name": row["full_name"],
            "first_name": first or row.get("first_name", ""),
            "last_name": last or row.get("last_name", ""),
            "domain": row.get("domain", ""),
            "company_name": row.get("company_name", ""),
            "title": row.get("title", ""),
            "linkedin_url": row.get("linkedin_url", ""),
            "employment_verified": bool(row.get("employment_verified", 0)),
        }

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "IdentityResolver not initialized. "
                "Call `await identity_resolver.init()` first."
            )


# Singleton
identity_resolver = IdentityResolver()