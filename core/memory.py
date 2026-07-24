"""
SalesNeuron — Persistent Memory (SQLite)
==========================================
Stores every scraped ProspectProfile in a local SQLite database.
The Researcher Agent checks here BEFORE scraping — if fresh data
exists it returns instantly without touching the browser or LLM.

Tables:
  prospects        — one row per company (full profile as JSON)
  buying_signals   — normalized signals for fast querying
  scrape_log       — every scrape attempt logged (success/fail/skipped)

Freshness:
  Default = 7 days. Re-scrapes automatically when data is stale.
  Override with CACHE_DAYS=3 in your .env

Usage:
  from core.memory import memory
  await memory.init()

  profile = await memory.get("https://stripe.com")   # None if not found/stale
  await memory.save(profile)
  companies = await memory.search(signal_type="recent_funding")
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiosqlite

from core.models import ProspectProfile

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))
CACHE_DAYS = int(os.getenv("CACHE_DAYS", "7"))


class Memory:
    """
    Async SQLite memory store for prospect profiles.
    Call await memory.init() once at startup before using.
    """

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False

    async def init(self):
        """Create DB file and tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS prospects (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    website         TEXT UNIQUE NOT NULL,
                    company_name    TEXT,
                    industry        TEXT,
                    headquarters    TEXT,
                    company_size    TEXT,
                    confidence      TEXT,
                    profile_json    TEXT NOT NULL,
                    scraped_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS buying_signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    website         TEXT NOT NULL,
                    company_name    TEXT,
                    signal_type     TEXT NOT NULL,
                    description     TEXT,
                    strength        TEXT,
                    source_url      TEXT,
                    scraped_at      TEXT NOT NULL,
                    FOREIGN KEY (website) REFERENCES prospects(website)
                );

                CREATE TABLE IF NOT EXISTS scrape_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    website     TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    note        TEXT,
                    logged_at   TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_prospects_website
                    ON prospects(website);
                CREATE INDEX IF NOT EXISTS idx_signals_type
                    ON buying_signals(signal_type);
                CREATE INDEX IF NOT EXISTS idx_signals_strength
                    ON buying_signals(strength);
            """)
            await db.commit()

        self._ready = True
        logger.info(f"💾 Memory initialized → {self._db_path.resolve()}")

    # ──────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────

    async def get(self, website: str) -> Optional[ProspectProfile]:
        """
        Return cached profile if it exists AND is fresh (< CACHE_DAYS old).
        Returns None if not found or stale — caller should re-scrape.
        """
        self._ensure_ready()
        website = self._normalize_url(website)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT profile_json, scraped_at FROM prospects WHERE website = ?",
                (website,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            logger.info(f"💾 Cache MISS — {website} not in DB")
            return None

        # Check freshness
        scraped_at = datetime.fromisoformat(row["scraped_at"])
        age = datetime.now() - scraped_at
        if age > timedelta(days=CACHE_DAYS):
            logger.info(
                f"💾 Cache STALE — {website} is {age.days} days old "
                f"(limit={CACHE_DAYS}d), will re-scrape"
            )
            await self._log(website, "stale", f"Age: {age.days} days")
            return None

        logger.info(
            f"💾 Cache HIT — {website} "
            f"(scraped {age.seconds // 3600}h {(age.seconds % 3600) // 60}m ago)"
        )
        await self._log(website, "cache_hit", f"Age: {age.days}d {age.seconds//3600}h")

        try:
            data = json.loads(row["profile_json"])
            return ProspectProfile(**data)
        except Exception as e:
            logger.warning(f"💾 Failed to deserialize cached profile: {e}")
            return None

    async def exists(self, website: str) -> bool:
        """Quick check — is this company in the DB at all (fresh or stale)?"""
        self._ensure_ready()
        website = self._normalize_url(website)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM prospects WHERE website = ?", (website,)
            ) as cursor:
                return await cursor.fetchone() is not None

    # ──────────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────────

    async def save(self, profile: ProspectProfile):
        """
        Insert or update a prospect profile.
        Also writes all buying signals to their own table for querying.
        """
        self._ensure_ready()
        website = self._normalize_url(profile.website)
        now = datetime.now().isoformat()
        profile_json = profile.model_dump_json()

        async with aiosqlite.connect(self._db_path) as db:
            # Upsert the main profile
            await db.execute(
                """
                INSERT INTO prospects
                    (website, company_name, industry, headquarters,
                     company_size, confidence, profile_json, scraped_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(website) DO UPDATE SET
                    company_name  = excluded.company_name,
                    industry      = excluded.industry,
                    headquarters  = excluded.headquarters,
                    company_size  = excluded.company_size,
                    confidence    = excluded.confidence,
                    profile_json  = excluded.profile_json,
                    updated_at    = excluded.updated_at
                """,
                (
                    website,
                    profile.company_name,
                    profile.industry,
                    profile.headquarters,
                    profile.company_size,
                    profile.research_confidence,
                    profile_json,
                    now,
                    now,
                ),
            )

            # Delete old signals for this company, re-insert fresh ones
            await db.execute(
                "DELETE FROM buying_signals WHERE website = ?", (website,)
            )
            for signal in profile.buying_signals:
                await db.execute(
                    """
                    INSERT INTO buying_signals
                        (website, company_name, signal_type, description,
                         strength, source_url, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        website,
                        profile.company_name,
                        signal.signal_type,
                        signal.description,
                        signal.strength,
                        signal.source_url,
                        now,
                    ),
                )

            await db.commit()

        await self._log(website, "saved", f"{len(profile.buying_signals)} signals")
        logger.info(f"💾 Saved → {profile.company_name} ({website})")

    # ──────────────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────────────

    async def search(
        self,
        signal_type: Optional[str] = None,
        strength: Optional[str] = None,
        industry: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Query stored companies by signal type, strength, or industry.

        Examples:
          await memory.search(signal_type="recent_funding")
          await memory.search(strength="high")
          await memory.search(industry="SaaS", signal_type="hiring_surge")
        """
        self._ensure_ready()

        conditions = []
        params = []

        if signal_type:
            conditions.append("s.signal_type = ?")
            params.append(signal_type)
        if strength:
            conditions.append("s.strength = ?")
            params.append(strength)
        if industry:
            conditions.append("p.industry LIKE ?")
            params.append(f"%{industry}%")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT DISTINCT
                p.company_name,
                p.website,
                p.industry,
                p.headquarters,
                p.confidence,
                p.updated_at,
                s.signal_type,
                s.description,
                s.strength
            FROM prospects p
            LEFT JOIN buying_signals s ON p.website = s.website
            {where}
            ORDER BY p.updated_at DESC
            LIMIT ?
        """

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def all_companies(self) -> list[dict]:
        """Return a summary of all stored companies."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT
                    company_name, website, industry,
                    headquarters, company_size, confidence, updated_at,
                    (SELECT COUNT(*) FROM buying_signals s
                     WHERE s.website = p.website) as signal_count
                FROM prospects p
                ORDER BY updated_at DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def delete(self, website: str):
        """Remove a company from the DB (force re-scrape next time)."""
        self._ensure_ready()
        website = self._normalize_url(website)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM prospects WHERE website = ?", (website,))
            await db.execute("DELETE FROM buying_signals WHERE website = ?", (website,))
            await db.commit()
        logger.info(f"💾 Deleted — {website}")

    async def stats(self) -> dict:
        """Return DB statistics."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM prospects") as c:
                total = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM buying_signals") as c:
                signals = (await c.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM prospects WHERE updated_at > datetime('now', ?)",
                (f"-{CACHE_DAYS} days",)
            ) as c:
                fresh = (await c.fetchone())[0]

        return {
            "total_companies": total,
            "fresh_companies": fresh,
            "stale_companies": total - fresh,
            "total_signals": signals,
            "cache_days": CACHE_DAYS,
            "db_path": str(self._db_path.resolve()),
        }

    # ──────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────

    async def _log(self, website: str, status: str, note: str = ""):
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO scrape_log (website, status, note, logged_at) VALUES (?, ?, ?, ?)",
                (website, status, note, datetime.now().isoformat()),
            )
            await db.commit()

    def _normalize_url(self, url: str) -> str:
        """Strip trailing slash for consistent DB keys."""
        return url.rstrip("/").lower()

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "Memory not initialized. Call `await memory.init()` before using."
            )


# Singleton — import this everywhere
memory = Memory()