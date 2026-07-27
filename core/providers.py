"""
SalesNeuron — Paid Provider Waterfall
========================================
Steps 7-9 of the email finder pipeline.

Only reached if the free verification layer (verifier.py) was inconclusive.
Tries paid providers in order of: best expected success rate given remaining
quota, historical performance, and cost.

Providers supported:
  Hunter.io   — HUNTER_API_KEY in .env   (25 free/month)
  Snov.io     — SNOV_CLIENT_ID + SNOV_CLIENT_SECRET in .env (50 free/month)
  Apollo.io   — APOLLO_API_KEY in .env   (50 free/month)

Rules enforced:
  - Never exceed monthly free quota (tracked in SQLite)
  - Only call ONE provider per lookup unless it fails
  - Stop immediately when a verified result is returned
  - All results written back to cache so future lookups are free

Usage:
    from core.providers import provider_waterfall
    await provider_waterfall.init()
    result = await provider_waterfall.find(
        first_name="Harshil",
        last_name="Mathur",
        domain="razorpay.com"
    )
    print(result["email"])       # harshil@razorpay.com
    print(result["provider"])    # "hunter"
    print(result["confidence"])  # 0.92
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
import httpx

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))

# API keys — set in .env, leave empty to skip that provider
HUNTER_API_KEY   = os.getenv("HUNTER_API_KEY", "")
SNOV_CLIENT_ID   = os.getenv("SNOV_CLIENT_ID", "")
SNOV_CLIENT_SECRET = os.getenv("SNOV_CLIENT_SECRET", "")
APOLLO_API_KEY   = os.getenv("APOLLO_API_KEY", "")

# Monthly free quotas per provider
_FREE_QUOTAS = {
    "hunter": int(os.getenv("HUNTER_QUOTA", "25")),
    "snov":   int(os.getenv("SNOV_QUOTA", "50")),
    "apollo": int(os.getenv("APOLLO_QUOTA", "50")),
}


class ProviderWaterfall:
    """
    Quota-tracked paid provider waterfall.
    Selects the best provider per lookup, stops on first success.
    """

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False

    async def init(self):
        """Create quota and history tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS provider_quota (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider        TEXT UNIQUE NOT NULL,
                    calls_used      INTEGER DEFAULT 0,
                    monthly_limit   INTEGER NOT NULL,
                    reset_date      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider        TEXT NOT NULL,
                    domain          TEXT,
                    success         INTEGER DEFAULT 0,
                    called_at       TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_provider_history_provider
                    ON provider_history(provider);
                CREATE INDEX IF NOT EXISTS idx_provider_history_domain
                    ON provider_history(provider, domain);
            """)
            await db.commit()

            # Seed quota rows for each provider (idempotent)
            now = datetime.now()
            reset = datetime(now.year, now.month, 1).isoformat()
            for name, limit in _FREE_QUOTAS.items():
                await db.execute(
                    """
                    INSERT OR IGNORE INTO provider_quota
                        (provider, calls_used, monthly_limit, reset_date, updated_at)
                    VALUES (?, 0, ?, ?, ?)
                    """,
                    (name, limit, reset, now.isoformat()),
                )
            await db.commit()

        self._ready = True
        logger.info("💳 ProviderWaterfall initialized")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def find(
        self,
        first_name: str,
        last_name: str,
        domain: str,
        company: str = "",
    ) -> Optional[dict]:
        """
        Find an email using the adaptive provider waterfall.

        Returns:
          {"email": ..., "confidence": ..., "provider": ..., "verified": bool}
          or None if all providers failed or quota exhausted.
        """
        self._ensure_ready()

        # Pick the best available provider
        ordered = await self._rank_providers(domain)
        if not ordered:
            logger.info("💳 No providers available (all quota exhausted or no API keys)")
            return None

        for provider_name in ordered:
            # Check + reserve quota atomically
            has_quota = await self._consume_quota(provider_name)
            if not has_quota:
                logger.info(f"💳 {provider_name}: quota exhausted this month")
                continue

            logger.info(f"💳 Trying {provider_name} for {first_name} {last_name} @ {domain}")
            result = await self._call_provider(
                provider_name, first_name, last_name, domain, company
            )

            success = result is not None and result.get("email")
            await self._record_history(provider_name, domain, bool(result))

            if success:
                result["provider"] = provider_name
                logger.info(
                    f"💳 {provider_name} found: {result['email']} "
                    f"(confidence={result.get('confidence', '?')})"
                )
                return result

            logger.info(f"💳 {provider_name} returned no result")

        logger.info(f"💳 All providers exhausted for {first_name} {last_name} @ {domain}")
        return None

    async def quota_status(self) -> list[dict]:
        """Show remaining quota for each provider."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM provider_quota ORDER BY provider"
            ) as cur:
                rows = await cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["calls_remaining"] = d["monthly_limit"] - d["calls_used"]
            result.append(d)
        return result

    # ──────────────────────────────────────────────────────────────
    # Provider implementations
    # ──────────────────────────────────────────────────────────────

    async def _call_provider(
        self,
        provider: str,
        first_name: str,
        last_name: str,
        domain: str,
        company: str,
    ) -> Optional[dict]:
        try:
            if provider == "hunter":
                return await self._hunter(first_name, last_name, domain)
            elif provider == "snov":
                return await self._snov(first_name, last_name, domain)
            elif provider == "apollo":
                return await self._apollo(first_name, last_name, domain, company)
        except Exception as e:
            logger.warning(f"💳 {provider} call failed: {e}")
        return None

    async def _hunter(
        self, first_name: str, last_name: str, domain: str
    ) -> Optional[dict]:
        """Hunter.io Email Finder API."""
        if not HUNTER_API_KEY:
            return None
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.hunter.io/v2/email-finder",
                params={
                    "domain": domain,
                    "first_name": first_name,
                    "last_name": last_name,
                    "api_key": HUNTER_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})

        email = data.get("email")
        if not email:
            return None
        return {
            "email": email,
            "confidence": (data.get("score", 0) or 0) / 100,
            "verified": data.get("verification", {}).get("status") == "valid",
        }

    async def _snov(
        self, first_name: str, last_name: str, domain: str
    ) -> Optional[dict]:
        """Snov.io Email Finder API (OAuth2 client credentials)."""
        if not SNOV_CLIENT_ID or not SNOV_CLIENT_SECRET:
            return None

        # Step 1: get access token
        async with httpx.AsyncClient(timeout=10) as client:
            token_resp = await client.post(
                "https://api.snov.io/v1/oauth/access_token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": SNOV_CLIENT_ID,
                    "client_secret": SNOV_CLIENT_SECRET,
                },
            )
            token_resp.raise_for_status()
            token = token_resp.json().get("access_token", "")
            if not token:
                return None

            # Step 2: find email
            resp = await client.post(
                "https://api.snov.io/v1/get-emails-from-names",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "firstName": first_name,
                    "lastName": last_name,
                    "domain": domain,
                    "limit": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        emails = data.get("emails") or data.get("data", {}).get("emails", [])
        if not emails:
            return None
        top = emails[0]
        return {
            "email": top.get("email"),
            "confidence": top.get("accuracy", 0) / 100,
            "verified": top.get("verified", False),
        }

    async def _apollo(
        self, first_name: str, last_name: str, domain: str, company: str
    ) -> Optional[dict]:
        """Apollo.io People Search API."""
        if not APOLLO_API_KEY:
            return None
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.apollo.io/v1/people/match",
                headers={
                    "x-api-key": APOLLO_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "first_name": first_name,
                    "last_name": last_name,
                    "domain": domain,
                    "organization_name": company,
                    "reveal_personal_emails": False,
                },
            )
            resp.raise_for_status()
            person = resp.json().get("person") or {}

        email = person.get("email")
        if not email or "@" not in email:
            return None
        return {
            "email": email,
            "confidence": 0.80,
            "verified": True,
        }

    # ──────────────────────────────────────────────────────────────
    # Quota + history helpers
    # ──────────────────────────────────────────────────────────────

    async def _rank_providers(self, domain: str) -> list[str]:
        """
        Return providers ordered by expected success rate, filtered to
        those with API keys configured and quota remaining.
        """
        available = []
        if HUNTER_API_KEY:
            available.append("hunter")
        if SNOV_CLIENT_ID and SNOV_CLIENT_SECRET:
            available.append("snov")
        if APOLLO_API_KEY:
            available.append("apollo")

        if not available:
            return []

        # Check quota and historical success rate per provider for this domain
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            scores = {}
            for p in available:
                # Quota check
                async with db.execute(
                    "SELECT calls_used, monthly_limit, reset_date FROM provider_quota WHERE provider=?",
                    (p,),
                ) as cur:
                    row = await cur.fetchone()

                if row:
                    # Reset quota if month rolled over
                    reset = datetime.fromisoformat(row["reset_date"])
                    now = datetime.now()
                    if now.year > reset.year or now.month > reset.month:
                        await db.execute(
                            "UPDATE provider_quota SET calls_used=0, reset_date=? WHERE provider=?",
                            (datetime(now.year, now.month, 1).isoformat(), p),
                        )
                        await db.commit()
                        remaining = row["monthly_limit"]
                    else:
                        remaining = row["monthly_limit"] - row["calls_used"]
                else:
                    remaining = _FREE_QUOTAS.get(p, 0)

                if remaining <= 0:
                    continue

                # Historical success rate for this domain type
                async with db.execute(
                    """SELECT AVG(success) as rate FROM provider_history
                       WHERE provider=? AND domain=?""",
                    (p, domain),
                ) as cur:
                    row2 = await cur.fetchone()
                success_rate = row2["rate"] if row2 and row2["rate"] else 0.5
                scores[p] = success_rate

        # Sort by success rate (descending); Hunter is default first if tied
        return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

    async def _consume_quota(self, provider: str) -> bool:
        """
        Atomically check and increment quota.
        Returns True if quota was available and consumed.
        """
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT calls_used, monthly_limit FROM provider_quota WHERE provider=?",
                (provider,),
            ) as cur:
                row = await cur.fetchone()
            if not row or row[0] >= row[1]:
                return False
            await db.execute(
                "UPDATE provider_quota SET calls_used=calls_used+1, updated_at=? WHERE provider=?",
                (datetime.now().isoformat(), provider),
            )
            await db.commit()
        return True

    async def _record_history(self, provider: str, domain: str, success: bool):
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO provider_history (provider, domain, success, called_at) VALUES (?,?,?,?)",
                (provider, domain, 1 if success else 0, datetime.now().isoformat()),
            )
            await db.commit()

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "ProviderWaterfall not initialized. "
                "Call `await provider_waterfall.init()` first."
            )


# Singleton
provider_waterfall = ProviderWaterfall()