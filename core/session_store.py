"""
SalesNeuron — Session Store
=============================
Saves Playwright browser storage_state (cookies + localStorage) per
domain to disk, so the Navigator doesn't have to log in on every run.

Works for ANY site — generic by design.

Usage:
  from core.session_store import session_store

  state_path = session_store.path_for("internshala.com")
  if session_store.is_fresh("internshala.com"):
      context = await browser.new_context(storage_state=str(state_path))
  else:
      context = await browser.new_context()
      # ... perform login ...
      await session_store.save("internshala.com", context)
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "data/sessions"))
SESSION_MAX_AGE_HOURS = int(os.getenv("SESSION_MAX_AGE_HOURS", "12"))


class SessionStore:
    """Persists browser login sessions to disk, per domain."""

    def __init__(self):
        self._dir = SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, domain: str) -> Path:
        domain = self._normalize_domain(domain)
        return self._dir / f"{domain}.json"

    def is_fresh(self, domain: str) -> bool:
        """True if a saved session exists and isn't too old."""
        path = self.path_for(domain)
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        fresh = age < timedelta(hours=SESSION_MAX_AGE_HOURS)
        if not fresh:
            logger.info(f"🍪 Session for {domain} is {age} old — will re-login")
        return fresh

    async def save(self, domain: str, context) -> None:
        """Save the current browser context's cookies/storage to disk."""
        path = self.path_for(domain)
        await context.storage_state(path=str(path))
        logger.info(f"🍪 Session saved for {domain} → {path}")

    async def load(self, domain: str) -> Optional[list[dict]]:
        """
        Load saved cookies for a domain as a list of Playwright-format
        cookie dicts ({name, value, domain, path, ...}) — the same shape
        found under storage_state()["cookies"]. Returns None if there's
        no saved session or it's gone stale (see SESSION_MAX_AGE_HOURS).

        Used by anything that wants raw cookies without a full Playwright
        context — e.g. injecting into a curl-cffi session for LinkedIn.
        """
        if not self.is_fresh(domain):
            return None
        path = self.path_for(domain)
        try:
            import json
            data = json.loads(path.read_text())
            cookies = data.get("cookies", [])
            if not cookies:
                logger.debug(f"🍪 Session file for {domain} has no cookies")
                return None
            return cookies
        except Exception as e:
            logger.warning(f"🍪 Failed to load session for {domain}: {e}")
            return None

    def clear(self, domain: str) -> None:
        path = self.path_for(domain)
        if path.exists():
            path.unlink()
            logger.info(f"🍪 Session cleared for {domain}")

    def _normalize_domain(self, domain: str) -> str:
        domain = domain.replace("https://", "").replace("http://", "")
        domain = domain.replace("www.", "").rstrip("/")
        return domain.lower()


# Singleton
session_store = SessionStore()