"""
SalesNeuron — Credential Store (encrypted, SQLite)
====================================================
Stores login credentials per-domain so the Navigator can authenticate
on ANY site — not just Internshala. Passwords are encrypted at rest
with Fernet symmetric encryption.

Shares the same SQLite DB as memory.py and graph_store.py.

Setup:
  Generate a key once and put it in .env as CREDENTIALS_KEY:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

  CREDENTIALS_KEY=<paste here>

  If CREDENTIALS_KEY is missing, a key is generated and written to
  data/.credentials_key automatically on first use (dev convenience —
  for production always set it via env).

Usage:
  from core.credentials import credentials

  await credentials.init()
  await credentials.save("internshala.com", email="me@x.com", password="secret")
  creds = await credentials.get("internshala.com")   # {"email":..., "password":...} or None
  await credentials.delete("internshala.com")
"""

import logging
import os
from pathlib import Path
from typing import Optional

import aiosqlite
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))
_KEY_FALLBACK_PATH = Path("data/.credentials_key")


def _load_or_create_key() -> bytes:
    env_key = os.getenv("CREDENTIALS_KEY")
    if env_key:
        return env_key.encode()

    if _KEY_FALLBACK_PATH.exists():
        return _KEY_FALLBACK_PATH.read_bytes()

    logger.warning(
        "⚠️  CREDENTIALS_KEY not set in .env — generating a local dev key at "
        f"{_KEY_FALLBACK_PATH}. Set CREDENTIALS_KEY in .env for production."
    )
    _KEY_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    _KEY_FALLBACK_PATH.write_bytes(key)
    return key


class CredentialStore:
    """Encrypted, per-domain credential storage. Works for any website."""

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False
        self._fernet: Optional[Fernet] = None

    async def init(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(_load_or_create_key())

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain          TEXT UNIQUE NOT NULL,
                    email_enc       TEXT NOT NULL,
                    password_enc    TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );
            """)
            await db.commit()

        self._ready = True
        logger.info(f"🔐 CredentialStore initialized → {self._db_path.resolve()}")

    async def save(self, domain: str, email: str, password: str):
        """Store (or overwrite) credentials for a domain, encrypted."""
        self._ensure_ready()
        domain = self._normalize_domain(domain)
        from datetime import datetime
        now = datetime.now().isoformat()

        email_enc = self._fernet.encrypt(email.encode()).decode()
        password_enc = self._fernet.encrypt(password.encode()).decode()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO credentials (domain, email_enc, password_enc, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    email_enc = excluded.email_enc,
                    password_enc = excluded.password_enc,
                    updated_at = excluded.updated_at
                """,
                (domain, email_enc, password_enc, now, now),
            )
            await db.commit()

        logger.info(f"🔐 Credentials saved for {domain}")

    async def get(self, domain: str) -> Optional[dict]:
        """Return {"email": ..., "password": ...} or None if not stored."""
        self._ensure_ready()
        domain = self._normalize_domain(domain)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT email_enc, password_enc FROM credentials WHERE domain = ?",
                (domain,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        try:
            return {
                "email": self._fernet.decrypt(row["email_enc"].encode()).decode(),
                "password": self._fernet.decrypt(row["password_enc"].encode()).decode(),
            }
        except Exception as e:
            logger.warning(f"🔐 Failed to decrypt credentials for {domain}: {e}")
            return None

    async def exists(self, domain: str) -> bool:
        self._ensure_ready()
        domain = self._normalize_domain(domain)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM credentials WHERE domain = ?", (domain,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def delete(self, domain: str):
        self._ensure_ready()
        domain = self._normalize_domain(domain)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM credentials WHERE domain = ?", (domain,))
            await db.commit()
        logger.info(f"🔐 Credentials deleted for {domain}")

    async def list_domains(self) -> list[str]:
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT domain FROM credentials") as cursor:
                rows = await cursor.fetchall()
        return [r[0] for r in rows]

    def _normalize_domain(self, domain: str) -> str:
        domain = domain.replace("https://", "").replace("http://", "")
        domain = domain.replace("www.", "").rstrip("/")
        return domain.lower()

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "CredentialStore not initialized. Call `await credentials.init()` first."
            )


# Singleton
credentials = CredentialStore()
