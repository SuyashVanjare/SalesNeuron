"""
SalesNeuron — Free Verification Layer
=======================================
Step 6 of the email finder pipeline.

Verifies email candidates using only free methods — always attempted
BEFORE any paid provider call.

Verification chain (in order):
  1. Syntax check         — is it a valid email format?
  2. Disposable check     — is the domain a throwaway service?
  3. MX record check      — does the domain actually receive email?
                            (requires dnspython — pip install dnspython)
  4. Reacher SMTP check   — optional self-hosted SMTP verifier
                            Set REACHER_URL=http://localhost:8080 in .env
                            If not set, this step is skipped gracefully.

Result codes:
  'valid'       — passed all available checks, high confidence
  'risky'       — MX exists but catch-all domain (SMTP accepted everything)
  'invalid'     — failed syntax or MX check
  'unknown'     — could not determine (DNS timeout, Reacher unavailable)

Usage:
    from core.verifier import verifier
    result = await verifier.verify("harshil@razorpay.com")
    print(result["status"])      # 'valid' / 'invalid' / 'risky' / 'unknown'
    print(result["confidence"])  # 0.0 - 1.0
    print(result["reason"])      # human-readable explanation
"""

import asyncio
import logging
import os
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

REACHER_URL = os.getenv("REACHER_URL", "")          # e.g. http://localhost:8080
REACHER_API_KEY = os.getenv("REACHER_API_KEY", "")  # if your Reacher has auth
SMTP_TIMEOUT = float(os.getenv("SMTP_TIMEOUT", "8"))
DNS_TIMEOUT = float(os.getenv("DNS_TIMEOUT", "5"))

# Known disposable/throwaway email domains — block these immediately
_DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "10minutemail.com",
    "tempmail.com", "throwaway.email", "yopmail.com",
    "trashmail.com", "fakeinbox.com", "sharklasers.com",
    "guerrillamailblock.com", "grr.la", "guerrillamail.info",
    "spam4.me", "trashmail.me", "mailnull.com", "spamgourmet.com",
    "dispostable.com", "discard.email", "getnada.com",
}

# Regex for RFC-5321 email syntax (simplified but covers 99.9% of real emails)
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


class EmailVerifier:
    """
    Free, layered email verification.
    Tries the cheapest checks first, escalates only if needed.
    """

    async def verify(self, email: str) -> dict:
        """
        Run the full free verification chain.
        Returns a result dict with status, confidence, and reason.
        """
        email = email.strip().lower()

        # ── 1. Syntax check ──────────────────────────────────────
        if not self._is_valid_syntax(email):
            return self._result("invalid", 0.0, f"Invalid email syntax: {email}")

        domain = email.split("@")[1]

        # ── 2. Disposable domain check ───────────────────────────
        if domain in _DISPOSABLE_DOMAINS:
            return self._result("invalid", 0.0, f"Disposable email domain: {domain}")

        # ── 3. MX record check ───────────────────────────────────
        mx_ok, mx_reason = await self._check_mx(domain)
        if not mx_ok:
            return self._result("invalid", 0.1, f"No MX records: {mx_reason}")

        # ── 4. Reacher SMTP check (optional) ─────────────────────
        if REACHER_URL:
            reacher_result = await self._check_reacher(email)
            if reacher_result:
                return reacher_result
            # Reacher unavailable or timed out — fall through to 'unknown'
            logger.info(f"🔍 Reacher unavailable for {email} — returning 'risky'")

        # Passed syntax + MX but no SMTP confirmation
        return self._result(
            "risky", 0.65,
            f"Syntax valid, MX exists for {domain}, no SMTP confirmation"
        )

    async def verify_batch(
        self, emails: list[str], delay_seconds: float = 1.0
    ) -> list[dict]:
        """
        Verify multiple emails with a delay between SMTP checks to avoid
        flagging the server's IP. Use delay_seconds=2.0 for production.
        """
        results = []
        for email in emails:
            result = await self.verify(email)
            results.append(result)
            if REACHER_URL and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
        return results

    # ──────────────────────────────────────────────────────────────
    # Check implementations
    # ──────────────────────────────────────────────────────────────

    def _is_valid_syntax(self, email: str) -> bool:
        return bool(_EMAIL_RE.match(email))

    async def _check_mx(self, domain: str) -> tuple[bool, str]:
        """
        Check if the domain has MX records (i.e. it receives email).
        Uses dnspython if available, falls back to a simple socket check.
        """
        # Try dnspython first (more reliable)
        try:
            import dns.resolver
            import dns.exception

            resolver = dns.resolver.Resolver()
            resolver.lifetime = DNS_TIMEOUT
            answers = resolver.resolve(domain, "MX")
            if answers:
                mx_hosts = [str(r.exchange).rstrip(".") for r in answers]
                logger.debug(f"🔍 MX for {domain}: {mx_hosts[:2]}")
                return True, f"MX: {mx_hosts[0]}"
            return False, "No MX records found"

        except ImportError:
            # dnspython not installed — fall back to basic socket check
            logger.debug(
                "dnspython not installed — using socket fallback for MX check. "
                "Install with: pip install dnspython"
            )
            return await self._check_mx_socket(domain)

        except Exception as e:
            logger.debug(f"🔍 DNS lookup failed for {domain}: {e}")
            return False, str(e)

    async def _check_mx_socket(self, domain: str) -> tuple[bool, str]:
        """
        Fallback MX check using a simple socket connection to port 25.
        Less reliable than DNS but works without dnspython.
        """
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: __import__("socket").getaddrinfo(domain, 25)
                ),
                timeout=DNS_TIMEOUT,
            )
            return True, f"Socket resolved {domain}:25"
        except Exception as e:
            return False, f"Socket check failed: {e}"

    async def _check_reacher(self, email: str) -> Optional[dict]:
        """
        Call self-hosted Reacher for SMTP-level verification.
        Reacher performs a full SMTP handshake without sending a real email.

        Returns a result dict if Reacher responded, None if unavailable.

        Run Reacher with Docker:
          docker run -p 8080:8080 reacherhq/core:latest

        Docs: https://reacher.email/docs
        """
        url = f"{REACHER_URL.rstrip('/')}/v0/check_email"
        headers = {"Content-Type": "application/json"}
        if REACHER_API_KEY:
            headers["Authorization"] = f"Bearer {REACHER_API_KEY}"

        try:
            async with httpx.AsyncClient(timeout=SMTP_TIMEOUT + 5) as client:
                resp = await client.post(
                    url,
                    json={"to_email": email},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

        except httpx.ConnectError:
            logger.debug(f"🔍 Reacher not reachable at {REACHER_URL}")
            return None
        except httpx.TimeoutException:
            logger.debug(f"🔍 Reacher timed out for {email}")
            return None
        except Exception as e:
            logger.debug(f"🔍 Reacher error for {email}: {e}")
            return None

        # Parse Reacher response
        # Reacher returns: {"is_reachable": "safe"/"risky"/"invalid"/"unknown"}
        is_reachable = data.get("is_reachable", "unknown")
        misc = data.get("misc", {}) or {}
        smtp = data.get("smtp", {}) or {}
        is_catch_all = misc.get("is_catch_all", False) or smtp.get("is_catch_all", False)

        if is_reachable == "safe":
            return self._result("valid", 0.95, "Reacher: SMTP handshake successful")
        elif is_reachable == "risky" or is_catch_all:
            return self._result(
                "risky", 0.55,
                "Reacher: catch-all domain or risky — email may still be valid"
            )
        elif is_reachable == "invalid":
            return self._result("invalid", 0.05, "Reacher: SMTP rejected this address")
        else:
            return self._result("unknown", 0.4, f"Reacher: {is_reachable}")

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _result(self, status: str, confidence: float, reason: str) -> dict:
        return {
            "status": status,       # valid / invalid / risky / unknown
            "confidence": confidence,
            "reason": reason,
        }


# Singleton
verifier = EmailVerifier()