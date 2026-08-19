"""
SalesNeuron — Sequence Manager
================================
Sends cold emails via Gmail API, tracks replies, and auto-sends
follow-ups if no reply after Day 3, Day 7, and Day 14.

Pipeline:
  1. ENRICH    — find recipient email via EmailFinder (full 10-step pipeline)
  2. SEND      — send via Gmail API (OAuth2 — your own Gmail account)
  3. TRACK     — poll Gmail inbox for replies to each sent email
  4. FOLLOW-UP — auto-send follow-up if no reply after configured days
  5. FEEDBACK  — record delivered/bounced/replied events for pattern learning

Gmail OAuth Setup (one-time):
  1. Go to Google Cloud Console → APIs & Services → Credentials
  2. Create OAuth 2.0 Client ID (Desktop app)
  3. Download credentials JSON → save as data/gmail_credentials.json
  4. First run will open browser for authorization
  5. Token saved to data/gmail_token.json (auto-refreshes)

Usage:
    from agents.sequence_manager import sequence_manager

    await sequence_manager.init()

    # Enrich + send one email
    result = await sequence_manager.send(
        email_file="data/emails/stripe_20260724.json",
        sender_email="you@gmail.com",
    )

    # Process all pending follow-ups
    await sequence_manager.process_followups(sender_email="you@gmail.com")

    # Check for new replies
    await sequence_manager.check_replies(sender_email="you@gmail.com")
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import aiosqlite

from core.email_finder import email_finder
from core.email_models import ColdEmail
from core.knowledge_base import kb

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))
GMAIL_CREDS_PATH = Path(os.getenv("GMAIL_CREDS_PATH", "data/gmail_credentials.json"))
GMAIL_TOKEN_PATH = Path(os.getenv("GMAIL_TOKEN_PATH", "data/gmail_token.json"))

# Follow-up schedule in days after the initial send
FOLLOWUP_DAYS = [
    int(d) for d in os.getenv("FOLLOWUP_DAYS", "3,7,14").split(",")
]

# Gmail API scopes
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Follow-up email templates (LLM will improve these if personalizer is available)
_FOLLOWUP_TEMPLATES = {
    1: (
        "Hi {first_name},\n\n"
        "Just wanted to bump this up in case it got buried.\n\n"
        "{original_cta}\n\n"
        "Best,\n{sender_name}"
    ),
    2: (
        "Hi {first_name},\n\n"
        "One last try — happy to share a quick 2-min demo if easier.\n\n"
        "{original_cta}\n\n"
        "Best,\n{sender_name}"
    ),
    3: (
        "Hi {first_name},\n\n"
        "I'll leave you alone after this, I promise.\n"
        "If timing is off, I'm happy to reconnect next quarter.\n\n"
        "Best,\n{sender_name}"
    ),
}


class SequenceManager:
    """
    End-to-end email sequence: find email → send → track → follow-up.
    All state stored in SQLite for crash recovery.
    """

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False
        self._gmail = None  # initialized lazily

    async def init(self):
        """Initialize DB tables and email finder."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS sequences (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name        TEXT NOT NULL,
                    website             TEXT,
                    contact_name        TEXT,
                    contact_email       TEXT,
                    subject             TEXT NOT NULL,
                    body                TEXT NOT NULL,
                    buying_signal_used  TEXT,
                    knowledge_chunks_used TEXT,
                    sender_email        TEXT,
                    status              TEXT DEFAULT 'pending',
                    gmail_message_id    TEXT,
                    gmail_thread_id     TEXT,
                    sent_at             TEXT,
                    replied_at          TEXT,
                    bounced_at          TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS followups (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id     INTEGER NOT NULL,
                    followup_number INTEGER NOT NULL,
                    scheduled_for   TEXT NOT NULL,
                    sent_at         TEXT,
                    status          TEXT DEFAULT 'pending',
                    gmail_message_id TEXT,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (sequence_id) REFERENCES sequences(id)
                );

                CREATE INDEX IF NOT EXISTS idx_sequences_status
                    ON sequences(status);
                CREATE INDEX IF NOT EXISTS idx_sequences_website
                    ON sequences(website);
                CREATE INDEX IF NOT EXISTS idx_followups_scheduled
                    ON followups(scheduled_for, status);
                CREATE INDEX IF NOT EXISTS idx_followups_sequence
                    ON followups(sequence_id);
            """)
            await db.commit()

            # Migration: existing DBs created before this column existed
            # won't get it from CREATE TABLE IF NOT EXISTS above (that
            # only applies to brand-new tables). Add it if missing.
            try:
                await db.execute(
                    "ALTER TABLE sequences ADD COLUMN knowledge_chunks_used TEXT"
                )
                await db.commit()
            except Exception:
                pass  # column already exists — expected on repeat runs

        await email_finder.init()
        self._ready = True
        logger.info(f"📬 SequenceManager initialized → {self._db_path}")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def send(
        self,
        cold_email: ColdEmail,
        sender_email: str,
        sender_name: str = "",
        recipient_email: Optional[str] = None,
        find_email_if_missing: bool = True,
        scraper=None,
    ) -> dict:
        """
        Send a cold email. If recipient_email is not provided, runs the
        full 10-step email finder pipeline to discover it automatically.

        Args:
            cold_email:            ColdEmail from PersonalizerAgent
            sender_email:          Your Gmail address
            sender_name:           Your name (for follow-up templates)
            recipient_email:       Override — skip email finder if known
            find_email_if_missing: Run EmailFinder if email unknown
            scraper:               BrowserScraper for pattern discovery

        Returns dict with: sequence_id, status, contact_email, message_id
        """
        self._ensure_ready()
        now = datetime.now().isoformat()

        # ── Step 1: Find recipient email ──────────────────────────
        contact_email = recipient_email
        person_id = None

        if not contact_email and find_email_if_missing:
            logger.info(f"📬 Finding email for {cold_email.contact_name} @ {cold_email.company_name}")
            find_result = await email_finder.find(
                name=cold_email.contact_name or cold_email.company_name,
                company=cold_email.company_name,
                website=cold_email.website,
                scraper=scraper,
            )
            contact_email = find_result.get("email")
            person_id = find_result.get("person_id")

            if not contact_email:
                logger.warning(
                    f"📬 Could not find email for {cold_email.contact_name} "
                    f"@ {cold_email.company_name} — sequence saved as 'no_email'"
                )
                seq_id = await self._save_sequence(
                    cold_email=cold_email,
                    sender_email=sender_email,
                    contact_email=None,
                    status="no_email",
                    now=now,
                )
                return {
                    "sequence_id": seq_id,
                    "status": "no_email",
                    "contact_email": None,
                    "message_id": None,
                }

        # ── Step 2: Send via Gmail ────────────────────────────────
        logger.info(
            f"📬 Sending to {contact_email} — "
            f"{cold_email.company_name}: {cold_email.subject}"
        )

        try:
            message_id, thread_id = await self._gmail_send(
                sender_email=sender_email,
                to_email=contact_email,
                subject=cold_email.subject,
                body=cold_email.body,
            )
        except Exception as e:
            logger.error(f"📬 Gmail send failed: {e}")
            seq_id = await self._save_sequence(
                cold_email=cold_email,
                sender_email=sender_email,
                contact_email=contact_email,
                status="send_failed",
                now=now,
            )
            return {
                "sequence_id": seq_id,
                "status": "send_failed",
                "contact_email": contact_email,
                "message_id": None,
                "error": str(e),
            }

        # ── Step 3: Save sequence + schedule follow-ups ───────────
        sent_at = datetime.now().isoformat()
        seq_id = await self._save_sequence(
            cold_email=cold_email,
            sender_email=sender_email,
            contact_email=contact_email,
            status="sent",
            now=now,
            sent_at=sent_at,
            gmail_message_id=message_id,
            gmail_thread_id=thread_id,
        )
        await self._schedule_followups(seq_id, sent_at, cold_email, sender_name)

        # Record feedback event
        if person_id:
            await email_finder.record_feedback(
                email=contact_email,
                event_type="delivered",
                person_id=person_id,
            )

        # Record which knowledge chunks were used, so a future reply
        # on this sequence can credit them (see check_replies() below).
        if cold_email.knowledge_chunks_used:
            try:
                await kb.record_feedback(cold_email.knowledge_chunks_used, "sent")
            except Exception as e:
                logger.debug(f"📬 Knowledge feedback recording failed (non-fatal): {e}")

        logger.info(
            f"📬 ✅ Sent to {contact_email} | sequence_id={seq_id} | "
            f"message_id={message_id}"
        )
        return {
            "sequence_id": seq_id,
            "status": "sent",
            "contact_email": contact_email,
            "message_id": message_id,
        }

    async def send_from_file(
        self,
        email_file: str,
        sender_email: str,
        sender_name: str = "",
        recipient_email: Optional[str] = None,
        scraper=None,
    ) -> dict:
        """Load a ColdEmail from a JSON file and send it."""
        path = Path(email_file)
        if not path.exists():
            raise FileNotFoundError(f"Email file not found: {email_file}")
        data = json.loads(path.read_text())
        cold_email = ColdEmail(**data)
        return await self.send(
            cold_email=cold_email,
            sender_email=sender_email,
            sender_name=sender_name,
            recipient_email=recipient_email,
            scraper=scraper,
        )

    async def process_followups(self, sender_email: str, sender_name: str = "") -> int:
        """
        Check for all pending follow-ups that are due today or earlier.
        Send them if the original sequence hasn't received a reply.
        Returns number of follow-ups sent.
        """
        self._ensure_ready()
        now_str = datetime.now().isoformat()
        sent_count = 0

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT f.*, s.contact_email, s.contact_name, s.subject,
                       s.body, s.gmail_thread_id, s.company_name,
                       s.buying_signal_used, s.status as seq_status
                FROM followups f
                JOIN sequences s ON f.sequence_id = s.id
                WHERE f.status = 'pending'
                  AND f.scheduled_for <= ?
                  AND s.status = 'sent'
                ORDER BY f.scheduled_for ASC
                """,
                (now_str,),
            ) as cur:
                due = await cur.fetchall()

        logger.info(f"📬 {len(due)} follow-up(s) due")

        for row in due:
            seq_id = row["sequence_id"]
            followup_num = row["followup_number"]
            contact_email = row["contact_email"]
            first_name = (row["contact_name"] or "there").split()[0]

            # Build follow-up email
            template = _FOLLOWUP_TEMPLATES.get(followup_num, _FOLLOWUP_TEMPLATES[3])
            body = template.format(
                first_name=first_name,
                original_cta="Would you be open to a 15-minute call?",
                sender_name=sender_name or "Team SalesNeuron",
            )
            subject = f"Re: {row['subject']}"

            try:
                message_id, _ = await self._gmail_send(
                    sender_email=sender_email,
                    to_email=contact_email,
                    subject=subject,
                    body=body,
                    thread_id=row["gmail_thread_id"],
                )
                # Mark follow-up as sent
                async with aiosqlite.connect(self._db_path) as db:
                    await db.execute(
                        """UPDATE followups SET status='sent', sent_at=?, gmail_message_id=?
                           WHERE id=?""",
                        (datetime.now().isoformat(), message_id, row["id"]),
                    )
                    await db.commit()

                sent_count += 1
                logger.info(
                    f"📬 Follow-up {followup_num} sent to {contact_email} "
                    f"(seq={seq_id})"
                )
            except Exception as e:
                logger.error(f"📬 Follow-up send failed for {contact_email}: {e}")

        return sent_count

    async def _is_auto_reply(self, gmail, message_id: str) -> bool:
        """
        Detect whether a Gmail message is an automated response rather
        than a genuine human reply. Checks two independent signals:

        1. Headers — RFC 3834 defines 'Auto-Submitted' for exactly this
           purpose; many systems also send 'X-Autoreply' or
           'X-Autorespond'. Any value other than 'no' means automated.
        2. Subject/snippet phrasing — catches auto-responders that don't
           set proper headers (surprisingly common: some ticketing
           systems and basic out-of-office setups skip them entirely).

        Returns True if this looks automated — caller should NOT count
        it as a real reply (no reply-rate credit, no RAG feedback boost,
        follow-ups stay scheduled since a human hasn't actually responded).
        """
        try:
            msg = gmail.users().messages().get(
                userId="me", id=message_id, format="metadata",
                metadataHeaders=["Auto-Submitted", "X-Autoreply", "X-Autorespond", "Subject"],
            ).execute()

            headers = {
                h["name"].lower(): h["value"]
                for h in msg.get("payload", {}).get("headers", [])
            }

            auto_submitted = headers.get("auto-submitted", "no").lower()
            if auto_submitted not in ("no", ""):
                return True
            if headers.get("x-autoreply") or headers.get("x-autorespond"):
                return True

            subject = headers.get("subject", "").lower()
            snippet = (msg.get("snippet", "") or "").lower()
            auto_phrases = [
                "out of office", "automatic reply", "auto-reply",
                "away from my desk", "on leave", "on vacation",
                "ticket has been created", "we have received your",
                "this is an automated", "do not reply to this email",
                "undeliverable", "delivery status notification",
                "auto acknowledgement", "auto acknowledgment",
            ]
            combined = f"{subject} {snippet}"
            return any(phrase in combined for phrase in auto_phrases)

        except Exception as e:
            logger.debug(f"📬 Auto-reply check failed for {message_id}: {e}")
            return False  # Fail open — better to over-count than miss a real reply

    async def check_replies(self, sender_email: str) -> int:
        """
        Poll Gmail inbox for replies to sent sequences.
        Updates sequence status to 'replied' and cancels pending follow-ups.
        Returns count of new replies found.
        """
        self._ensure_ready()

        # Get all sent sequences with Gmail thread IDs
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, gmail_thread_id, contact_email, knowledge_chunks_used
                   FROM sequences
                   WHERE status = 'sent' AND gmail_thread_id IS NOT NULL""",
            ) as cur:
                sequences = await cur.fetchall()

        if not sequences:
            return 0

        gmail = self._get_gmail_service()
        if not gmail:
            logger.warning("📬 Gmail not configured — cannot check replies")
            return 0

        reply_count = 0
        for seq in sequences:
            try:
                # Get the thread to check for replies
                thread = gmail.users().threads().get(
                    userId="me", id=seq["gmail_thread_id"]
                ).execute()
                messages = thread.get("messages", [])

                # If thread has more than 1 message, someone (or something)
                # replied. Before crediting it as a real reply, filter out
                # auto-responders — out-of-office, ticket-system acks, and
                # mail-loop bounces all add a message to the thread too,
                # and were previously counted identically to a human reply.
                if len(messages) > 1:
                    latest_msg = messages[-1]
                    if await self._is_auto_reply(gmail, latest_msg["id"]):
                        logger.info(
                            f"📬 Skipped auto-reply from {seq['contact_email']} "
                            f"(seq={seq['id']}) — not counted as engagement"
                        )
                        continue

                    reply_at = datetime.now().isoformat()
                    async with aiosqlite.connect(self._db_path) as db:
                        await db.execute(
                            """UPDATE sequences SET status='replied', replied_at=?,
                               updated_at=? WHERE id=?""",
                            (reply_at, reply_at, seq["id"]),
                        )
                        # Cancel pending follow-ups for this sequence
                        await db.execute(
                            """UPDATE followups SET status='cancelled'
                               WHERE sequence_id=? AND status='pending'""",
                            (seq["id"],),
                        )
                        await db.commit()

                    # Record feedback for pattern learning
                    await email_finder.record_feedback(
                        email=seq["contact_email"],
                        event_type="replied",
                    )

                    # Credit whichever knowledge chunks were used in this
                    # email — this is what lets search() start preferring
                    # chunks with a proven reply track record over merely
                    # semantically-similar ones. Without this, the "sent"
                    # feedback from send() was being recorded but nothing
                    # ever completed the loop by recording "replied".
                    try:
                        chunks = json.loads(seq["knowledge_chunks_used"] or "[]")
                        if chunks:
                            await kb.record_feedback(chunks, "replied")
                    except Exception as e:
                        logger.debug(f"📬 Knowledge feedback on reply failed (non-fatal): {e}")

                    reply_count += 1
                    logger.info(
                        f"📬 Reply detected from {seq['contact_email']} "
                        f"(seq={seq['id']})"
                    )
            except Exception as e:
                logger.debug(f"📬 Thread check failed for seq {seq['id']}: {e}")

        logger.info(f"📬 Reply check complete — {reply_count} new reply(s)")
        return reply_count

    async def stats(self) -> dict:
        """Return sequence statistics."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """SELECT status, COUNT(*) as count
                   FROM sequences GROUP BY status"""
            ) as cur:
                rows = await cur.fetchall()
            async with db.execute(
                "SELECT COUNT(*) FROM followups WHERE status='pending'"
            ) as cur:
                pending_followups = (await cur.fetchone())[0]

        status_counts = {row[0]: row[1] for row in rows}
        total = sum(status_counts.values())
        sent = status_counts.get("sent", 0) + status_counts.get("replied", 0)
        replied = status_counts.get("replied", 0)

        return {
            "total_sequences": total,
            "sent": sent,
            "replied": replied,
            "reply_rate": f"{(replied/sent*100):.1f}%" if sent else "0%",
            "pending_followups": pending_followups,
            "by_status": status_counts,
        }

    # ──────────────────────────────────────────────────────────────
    # Gmail integration
    # ──────────────────────────────────────────────────────────────

    def _get_gmail_service(self):
        """
        Initialize Gmail API service using OAuth2.
        On first run, opens a browser window for authorization.
        Token is saved to data/gmail_token.json and auto-refreshed.
        """
        if self._gmail:
            return self._gmail

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            logger.error(
                "📬 Gmail dependencies missing. Install with:\n"
                "  pip install google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib"
            )
            return None

        if not GMAIL_CREDS_PATH.exists():
            logger.error(
                f"📬 Gmail credentials not found at {GMAIL_CREDS_PATH}\n"
                "  1. Go to https://console.cloud.google.com\n"
                "  2. APIs & Services → Credentials → Create OAuth 2.0 Client ID\n"
                "  3. Application type: Desktop app\n"
                "  4. Download JSON → save as data/gmail_credentials.json"
            )
            return None

        creds = None
        if GMAIL_TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(
                str(GMAIL_TOKEN_PATH), GMAIL_SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(GMAIL_CREDS_PATH), GMAIL_SCOPES
                )
                creds = flow.run_local_server(port=0)
            GMAIL_TOKEN_PATH.write_text(creds.to_json())

        self._gmail = build("gmail", "v1", credentials=creds)
        return self._gmail

    async def _gmail_send(
        self,
        sender_email: str,
        to_email: str,
        subject: str,
        body: str,
        thread_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Send an email via Gmail API.
        Returns (message_id, thread_id).
        """
        gmail = self._get_gmail_service()
        if not gmail:
            raise RuntimeError(
                "Gmail not configured. See setup instructions above."
            )

        # Build MIME message
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        if thread_id:
            msg["In-Reply-To"] = thread_id
            msg["References"] = thread_id

        msg.attach(MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        body_payload = {"raw": raw}
        if thread_id:
            body_payload["threadId"] = thread_id

        # Run in executor (Gmail SDK is sync)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: gmail.users().messages().send(
                userId="me", body=body_payload
            ).execute()
        )

        return result.get("id", ""), result.get("threadId", "")

    # ──────────────────────────────────────────────────────────────
    # DB helpers
    # ──────────────────────────────────────────────────────────────

    async def _save_sequence(
        self,
        cold_email: ColdEmail,
        sender_email: str,
        contact_email: Optional[str],
        status: str,
        now: str,
        sent_at: Optional[str] = None,
        gmail_message_id: Optional[str] = None,
        gmail_thread_id: Optional[str] = None,
    ) -> int:
        chunks_json = json.dumps(cold_email.knowledge_chunks_used or [])
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO sequences
                    (company_name, website, contact_name, contact_email,
                     subject, body, buying_signal_used, knowledge_chunks_used,
                     sender_email, status, gmail_message_id, gmail_thread_id,
                     sent_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cold_email.company_name,
                    cold_email.website,
                    cold_email.contact_name,
                    contact_email,
                    cold_email.subject,
                    cold_email.body,
                    cold_email.buying_signal_used,
                    chunks_json,
                    sender_email,
                    status,
                    gmail_message_id,
                    gmail_thread_id,
                    sent_at,
                    now,
                    now,
                ),
            )
            seq_id = cursor.lastrowid
            await db.commit()
        return seq_id

    async def _schedule_followups(
        self,
        seq_id: int,
        sent_at: str,
        cold_email: ColdEmail,
        sender_name: str,
    ):
        """Create follow-up rows for Day 3, 7, 14 after send."""
        sent_dt = datetime.fromisoformat(sent_at)
        now = datetime.now().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            for i, days in enumerate(FOLLOWUP_DAYS, 1):
                scheduled = (sent_dt + timedelta(days=days)).isoformat()
                await db.execute(
                    """
                    INSERT INTO followups
                        (sequence_id, followup_number, scheduled_for,
                         status, created_at)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (seq_id, i, scheduled, now),
                )
            await db.commit()
        logger.info(
            f"📬 Scheduled {len(FOLLOWUP_DAYS)} follow-ups for seq {seq_id} "
            f"on days {FOLLOWUP_DAYS}"
        )

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "SequenceManager not initialized. "
                "Call `await sequence_manager.init()` first."
            )


# Singleton
sequence_manager = SequenceManager()