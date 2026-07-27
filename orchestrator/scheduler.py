"""
SalesNeuron — Scheduler
=========================
Recurring maintenance operations that span ALL companies at once,
not tied to any single pipeline run:

  - check_replies()    → poll Gmail for new replies (run hourly)
  - process_followups() → send due follow-ups (run once daily)

These are intentionally NOT LangGraph nodes (see graph.py docstring,
point 3) — they don't belong to one company's session, they belong to
the whole SQLite sequences table.

Usage (manual):
    python run_pipeline.py --daily-maintenance --sender-email you@gmail.com

Usage (cron / Task Scheduler):
    Windows Task Scheduler → run this on a timer:
      python run_pipeline.py --check-replies --sender-email you@gmail.com   (hourly)
      python run_pipeline.py --follow-ups --sender-email you@gmail.com      (daily)
"""

import logging
from agents.sequence_manager import SequenceManager

logger = logging.getLogger(__name__)


async def check_replies(sender_email: str) -> int:
    """Poll Gmail for new replies to any active sequence. Returns count found."""
    sm = SequenceManager()
    await sm.init()
    count = await sm.check_replies(sender_email)
    logger.info(f"📬 Reply check complete — {count} new reply(s)")
    return count


async def process_followups(sender_email: str, sender_name: str = "") -> int:
    """Send any follow-ups that are due today. Returns count sent."""
    sm = SequenceManager()
    await sm.init()
    count = await sm.process_followups(sender_email, sender_name)
    logger.info(f"📬 Follow-ups processed — {count} sent")
    return count


async def daily_maintenance(sender_email: str, sender_name: str = "") -> dict:
    """Convenience wrapper — runs both maintenance tasks in sequence."""
    replies = await check_replies(sender_email)
    followups = await process_followups(sender_email, sender_name)
    return {"replies_found": replies, "followups_sent": followups}