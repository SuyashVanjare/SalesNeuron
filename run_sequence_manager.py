"""
SalesNeuron — Sequence Manager CLI
=====================================
Send cold emails, process follow-ups, check replies, and view stats.

Commands:
  --send          Send a cold email (from JSON file or latest generated)
  --follow-ups    Process all pending follow-ups due today
  --check-replies Poll Gmail for replies to sent sequences
  --stats         Show pipeline statistics
  --list          List all sequences (optionally filter by status)
  --find-email    Find email for a person (test EmailFinder pipeline)

Examples:
  # Send a cold email from a specific file
  python run_sequence_manager.py --send --email-file data/emails/stripe_20260724.json \
      --sender-email you@gmail.com --sender-name "Suyash"

  # Send and provide recipient email directly (skip EmailFinder)
  python run_sequence_manager.py --send --email-file data/emails/stripe_20260724.json \
      --sender-email you@gmail.com --to ceo@stripe.com

  # Dry run (resolve email, preview, don't send)
  python run_sequence_manager.py --send --email-file data/emails/stripe_20260724.json \
      --sender-email you@gmail.com --dry-run

  # Process follow-ups (run daily)
  python run_sequence_manager.py --follow-ups --sender-email you@gmail.com

  # Check for replies from contacts
  python run_sequence_manager.py --check-replies --sender-email you@gmail.com

  # View stats
  python run_sequence_manager.py --stats

  # List sequences
  python run_sequence_manager.py --list
  python run_sequence_manager.py --list --status sent

  # Test EmailFinder for a person
  python run_sequence_manager.py --find-email --name "Harshil Mathur" \
      --website https://razorpay.com --title "CEO"

Gmail Setup (one-time):
  1. https://console.cloud.google.com → Enable Gmail API
  2. Create OAuth2 credentials (Desktop app) → Download JSON
  3. Save as data/gmail_credentials.json
  4. Run any --send command → browser will open for authorization
  5. Token auto-saved to data/gmail_token.json
"""

import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def usage():
    console.print(Panel(
        __doc__,
        title="[bold cyan]SalesNeuron — Sequence Manager[/bold cyan]",
        border_style="cyan",
    ))


def parse_args() -> dict:
    args = sys.argv[1:]
    parsed = {
        "send": "--send" in args,
        "follow_ups": "--follow-ups" in args,
        "check_replies": "--check-replies" in args,
        "stats": "--stats" in args,
        "list": "--list" in args,
        "find_email": "--find-email" in args,
        "dry_run": "--dry-run" in args,
        "email_file": None,
        "sender_email": None,
        "sender_name": "",
        "to": None,
        "status": None,
        "name": None,
        "website": None,
        "title": None,
        "company": None,
    }

    for flag, key in [
        ("--email-file", "email_file"),
        ("--sender-email", "sender_email"),
        ("--sender-name", "sender_name"),
        ("--to", "to"),
        ("--status", "status"),
        ("--name", "name"),
        ("--website", "website"),
        ("--title", "title"),
        ("--company", "company"),
    ]:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                parsed[key] = args[idx + 1]

    return parsed


async def main():
    args = parse_args()

    if not any([
        args["send"], args["follow_ups"], args["check_replies"],
        args["stats"], args["list"], args["find_email"],
        "--auth" in sys.argv,
    ]):
        usage()
        return

    # ── --auth (standalone Gmail authorization) ───────────────────
    if "--auth" in sys.argv:
        console.print("\n[bold green]SalesNeuron[/bold green] — Gmail Authorization\n")
        from agents.sequence_manager import SequenceManager
        sm = SequenceManager()
        await sm.init()
        gmail = sm._get_gmail_service()
        if gmail:
            console.print("[bold green]✅ Gmail authorized and ready![/bold green]")
            console.print(f"[dim]Token saved → data/gmail_token.json[/dim]\n")
        else:
            console.print("[red]❌ Authorization failed — check the steps above[/red]")
        return

    # ── --find-email ──────────────────────────────────────────────
    if args["find_email"]:
        if not args["name"]:
            console.print("[red]--find-email requires --name[/red]")
            return
        await cmd_find_email(args)
        return

    # ── --stats ───────────────────────────────────────────────────
    if args["stats"]:
        await cmd_stats()
        return

    # ── --list ────────────────────────────────────────────────────
    if args["list"]:
        await cmd_list(args.get("status"))
        return

    # ── --check-replies ───────────────────────────────────────────
    if args["check_replies"]:
        if not args["sender_email"]:
            console.print("[red]--check-replies requires --sender-email[/red]")
            return
        await cmd_check_replies(args["sender_email"])
        return

    # ── --follow-ups ──────────────────────────────────────────────
    if args["follow_ups"]:
        if not args["sender_email"]:
            console.print("[red]--follow-ups requires --sender-email[/red]")
            return
        await cmd_follow_ups(args["sender_email"], args["sender_name"])
        return

    # ── --send ────────────────────────────────────────────────────
    if args["send"]:
        if not args["sender_email"]:
            console.print("[red]--send requires --sender-email[/red]")
            return
        await cmd_send(args)
        return


# ──────────────────────────────────────────────────────────────────
# Command implementations
# ──────────────────────────────────────────────────────────────────

async def cmd_send(args: dict):
    """Send a cold email from a file or find latest in data/emails/."""
    from agents.sequence_manager import sequence_manager
    from core.email_models import ColdEmail

    email_file = args["email_file"]

    # If no file given, pick the most recent in data/emails/
    if not email_file:
        email_dir = Path("data/emails")
        json_files = sorted(email_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if not json_files:
            console.print("[red]No email files found in data/emails/ — run the personalizer first[/red]")
            console.print("  python run_personalizer.py --company stripe.com --sender 'Suyash' --role 'Founder'")
            return
        email_file = str(json_files[-1])
        console.print(f"[dim]Using latest email file: {email_file}[/dim]")

    if not Path(email_file).exists():
        console.print(f"[red]File not found: {email_file}[/red]")
        return

    # Load email
    data = json.loads(Path(email_file).read_text())
    cold_email = ColdEmail(**data)

    # Preview before sending
    console.print(Panel(
        f"[bold]To:[/bold]       {cold_email.contact_name or '?'} @ {cold_email.company_name}\n"
        f"[bold]Website:[/bold]  {cold_email.website}\n"
        f"[bold]Subject:[/bold]  {cold_email.subject}\n"
        f"[bold]Signal:[/bold]   {cold_email.buying_signal_used[:80] if cold_email.buying_signal_used else 'N/A'}\n\n"
        f"[dim]{cold_email.body[:300]}{'...' if len(cold_email.body) > 300 else ''}[/dim]",
        title="📧 Email Preview",
        border_style="cyan",
    ))

    if args["dry_run"]:
        console.print("\n[yellow]🔍 DRY RUN — will resolve email but not send[/yellow]\n")

    await sequence_manager.init()
    result = await sequence_manager.send(
        cold_email=cold_email,
        sender_email=args["sender_email"],
        sender_name=args["sender_name"] or "",
        recipient_email=args.get("to"),
    )

    _print_send_result(result, args["dry_run"])


async def cmd_follow_ups(sender_email: str, sender_name: str = ""):
    """Process all follow-ups due today."""
    from agents.sequence_manager import sequence_manager

    console.print("\n[bold cyan]📬 Processing follow-ups...[/bold cyan]")
    await sequence_manager.init()
    count = await sequence_manager.process_followups(
        sender_email=sender_email,
        sender_name=sender_name,
    )
    if count == 0:
        console.print("[dim]No follow-ups due today.[/dim]")
    else:
        console.print(f"\n[green]✅ {count} follow-up(s) sent[/green]")


async def cmd_check_replies(sender_email: str):
    """Check Gmail for replies."""
    from agents.sequence_manager import sequence_manager

    console.print("\n[bold cyan]📬 Checking for replies...[/bold cyan]")
    await sequence_manager.init()
    count = await sequence_manager.check_replies(sender_email=sender_email)
    if count == 0:
        console.print("[dim]No new replies found.[/dim]")
    else:
        console.print(f"\n[green]🎉 {count} new reply(s) detected![/green]")


async def cmd_stats():
    """Show pipeline statistics."""
    from agents.sequence_manager import sequence_manager

    await sequence_manager.init()
    stats = await sequence_manager.stats()

    table = Table(title="📊 Sequence Manager Stats", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total sequences", str(stats["total_sequences"]))
    table.add_row("Sent", str(stats["sent"]))
    table.add_row("Replied", str(stats["replied"]))
    table.add_row("Reply rate", str(stats.get("reply_rate", "0%")))
    table.add_row("Pending follow-ups", str(stats.get("pending_followups", 0)))

    console.print()
    console.print(table)

    if stats.get("by_status"):
        console.print("\n[bold]By status:[/bold]")
        for status, count in stats["by_status"].items():
            console.print(f"  {status}: {count}")
    console.print()


async def cmd_list(status: str = None):
    """List all sequences."""
    from agents.sequence_manager import sequence_manager

    await sequence_manager.init()
    sequences = await sequence_manager.stats()

    # Re-query to get actual rows
    import aiosqlite
    import os
    db_path = Path(os.getenv("DB_PATH", "data/salesneuron.db"))

    where = "WHERE status=?" if status else ""
    params = [status] if status else []

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"""SELECT company_name, contact_name, contact_email,
                       subject, status, sent_at, replied_at
               FROM sequences {where}
               ORDER BY created_at DESC LIMIT 50""",
            params,
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    if not rows:
        console.print(f"[dim]No sequences found{' with status=' + status if status else ''}.[/dim]")
        return

    table = Table(
        title=f"📬 Sequences{' [' + status + ']' if status else ''}",
        border_style="cyan",
    )
    table.add_column("Company", style="bold")
    table.add_column("Contact")
    table.add_column("Email")
    table.add_column("Status")
    table.add_column("Sent")
    table.add_column("Replied")

    status_colors = {
        "sent": "yellow",
        "replied": "green",
        "bounced": "red",
        "no_email": "dim",
        "pending": "cyan",
        "send_failed": "red",
    }

    for row in rows:
        s = row["status"]
        color = status_colors.get(s, "white")
        table.add_row(
            row["company_name"] or "?",
            row["contact_name"] or "?",
            row["contact_email"] or "[dim]not found[/dim]",
            f"[{color}]{s}[/{color}]",
            (row["sent_at"] or "")[:10],
            (row["replied_at"] or "")[:10],
        )

    console.print()
    console.print(table)
    console.print()


async def cmd_find_email(args: dict):
    """Test EmailFinder for a person."""
    from core.email_finder import email_finder

    console.print(
        f"\n[bold cyan]📧 Finding email for: "
        f"{args['name']} @ {args.get('website') or args.get('company', '?')}[/bold cyan]\n"
    )

    await email_finder.init()
    result = await email_finder.find(
        name=args["name"],
        company=args.get("company") or "",
        website=args.get("website") or "",
        title=args.get("title") or "",
    )

    email = result.get("email")
    status = "[green]✅ FOUND[/green]" if email else "[red]❌ NOT FOUND[/red]"

    console.print(Panel(
        f"{status}\n\n"
        f"[bold]Email:[/bold]        {email or 'N/A'}\n"
        f"[bold]Confidence:[/bold]   {result.get('confidence', 0):.0%}\n"
        f"[bold]Source:[/bold]       {result.get('source', 'N/A')}\n"
        f"[bold]Verification:[/bold] {result.get('verification_status', 'N/A')}\n"
        f"[bold]Pattern:[/bold]      {result.get('pattern_used', 'N/A')}\n"
        f"[bold]Provider:[/bold]     {result.get('provider_used', 'N/A')}",
        title="🔍 Email Finder Result",
        border_style="green" if email else "red",
        padding=(1, 2),
    ))
    console.print()


def _print_send_result(result: dict, dry_run: bool = False):
    status = result.get("status")
    success = status in ("sent",)
    dry = status == "no_email" and dry_run

    if dry_run:
        color = "yellow"
        icon = "🔍"
    elif success:
        color = "green"
        icon = "✅"
    else:
        color = "red"
        icon = "❌"

    content = (
        f"[bold]Status:[/bold]       {status}\n"
        f"[bold]Sequence ID:[/bold]  {result.get('sequence_id')}\n"
        f"[bold]To:[/bold]           {result.get('contact_email') or 'N/A'}\n"
        f"[bold]Message ID:[/bold]   {result.get('gmail_message_id') or 'N/A'}\n"
    )
    if result.get("next_followup"):
        content += f"[bold]Next follow-up:[/bold] {result['next_followup'][:10]}\n"
    if result.get("error"):
        content += f"\n[red]Error: {result['error']}[/red]"

    console.print()
    console.print(Panel(
        content,
        title=f"{icon} {'Dry Run — Not Sent' if dry_run else 'Send Result'}",
        border_style=color,
        padding=(1, 2),
    ))
    console.print()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        handlers=[logging.StreamHandler()],
    )
    asyncio.run(main())