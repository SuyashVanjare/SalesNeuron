"""
SalesNeuron — Pipeline CLI (LangGraph Orchestrator)
======================================================
The single command that runs the entire autonomous sales pipeline:

    research → personalize → find email → approve → send

Usage:
    # Full run, pauses for your approval before sending (default, safest)
    python run_pipeline.py https://razorpay.com --sender "Suyash" --role "Founder, SalesNeuron" --sender-email you@gmail.com

    # Fully autonomous — auto-sends if email confidence is high enough
    python run_pipeline.py https://razorpay.com --sender "Suyash" --sender-email you@gmail.com --auto-send

    # Force fresh research (ignore cache)
    python run_pipeline.py https://razorpay.com --sender-email you@gmail.com --refresh

    # Daily maintenance — check replies + send due follow-ups
    python run_pipeline.py --daily-maintenance --sender-email you@gmail.com
    python run_pipeline.py --check-replies --sender-email you@gmail.com
    python run_pipeline.py --follow-ups --sender-email you@gmail.com
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ["playwright", "httpx", "httpcore", "urllib3",
              "chromadb", "sentence_transformers", "transformers",
              "google_genai.models", "langgraph"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Confirm
from rich import box

from orchestrator.graph import run_pipeline, resume_and_send
from orchestrator.scheduler import check_replies, process_followups, daily_maintenance

console = Console()


def print_state_summary(state: dict) -> None:
    """Pretty print the pipeline's final (or paused) state."""
    console.print()
    status = state.get("status", "unknown")

    status_style = {
        "sent": ("✅ SENT", "green"),
        "awaiting_approval": ("⏸️  AWAITING APPROVAL", "yellow"),
        "skipped_no_email": ("📭 NO EMAIL FOUND", "yellow"),
        "failed": ("❌ FAILED", "red"),
    }.get(status, (f"● {status.upper()}", "white"))

    label, color = status_style
    console.print(Rule(f"[bold {color}]{label}[/bold {color}]"))

    profile = state.get("profile") or {}
    cold_email = state.get("cold_email") or {}
    found_email = state.get("found_email") or {}

    if profile:
        console.print(
            f"\n[bold]{profile.get('company_name', '—')}[/bold] "
            f"[dim]({profile.get('research_confidence', '—').upper()} confidence, "
            f"{len(profile.get('buying_signals', []))} signals)[/dim]"
        )

    if cold_email:
        console.print(f"\n[bold]Subject:[/bold] {cold_email.get('subject', '—')}")
        console.print(Panel(
            cold_email.get("body", ""),
            border_style="dim",
            padding=(1, 2),
        ))

    if found_email:
        conf = found_email.get("confidence", 0) or 0
        console.print(
            f"\n[dim]Contact:[/dim] {found_email.get('email') or 'not found'} "
            f"[dim](confidence: {conf:.0%}, source: {found_email.get('source', '—')})[/dim]"
        )

    if state.get("error"):
        console.print(f"\n[red]Error: {state['error']}[/red]")

    if state.get("log"):
        console.print("\n[bold]Pipeline Log[/bold]")
        for line in state["log"]:
            console.print(f"  • {line}")

    console.print()


def usage():
    console.print(
        "[bold]SalesNeuron — Pipeline Orchestrator[/bold]\n\n"
        "Usage:\n"
        "  python run_pipeline.py [URL] --sender-email EMAIL [options]\n"
        "  python run_pipeline.py --check-replies --sender-email EMAIL\n"
        "  python run_pipeline.py --follow-ups --sender-email EMAIL\n"
        "  python run_pipeline.py --daily-maintenance --sender-email EMAIL\n\n"
        "Options:\n"
        "  --sender NAME        Your name in the email signature\n"
        "  --role TITLE         Your role in the email signature\n"
        "  --sender-email EMAIL Your Gmail address (required for sending)\n"
        "  --refresh            Force fresh research, ignore cache\n"
        "  --auto-send          Send automatically if confidence is high enough\n"
        "  --min-confidence N   Confidence threshold for auto-send (default 0.70)\n\n"
        "Examples:\n"
        '  python run_pipeline.py https://razorpay.com --sender "Suyash" --sender-email you@gmail.com\n'
        '  python run_pipeline.py https://stripe.com --sender-email you@gmail.com --auto-send\n'
    )


async def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        usage()
        return

    def get_flag_value(flag: str, default: str = "") -> str:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                return args[idx + 1]
        return default

    sender_email = get_flag_value("--sender-email")

    # ── Maintenance commands ───────────────────────────────────────
    if "--daily-maintenance" in args:
        if not sender_email:
            console.print("[red]--sender-email is required[/red]")
            return
        console.print("\n[bold green]SalesNeuron[/bold green] — Daily Maintenance\n")
        result = await daily_maintenance(sender_email, get_flag_value("--sender", ""))
        console.print(
            f"[green]✅ Done[/green] — "
            f"{result['replies_found']} new replies, "
            f"{result['followups_sent']} follow-ups sent\n"
        )
        return

    if "--check-replies" in args:
        if not sender_email:
            console.print("[red]--sender-email is required[/red]")
            return
        console.print("\n[bold green]SalesNeuron[/bold green] — Checking Replies\n")
        count = await check_replies(sender_email)
        console.print(f"[green]✅ {count} new reply(s) found[/green]\n")
        return

    if "--follow-ups" in args:
        if not sender_email:
            console.print("[red]--sender-email is required[/red]")
            return
        console.print("\n[bold green]SalesNeuron[/bold green] — Processing Follow-ups\n")
        count = await process_followups(sender_email, get_flag_value("--sender", ""))
        console.print(f"[green]✅ {count} follow-up(s) sent[/green]\n")
        return

    # ── Full pipeline run ──────────────────────────────────────────
    url = args[0]
    if not url.startswith("http"):
        url = "https://" + url

    if not sender_email:
        console.print("[red]--sender-email is required to run the pipeline[/red]")
        return

    sender_name = get_flag_value("--sender", "Your Name")
    sender_role = get_flag_value("--role", "Founder, SalesNeuron")
    force_refresh = "--refresh" in args
    auto_send = "--auto-send" in args
    min_confidence = float(get_flag_value("--min-confidence", "0.70"))

    console.print(
        f"\n[bold green]SalesNeuron[/bold green] [dim]— Full Pipeline (LangGraph)[/dim]\n"
        f"[dim]Target: {url}[/dim]\n"
        f"[dim]Sender: {sender_name} · {sender_role} · {sender_email}[/dim]\n"
        f"[dim]Mode: {'auto-send' if auto_send else 'human approval required'}[/dim]\n"
    )

    result = await run_pipeline(
        company_url=url,
        sender_name=sender_name,
        sender_role=sender_role,
        sender_email=sender_email,
        force_refresh=force_refresh,
        auto_send=auto_send,
        min_send_confidence=min_confidence,
    )

    print_state_summary(result)

    # ── Human approval loop ────────────────────────────────────────
    if result.get("status") == "awaiting_approval":
        found = result.get("found_email") or {}
        confirmed = Confirm.ask(
            f"Send this email to [bold]{found.get('email')}[/bold]?",
            default=False,
        )
        if confirmed:
            result = await resume_and_send(result)
            console.print()
            if result.get("status") == "sent":
                console.print("[bold green]✅ Sent![/bold green]")
            else:
                console.print(f"[red]❌ Send did not complete: {result.get('status')}[/red]")
        else:
            console.print("[dim]Skipped — email not sent.[/dim]")

    # ── Save run log ────────────────────────────────────────────────
    out_dir = Path("data/pipeline_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    company_slug = (result.get("profile") or {}).get(
        "company_name", "unknown"
    ).lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{company_slug}_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    console.print(f"\n[dim]💾 Run log saved → {out_path}[/dim]\n")

    # Auto check for new replies after every successful send
    if result.get("status") == "sent" and sender_email:
        try:
            new_replies = await check_replies(sender_email)
            if new_replies > 0:
                console.print(f"[bold green]📬 {new_replies} new reply(s) detected![/bold green]\n")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())