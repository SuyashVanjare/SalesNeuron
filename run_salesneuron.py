"""
SalesNeuron — Full Pipeline CLI
==================================
Runs the complete pipeline in one command:
  1. Research the company (or load from cache)
  2. Generate personalized cold email via RAG

Usage:
    python run_salesneuron.py https://stripe.com
    python run_salesneuron.py https://linear.app --sender "Suyash" --role "Founder, SalesNeuron"
    python run_salesneuron.py https://vercel.com --refresh
    python run_salesneuron.py --build-kb          (rebuild knowledge base)
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
              "chromadb", "sentence_transformers", "transformers"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from agents.researcher import ResearcherAgent
from agents.personalizer import PersonalizerAgent
from core.knowledge_base import kb

console = Console()


def print_email(email) -> None:
    """Pretty print the generated cold email."""
    console.print()
    console.print(Rule("[bold green]✉️  Generated Cold Email[/bold green]"))
    console.print()

    # Metadata
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("", style="dim", width=22)
    t.add_column("", style="white")
    t.add_row("To", f"{email.contact_name or 'Decision Maker'} @ {email.company_name}")
    t.add_row("Subject", f"[bold]{email.subject}[/bold]")
    t.add_row("Signal used", email.buying_signal_used[:70])
    t.add_row("Product angle", email.product_angle[:70])
    score_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
        email.personalization_score, "white"
    )
    t.add_row(
        "Personalization",
        f"[{score_color}]{email.personalization_score.upper()}[/{score_color}]"
    )
    console.print(t)

    # Email body
    console.print()
    console.print(Panel(
        email.body,
        title="[dim]Email Body[/dim]",
        border_style="dim",
        padding=(1, 2),
    ))

    # RAG chunks used
    if email.knowledge_chunks_used:
        console.print(
            f"\n[dim]📚 Knowledge chunks used: "
            f"{', '.join(email.knowledge_chunks_used)}[/dim]"
        )
    console.print()


async def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        console.print(
            "[bold]SalesNeuron — Full Pipeline[/bold]\n\n"
            "Usage:\n"
            "  python run_salesneuron.py [URL] [options]\n\n"
            "Options:\n"
            "  --sender NAME    Your name in the email signature\n"
            "  --role TITLE     Your role in the email signature\n"
            "  --refresh        Force re-scrape even if cached\n"
            "  --build-kb       Rebuild the product knowledge base\n\n"
            "Examples:\n"
            '  python run_salesneuron.py https://stripe.com --sender "Suyash"\n'
            "  python run_salesneuron.py https://linear.app --refresh\n"
            "  python run_salesneuron.py --build-kb\n"
        )
        sys.exit(0)

    # ── --build-kb ─────────────────────────────────────────────────
    if args[0] == "--build-kb":
        console.print("\n[bold green]SalesNeuron[/bold green] — Building Knowledge Base\n")
        await kb.init()
        await kb.build()
        console.print("[green]✅ Knowledge base ready[/green]")
        return

    # ── Parse args ─────────────────────────────────────────────────
    url = args[0]
    if not url.startswith("http"):
        url = "https://" + url

    sender = "Your Name"
    role = "Founder, SalesNeuron"
    force_refresh = "--refresh" in args

    if "--sender" in args:
        idx = args.index("--sender")
        if idx + 1 < len(args):
            sender = args[idx + 1]
    if "--role" in args:
        idx = args.index("--role")
        if idx + 1 < len(args):
            role = args[idx + 1]

    console.print(
        f"\n[bold green]SalesNeuron[/bold green] [dim]— Full Pipeline[/dim]\n"
        f"[dim]Target: {url}[/dim]\n"
        f"[dim]Sender: {sender} · {role}[/dim]\n"
    )

    # ── Step 1: Research ───────────────────────────────────────────
    console.print("[bold]Step 1/2 — Researching prospect...[/bold]")
    try:
        researcher = ResearcherAgent()
        profile = await researcher.research(url, force_refresh=force_refresh)
    except Exception as e:
        console.print(f"[red]❌ Research failed: {e}[/red]")
        raise

    console.print(
        f"[green]✅ Research done[/green] — "
        f"{profile.company_name} | "
        f"{len(profile.buying_signals)} signals | "
        f"confidence: {profile.research_confidence.upper()}\n"
    )

    # ── Step 2: Personalize ────────────────────────────────────────
    console.print("[bold]Step 2/2 — Generating personalized email...[/bold]")
    try:
        personalizer = PersonalizerAgent()
        email = await personalizer.personalize(
            profile=profile,
            sender_name=sender,
            sender_role=role,
        )
    except Exception as e:
        console.print(f"[red]❌ Personalization failed: {e}[/red]")
        raise

    # ── Print results ──────────────────────────────────────────────
    print_email(email)

    # ── Save output ────────────────────────────────────────────────
    out_dir = Path("data/emails")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = profile.company_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"{slug}_{timestamp}_email.json"

    with open(out_path, "w") as f:
        json.dump(json.loads(email.model_dump_json()), f, indent=2)

    console.print(f"[dim]💾 Email saved → {out_path}[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())