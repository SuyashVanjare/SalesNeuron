"""
SalesNeuron — Batch Tester
=============================
Runs the full pipeline (explore -> research -> personalize -> find email)
against a list of companies WITHOUT sending anything, and prints a
summary table so you can spot patterns across companies at once instead
of testing one at a time and losing the comparison.

Never sends real emails — auto_send stays False, and the pipeline
naturally stops at the approval gate for every company. Nothing in this
script can send mail; it's read-only from Gmail's perspective.

Usage:
    python run_batch.py --companies razorpay.com,linear.app,stripe.com \
        --sender "Suyash" --role "Founder, SalesNeuron"

    python run_batch.py --file companies.txt --sender "Suyash" --role "Founder, SalesNeuron"

    (companies.txt = one URL or bare domain per line, # comments allowed)

Options:
    --refresh          Force fresh research for every company (ignore cache)
    --output PATH       Save results as JSON (default: data/batch_runs/TIMESTAMP.json)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,  # quiet during batch — summary table is the point
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from orchestrator.graph import run_pipeline

console = Console()


def get_flag(args: list, flag: str, default: str = "") -> str:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


def load_companies(args: list) -> list[str]:
    companies_flag = get_flag(args, "--companies")
    file_flag = get_flag(args, "--file")

    urls = []
    if companies_flag:
        urls = [c.strip() for c in companies_flag.split(",") if c.strip()]
    elif file_flag:
        path = Path(file_flag)
        if not path.exists():
            console.print(f"[red]File not found: {file_flag}[/red]")
            sys.exit(1)
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    return [u if u.startswith("http") else f"https://{u}" for u in urls]


async def run_one(url: str, sender_name: str, sender_role: str, force_refresh: bool) -> dict:
    """Run the pipeline for one company, catching errors so one bad
    company can't kill the whole batch."""
    try:
        result = await run_pipeline(
            company_url=url,
            sender_name=sender_name,
            sender_role=sender_role,
            sender_email="",  # not needed — auto_send stays False, never sends
            force_refresh=force_refresh,
            auto_send=False,
            min_send_confidence=0.70,
        )
        profile = result.get("profile") or {}
        cold_email = result.get("cold_email") or {}
        found_email = result.get("found_email") or {}

        return {
            "url": url,
            "status": result.get("status", "unknown"),
            "error": result.get("error"),
            "company_name": profile.get("company_name", "?"),
            "research_confidence": profile.get("research_confidence", "?"),
            "signals": len(profile.get("buying_signals", [])),
            "personalization": cold_email.get("personalization_score", "?"),
            "contact_name": cold_email.get("contact_name") or "—",
            "contact_email": found_email.get("email") or "—",
            "email_confidence": found_email.get("confidence", 0.0),
            "email_source": found_email.get("source", "—"),
            # Full data preserved for a separate batch-send step —
            # kept alongside (not instead of) the summary fields above
            # so run_batch.py's own table/print logic needs no changes.
            "cold_email": cold_email if cold_email else None,
            "found_email": found_email if found_email else None,
        }
    except Exception as e:
        logger_msg = f"{type(e).__name__}: {e}"
        return {
            "url": url,
            "status": "crashed",
            "error": logger_msg,
            "company_name": "?",
            "research_confidence": "?",
            "signals": 0,
            "personalization": "?",
            "contact_name": "—",
            "contact_email": "—",
            "email_confidence": 0.0,
            "email_source": "—",
        }


def print_summary(results: list[dict]):
    table = Table(title="📊 Batch Test Results", show_lines=True)
    table.add_column("Company", style="bold")
    table.add_column("Confidence")
    table.add_column("Signals", justify="right")
    table.add_column("Personalization")
    table.add_column("Contact")
    table.add_column("Email")
    table.add_column("Email Conf.", justify="right")
    table.add_column("Status")

    status_colors = {
        "awaiting_approval": "green",
        "skipped_no_email": "yellow",
        "failed": "red",
        "crashed": "red",
    }

    for r in results:
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
            r["research_confidence"], "white"
        )
        status_color = status_colors.get(r["status"], "white")
        email_conf_pct = f"{r['email_confidence']:.0%}" if r["email_confidence"] else "—"

        table.add_row(
            r["company_name"],
            f"[{conf_color}]{r['research_confidence']}[/{conf_color}]",
            str(r["signals"]),
            r["personalization"],
            r["contact_name"],
            r["contact_email"],
            email_conf_pct,
            f"[{status_color}]{r['status']}[/{status_color}]",
        )

    console.print()
    console.print(table)

    # Quick aggregate stats
    total = len(results)
    with_email = sum(1 for r in results if r["contact_email"] != "—")
    high_conf = sum(1 for r in results if r["research_confidence"] == "high")
    crashed = sum(1 for r in results if r["status"] == "crashed")

    console.print(Panel(
        f"Companies tested: {total}\n"
        f"Found a contact email: {with_email}/{total} ({with_email/total*100:.0f}%)\n"
        f"High research confidence: {high_conf}/{total} ({high_conf/total*100:.0f}%)\n"
        f"Crashed: {crashed}/{total}",
        title="Summary",
        border_style="cyan",
    ))

    if crashed:
        console.print("\n[red]Crashes:[/red]")
        for r in results:
            if r["status"] == "crashed":
                console.print(f"  • {r['url']}: {r['error']}")
    console.print()


async def main():
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        console.print(
            "[bold]SalesNeuron — Batch Tester[/bold]\n\n"
            "Usage:\n"
            "  python run_batch.py --companies url1,url2,url3 --sender NAME --role TITLE\n"
            "  python run_batch.py --file companies.txt --sender NAME --role TITLE\n\n"
            "Options:\n"
            "  --refresh       Force fresh research (ignore cache)\n"
            "  --output PATH   Save results as JSON\n\n"
            "Never sends real emails — stops at the approval gate for every company.\n"
        )
        return

    companies = load_companies(args)
    if not companies:
        console.print("[red]No companies provided. Use --companies or --file.[/red]")
        return

    sender_name = get_flag(args, "--sender", "Your Name")
    sender_role = get_flag(args, "--role", "Founder, SalesNeuron")
    force_refresh = "--refresh" in args
    output_path = get_flag(args, "--output")

    console.print(
        f"\n[bold green]SalesNeuron[/bold green] [dim]— Batch Tester[/dim]\n"
        f"[dim]Testing {len(companies)} companies · sender: {sender_name}[/dim]\n"
        f"[yellow]No emails will be sent — this stops at the approval gate.[/yellow]\n"
    )

    results = []
    for i, url in enumerate(companies, 1):
        console.print(f"[dim]({i}/{len(companies)}) Running: {url}...[/dim]")
        result = await run_one(url, sender_name, sender_role, force_refresh)
        results.append(result)
        icon = "✅" if result["status"] != "crashed" else "❌"
        console.print(f"  {icon} {result['company_name']} — {result['status']}")

    print_summary(results)

    if not output_path:
        out_dir = Path("data/batch_runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(out_dir / f"batch_{timestamp}.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[dim]💾 Full results saved → {output_path}[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())