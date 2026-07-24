"""
SalesNeuron — Researcher Agent CLI
===================================
Usage:
    python run_researcher.py https://stripe.com
    python run_researcher.py https://notion.so --output data/notion_profile.json
    python run_researcher.py https://linear.app --watch   # shows browser (non-headless)

The researched profile is printed to terminal + saved to ./data/<company>_profile.json
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Load .env before importing agents
from dotenv import load_dotenv
load_dotenv()

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy third-party loggers
for noisy in ["playwright", "httpx", "httpcore", "urllib3", "asyncio"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agents.researcher import ResearcherAgent
from core.memory import memory

console = Console()


def print_profile(profile) -> None:
    """Pretty-print the ProspectProfile with Rich."""

    console.print()
    console.print(Panel(
        f"[bold white]{profile.company_name}[/bold white]\n"
        f"[dim]{profile.website}[/dim]\n\n"
        f"[white]{profile.description}[/white]",
        title="[bold green]✅ Prospect Profile[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Basic info table
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    info.add_column("Field", style="dim", width=20)
    info.add_column("Value", style="white")

    fields = [
        ("Industry", profile.industry),
        ("Size", profile.company_size or "—"),
        ("Founded", profile.founded_year or "—"),
        ("HQ", profile.headquarters or "—"),
        ("Confidence", profile.research_confidence.upper()),
        ("Pages scraped", str(len(profile.pages_scraped))),
    ]
    for label, val in fields:
        info.add_row(label, val)
    console.print(info)

    # Key people
    if profile.key_people:
        console.print("\n[bold]👤 Key People[/bold]")
        for person in profile.key_people[:5]:
            console.print(f"  • {person.name} — [dim]{person.title}[/dim]")

    # Buying signals — the most important part for sales
    if profile.buying_signals:
        console.print("\n[bold]⚡ Buying Signals[/bold]")
        strength_colors = {"high": "bold red", "medium": "yellow", "low": "dim"}
        for signal in profile.buying_signals:
            color = strength_colors.get(signal.strength, "white")
            console.print(
                f"  [{color}][{signal.strength.upper()}][/{color}] "
                f"[cyan]{signal.signal_type}[/cyan]: {signal.description}"
            )
    else:
        console.print("\n[dim]No buying signals detected[/dim]")

    # Tech stack
    if profile.tech_stack:
        console.print("\n[bold]🔧 Tech Stack[/bold]")
        for stack in profile.tech_stack:
            tools = ", ".join(stack.tools)
            console.print(f"  • [dim]{stack.category}:[/dim] {tools}")

    # Open roles — structured display
    if profile.open_job_roles:
        velocity = f" [dim]({profile.hiring_velocity})[/dim]" if getattr(profile, 'hiring_velocity', None) else ""
        console.print(f"\n[bold]💼 Open Roles[/bold]{velocity}")
        for role in profile.open_job_roles[:8]:
            if isinstance(role, dict):
                title = role.get("title", "Unknown")
                loc   = role.get("location") or ""
                team  = role.get("team") or ""
                skills = ", ".join(role.get("skills", [])[:3])
                meta = " · ".join(filter(None, [team, loc, skills]))
                console.print(f"  • [white]{title}[/white]" + (f" [dim]— {meta}[/dim]" if meta else ""))
            else:
                console.print(f"  • {role}")

    # Recent news
    if profile.recent_news:
        console.print("\n[bold]📰 Recent News[/bold]")
        for news in profile.recent_news[:5]:
            console.print(f"  • {news}")

    console.print()


async def main():
    # Parse args
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        console.print(
            "[bold]SalesNeuron — Researcher Agent[/bold]\n\n"
            "Usage:\n"
            "  python run_researcher.py [URL] [options]\n\n"
            "Options:\n"
            "  --output FILE   Save JSON to this path\n"
            "  --watch         Show browser (non-headless mode)\n\n"
            "Examples:\n"
            "  python run_researcher.py https://stripe.com\n"
            "  python run_researcher.py https://notion.so --watch\n"
        )
        sys.exit(0)

    # ── Special commands ───────────────────────────────────────────
    # python run_researcher.py --list
    if args[0] == "--list":
        await memory.init()
        companies = await memory.all_companies()
        stats = await memory.stats()

        console.print(f"\n[bold green]SalesNeuron[/bold green] — Stored Companies\n")
        console.print(f"[dim]DB: {stats['db_path']}[/dim]")
        console.print(
            f"[dim]Total: {stats['total_companies']} companies · "
            f"{stats['fresh_companies']} fresh · "
            f"{stats['stale_companies']} stale · "
            f"{stats['total_signals']} signals[/dim]\n"
        )

        if not companies:
            console.print("[dim]No companies stored yet. Run the agent on a URL first.[/dim]")
            return

        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Company", style="white")
        table.add_column("Industry", style="dim")
        table.add_column("Signals", style="cyan")
        table.add_column("Confidence", style="")
        table.add_column("Last Updated", style="dim")

        for c in companies:
            updated = c["updated_at"][:16].replace("T", " ")
            conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
                (c["confidence"] or "low").lower(), "white"
            )
            table.add_row(
                c["company_name"] or "—",
                (c["industry"] or "—")[:30],
                str(c["signal_count"]),
                f"[{conf_color}]{(c['confidence'] or '—').upper()}[/{conf_color}]",
                updated,
            )
        console.print(table)
        return

    # ── Normal research flow ───────────────────────────────────────
    url = args[0]
    output_path = None
    force_refresh = "--refresh" in args

    if "--output" in args:
        idx = args.index("--output")
        output_path = args[idx + 1] if idx + 1 < len(args) else None
    if "--watch" in args:
        os.environ["HEADLESS_BROWSER"] = "false"

    # Validate URL
    if not url.startswith("http"):
        url = "https://" + url

    console.print(f"\n[bold green]SalesNeuron[/bold green] [dim]— Researcher Agent[/dim]")
    console.print(f"[dim]Target: {url}[/dim]")
    if force_refresh:
        console.print(f"[dim]Mode: force refresh (ignoring cache)[/dim]")
    console.print()

    # Check API keys
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not gemini_key and not groq_key:
        console.print(
            "[bold red]❌ No API keys found.[/bold red]\n\n"
            "Copy .env.example to .env and add your keys:\n"
            "  [cyan]cp .env.example .env[/cyan]\n\n"
            "Get free keys:\n"
            "  Gemini: https://aistudio.google.com (recommended)\n"
            "  Groq:   https://console.groq.com\n"
        )
        sys.exit(1)

    # Run the agent
    try:
        agent = ResearcherAgent()
        profile = await agent.research(url, force_refresh=force_refresh)
    except Exception as e:
        console.print(f"[bold red]❌ Research failed:[/bold red] {e}")
        raise

    # Print results
    print_profile(profile)

    # Save to JSON backup as well
    company_slug = profile.company_name.lower().replace(" ", "_").replace("/", "")
    if not output_path:
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = output_dir / f"{company_slug}_{timestamp}_profile.json"

    profile_json = json.loads(profile.model_dump_json())
    with open(output_path, "w") as f:
        json.dump(profile_json, f, indent=2)

    # Show DB stats
    stats = await memory.stats()
    console.print(
        f"[dim]💾 JSON → {output_path}[/dim]\n"
        f"[dim]🗄️  DB  → {stats['db_path']} "
        f"({stats['total_companies']} companies stored)[/dim]\n"
    )


if __name__ == "__main__":
    asyncio.run(main())