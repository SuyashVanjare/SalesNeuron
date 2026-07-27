"""
SalesNeuron — Site Explorer CLI
=================================
Teaches the agent how a website works. Run this once per site.
The knowledge graph is stored in SQLite and reused forever.

Usage:
    python run_explorer.py https://amazon.com
    python run_explorer.py https://linkedin.com --watch
    python run_explorer.py --list
    python run_explorer.py --flows amazon.com
    python run_explorer.py --forget amazon.com
    python run_explorer.py https://stripe.com --refresh
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
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ["playwright", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from knowledge.explorer import SiteExplorer
from knowledge.graph_store import graph_store

console = Console()


def print_graph(graph) -> None:
    """Pretty print a SiteGraph."""
    console.print()
    console.print(Panel(
        f"[bold white]{graph.domain}[/bold white]\n"
        f"[dim]{graph.base_url}[/dim]\n\n"
        f"[white]{graph.description}[/white]",
        title=f"[bold green]🗺️  Site Knowledge Graph[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Stats
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("", style="dim", width=20)
    t.add_column("", style="white")
    t.add_row("Site type", graph.site_type.upper())
    t.add_row("Pages mapped", str(len(graph.pages)))
    t.add_row("Edges found", str(len(graph.edges)))
    t.add_row("Flows built", str(len(graph.flows)))
    console.print(t)

    # Pages
    if graph.pages:
        console.print("\n[bold]📄 Pages Mapped[/bold]")
        for page in graph.pages:
            auth = " [dim](requires auth)[/dim]" if page.requires_auth else ""
            console.print(
                f"  [cyan]{page.page_type}[/cyan]{auth}\n"
                f"  [dim]{page.url[:70]}[/dim]\n"
                f"  {page.purpose}\n"
                f"  [dim]{len(page.elements)} interactive elements[/dim]\n"
            )

    # Flows — most important part
    if graph.flows:
        console.print("\n[bold]⚡ Navigation Flows (what agents can do)[/bold]")
        for flow in graph.flows:
            vars_str = (
                f" [dim]vars: {', '.join(flow.variables)}[/dim]"
                if flow.variables else ""
            )
            console.print(
                f"  [bold green]{flow.flow_name}[/bold green]{vars_str}\n"
                f"  {flow.description}\n"
                f"  [dim]{len(flow.steps)} steps[/dim]\n"
            )
            for i, step in enumerate(flow.steps, 1):
                console.print(
                    f"    {i}. [cyan]{step.action_type}[/cyan] "
                    f"[dim]{step.selector}[/dim]"
                    + (f" → type '{step.input_value}'" if step.input_value else "")
                    + f"\n       {step.description}"
                )
            console.print()
    else:
        console.print("\n[dim]No flows synthesized[/dim]")

    console.print()


async def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        console.print(
            "[bold]SalesNeuron — Site Explorer[/bold]\n\n"
            "Usage:\n"
            "  python run_explorer.py [URL]           Learn a website\n"
            "  python run_explorer.py --list          Show all learned sites\n"
            "  python run_explorer.py --flows DOMAIN  Show flows for a site\n"
            "  python run_explorer.py --forget DOMAIN Delete a site's graph\n\n"
            "Options:\n"
            "  --watch     Show browser window while learning\n"
            "  --refresh   Re-learn even if cached\n\n"
            "Examples:\n"
            "  python run_explorer.py https://amazon.com\n"
            "  python run_explorer.py https://linkedin.com --watch\n"
            "  python run_explorer.py --flows amazon.com\n"
        )
        sys.exit(0)

    await graph_store.init()

    # ── --list ─────────────────────────────────────────────────────
    if args[0] == "--list":
        sites = await graph_store.all_sites()
        stats = await graph_store.stats()

        console.print(f"\n[bold green]SalesNeuron[/bold green] — Learned Sites\n")
        console.print(
            f"[dim]{stats['total_sites']} sites · "
            f"{stats['total_flows']} flows · "
            f"{stats['total_flow_executions']} executions[/dim]\n"
        )

        if not sites:
            console.print("[dim]No sites learned yet. Run: python run_explorer.py https://example.com[/dim]")
            return

        t = Table(box=box.SIMPLE)
        t.add_column("Domain", style="white")
        t.add_column("Type", style="dim")
        t.add_column("Pages", style="cyan")
        t.add_column("Flows", style="green")
        t.add_column("Last Learned", style="dim")

        for s in sites:
            t.add_row(
                s["domain"],
                s["site_type"] or "—",
                str(s["pages_explored"]),
                str(s["flows_count"]),
                s["updated_at"][:16].replace("T", " "),
            )
        console.print(t)
        return

    # ── --flows DOMAIN ─────────────────────────────────────────────
    if args[0] == "--flows":
        domain = args[1] if len(args) > 1 else ""
        if not domain:
            console.print("[red]Usage: python run_explorer.py --flows amazon.com[/red]")
            return

        flows = await graph_store.list_flows(domain)
        if not flows:
            console.print(f"[dim]No flows for {domain}. Learn it first.[/dim]")
            return

        console.print(f"\n[bold]Flows for {domain}[/bold]\n")
        for f in flows:
            console.print(
                f"  [bold green]{f['flow_name']}[/bold green] — {f['description']}\n"
                f"  [dim]{f['steps_count']} steps · "
                f"used {f['times_used']}x · "
                f"success {f['success_rate']*100:.0f}%[/dim]\n"
            )
        return

    # ── --forget / --delete / --remove DOMAIN ──────────────────────
    if args[0] in ("--forget", "--delete", "--remove"):
        domain = args[1] if len(args) > 1 else ""
        domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        if not domain:
            console.print("[red]Usage: python run_explorer.py --forget DOMAIN[/red]")
            return
        await graph_store.delete(domain)
        console.print(f"[green]🗑️  Deleted learned graph and flows for {domain}[/green]")
        return

    # ── Learn a URL ────────────────────────────────────────────────
    url = args[0]
    force_refresh = "--refresh" in args
    if "--watch" in args:
        os.environ["HEADLESS_BROWSER"] = "false"

    if not url.startswith("http"):
        url = "https://" + url

    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace("www.", "")

    # Check cache
    if not force_refresh:
        cached = await graph_store.get(domain)
        if cached:
            console.print(
                f"\n[bold green]SalesNeuron[/bold green] — Site Explorer\n"
                f"[dim]⚡ Cache hit — {domain} already learned. Use --refresh to re-learn.[/dim]\n"
            )
            print_graph(cached)
            return

    console.print(
        f"\n[bold green]SalesNeuron[/bold green] [dim]— Site Explorer[/dim]\n"
        f"[dim]Learning: {url}[/dim]\n"
    )

    try:
        explorer = SiteExplorer()
        graph = await explorer.learn(url)
    except Exception as e:
        console.print(f"[bold red]❌ Learning failed:[/bold red] {e}")
        raise

    # Save to DB
    await graph_store.save(graph)

    # Print results
    print_graph(graph)

    # Save JSON backup
    out_dir = Path("data/site_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        f.write(graph.model_dump_json(indent=2))

    stats = await graph_store.stats()
    console.print(
        f"[dim]💾 JSON → {out_path}[/dim]\n"
        f"[dim]🗄️  DB  → {stats['db_path']} ({stats['total_sites']} sites learned)[/dim]\n"
    )


if __name__ == "__main__":
    asyncio.run(main())