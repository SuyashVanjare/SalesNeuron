"""
SalesNeuron — Navigator Agent CLI
====================================
Executes navigation flows on websites using the stored Site Knowledge Graph.

Usage:
    python run_navigator.py --site internshala.com --flow search_internships --var query="Python Developer"
    python run_navigator.py --site amazon.com --flow search_products --var query="laptop"
    python run_navigator.py --site linkedin.com --flow easy_apply --var role="AI Engineer"
    python run_navigator.py --list-flows internshala.com
    python run_navigator.py --task "search for Python internships" --site internshala.com

Requirements:
    Site must be learned first:
    python run_explorer.py https://internshala.com
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ["playwright", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agents.navigator import NavigatorAgent
from knowledge.graph_store import graph_store
from core.credentials import credentials
from core.session_store import session_store
import getpass


def masked_password(prompt: str = "Password: ") -> str:
    """
    Read a password from the terminal, echoing '*' for each character
    typed instead of showing nothing. Falls back to plain getpass
    (fully hidden) if the terminal doesn't support this.
    """
    try:
        if sys.platform == "win32":
            import msvcrt
            print(prompt, end="", flush=True)
            chars = []
            while True:
                ch = msvcrt.getch()
                if ch in (b"\r", b"\n"):
                    print()
                    break
                elif ch == b"\x08":  # backspace
                    if chars:
                        chars.pop()
                        print("\b \b", end="", flush=True)
                elif ch == b"\x03":  # Ctrl+C
                    raise KeyboardInterrupt
                else:
                    try:
                        c = ch.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    chars.append(c)
                    print("*", end="", flush=True)
            return "".join(chars)
        else:
            import termios
            import tty
            print(prompt, end="", flush=True)
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            chars = []
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        print()
                        break
                    elif ch == "\x7f":  # backspace
                        if chars:
                            chars.pop()
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                    elif ch == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt
                    else:
                        chars.append(ch)
                        sys.stdout.write("*")
                        sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return "".join(chars)
    except Exception:
        # Any terminal quirk — fall back to fully-hidden input rather than crash
        return getpass.getpass(prompt)

console = Console()


def print_result(result, site: str, flow_name: str) -> None:
    """Pretty print navigation result."""
    console.print()

    status = "[bold green]✅ SUCCESS[/bold green]" if result.success else "[bold red]❌ FAILED[/bold red]"

    console.print(Panel(
        f"{status}\n\n"
        f"[dim]Site:[/dim]  {site}\n"
        f"[dim]Flow:[/dim]  {flow_name}\n"
        f"[dim]Steps:[/dim] {result.steps_completed}/{result.total_steps} completed\n"
        f"[dim]URL:[/dim]   {result.final_url}",
        title="🧭 Navigation Result",
        border_style="green" if result.success else "red",
        padding=(1, 2),
    ))

    # Execution log
    if result.execution_log:
        console.print("\n[bold]Execution Log[/bold]")
        for entry in result.execution_log:
            icon = "✅" if "✅" in entry else "⚠️" if "⚠️" in entry else "❌" if "❌" in entry else "•"
            console.print(f"  {icon} {entry}")

    # Extracted data
    if result.extracted_data:
        console.print("\n[bold]📊 Extracted Data[/bold]")
        console.print(Panel(
            json.dumps(result.extracted_data, indent=2)[:2000],
            border_style="dim",
        ))

    if result.error:
        console.print(f"\n[red]Error: {result.error}[/red]")

    console.print()


def print_deep_result(result, site: str, flow_name: str) -> None:
    """Pretty print deep navigation result (listing + item pages)."""
    console.print()
    status = "[bold green]✅ SUCCESS[/bold green]" if result.success else "[bold red]❌ FAILED[/bold red]"

    console.print(Panel(
        f"{status}\n\n"
        f"[dim]Site:[/dim]   {site}\n"
        f"[dim]Flow:[/dim]   {flow_name}\n"
        f"[dim]Listing:[/dim] {result.listing_url}\n"
        f"[dim]Items scraped:[/dim] {len(result.items)}",
        title="🧭 Deep Navigation Result",
        border_style="green" if result.success else "red",
        padding=(1, 2),
    ))

    if result.execution_log:
        console.print("\n[bold]Execution Log[/bold]")
        for entry in result.execution_log:
            console.print(f"  • {entry}")

    if result.items:
        console.print(f"\n[bold]📦 {len(result.items)} Items[/bold]")
        for i, item in enumerate(result.items[:5], 1):
            console.print(Panel(json.dumps(item, indent=2)[:800], title=f"Item {i}", border_style="dim"))
        if len(result.items) > 5:
            console.print(f"[dim]... and {len(result.items) - 5} more (see saved JSON)[/dim]")

    if result.error:
        console.print(f"\n[red]Error: {result.error}[/red]")
    console.print()


async def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        console.print(
            "[bold]SalesNeuron — Navigator Agent[/bold]\n\n"
            "Usage:\n"
            "  python run_navigator.py --site DOMAIN --flow FLOW_NAME [--var key=value]\n"
            "  python run_navigator.py --task DESCRIPTION --site DOMAIN\n"
            "  python run_navigator.py --list-flows DOMAIN\n"
            "  python run_navigator.py --save-credentials DOMAIN\n"
            "  python run_navigator.py --site DOMAIN --flow FLOW_NAME --deep --max-items N [--paginate]\n\n"
            "Examples:\n"
            '  python run_navigator.py --site internshala.com --flow search_internships --var query="Python"\n'
            '  python run_navigator.py --site amazon.com --flow search_products --var query="laptop" --extract\n'
            '  python run_navigator.py --task "search for ML jobs" --site internshala.com\n'
            "  python run_navigator.py --list-flows internshala.com\n"
            "  python run_navigator.py --save-credentials internshala.com\n"
            '  python run_navigator.py --site internshala.com --flow search_internships --var query="Python" --deep --max-items 15 --paginate\n\n'
            "Note: Learn a site first with: python run_explorer.py https://DOMAIN\n"
        )
        sys.exit(0)

    await graph_store.init()

    # ── --list-flows DOMAIN ────────────────────────────────────────
    if args[0] == "--list-flows":
        domain = args[1] if len(args) > 1 else ""
        if not domain:
            console.print("[red]Usage: python run_navigator.py --list-flows DOMAIN[/red]")
            return

        flows = await graph_store.list_flows(domain)
        if not flows:
            console.print(
                f"[yellow]No flows found for {domain}.[/yellow]\n"
                f"Learn it first: [cyan]python run_explorer.py https://{domain}[/cyan]"
            )
            return

        console.print(f"\n[bold]Available flows for {domain}[/bold]\n")
        t = Table(box=box.SIMPLE)
        t.add_column("Flow Name", style="green")
        t.add_column("Description")
        t.add_column("Variables", style="cyan")
        t.add_column("Steps", style="dim")

        for f in flows:
            vars_raw = f.get("variables", "[]")
            try:
                vars_list = json.loads(vars_raw) if isinstance(vars_raw, str) else vars_raw
            except Exception:
                vars_list = []
            t.add_row(
                f["flow_name"],
                f["description"] or "—",
                ", ".join(vars_list) if vars_list else "none",
                str(f["steps_count"]),
            )
        console.print(t)
        return

    # ── --delete-credentials DOMAIN ─────────────────────────────────
    if "--delete-credentials" in args or "--remove-credentials" in args:
        idx = args.index("--delete-credentials") if "--delete-credentials" in args else args.index("--remove-credentials")
        domain = args[idx + 1] if idx + 1 < len(args) else ""
        if not domain:
            console.print("[red]Usage: python run_navigator.py --delete-credentials DOMAIN[/red]")
            return
        domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")

        await credentials.init()
        await credentials.delete(domain)
        session_store.clear(domain)
        console.print(f"[green]🗑️  Credentials and active session deleted for {domain}[/green]\n")
        return

    # ── --save-credentials DOMAIN ───────────────────────────────────
    if args[0] == "--save-credentials":
        domain = args[1] if len(args) > 1 else ""
        if not domain:
            console.print("[red]Usage: python run_navigator.py --save-credentials DOMAIN[/red]")
            return
        domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")

        console.print(f"\n[bold]Storing credentials for {domain}[/bold]")
        console.print("[dim]These are encrypted at rest and never sent anywhere except this site.[/dim]\n")
        email = input("Email/username: ").strip()
        password = masked_password("Password: ").strip()

        agent = NavigatorAgent()
        ok = await agent.login(domain, email, password)
        if ok:
            console.print(f"[green]✅ Credentials saved and login verified for {domain}[/green]\n")
        else:
            console.print(
                f"[yellow]⚠️  Credentials saved for {domain}, but login could not be verified "
                f"(no learned login flow yet, or login failed). Run the explorer on the login "
                f"page first, or check the credentials.[/yellow]\n"
            )
        return

    # ── Parse common args ──────────────────────────────────────────
    site = ""
    flow_name = ""
    task = ""
    variables = {}
    extract_after = "--extract" in args
    watch = "--watch" in args
    deep = "--deep" in args
    paginate = "--paginate" in args
    max_items = 10
    if "--max-items" in args:
        idx = args.index("--max-items")
        try:
            max_items = int(args[idx + 1])
        except (IndexError, ValueError):
            pass

    if watch:
        os.environ["HEADLESS_BROWSER"] = "false"

    # Parse --site
    if "--site" in args:
        idx = args.index("--site")
        site = args[idx + 1] if idx + 1 < len(args) else ""
        site = site.replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")

    # Parse --flow
    if "--flow" in args:
        idx = args.index("--flow")
        flow_name = args[idx + 1] if idx + 1 < len(args) else ""

    # Parse --task
    if "--task" in args:
        idx = args.index("--task")
        task = args[idx + 1] if idx + 1 < len(args) else ""

    # Parse --var key=value (multiple allowed)
    i = 0
    while i < len(args):
        if args[i] == "--var" and i + 1 < len(args):
            pair = args[i + 1]
            if "=" in pair:
                k, v = pair.split("=", 1)
                variables[k.strip()] = v.strip()
        i += 1

    if not site:
        console.print("[red]--site is required[/red]")
        sys.exit(1)

    if not flow_name and not task:
        console.print("[red]Either --flow or --task is required[/red]")
        sys.exit(1)

    console.print(
        f"\n[bold green]SalesNeuron[/bold green] [dim]— Navigator Agent[/dim]\n"
        f"[dim]Site: {site}[/dim]\n"
        f"[dim]Flow: {flow_name or 'auto-select from task'}[/dim]\n"
        f"[dim]Variables: {variables}[/dim]\n"
    )

    # ── Execute ────────────────────────────────────────────────────
    agent = NavigatorAgent()

    try:
        if deep:
            if not flow_name:
                console.print("[red]--deep requires --flow[/red]")
                sys.exit(1)
            result = await agent.execute_deep(
                site=site,
                flow_name=flow_name,
                variables=variables,
                max_items=max_items,
                follow_pagination=paginate,
            )
        elif task and not flow_name:
            # Let LLM pick the right flow from the task description
            result = await agent.execute_task(site, task, variables)
        else:
            result = await agent.execute(
                site=site,
                flow_name=flow_name,
                variables=variables,
                extract_after=extract_after,
            )
    except Exception as e:
        console.print(f"[red]❌ Navigation failed: {e}[/red]")
        raise

    # Print results
    if deep:
        print_deep_result(result, site, flow_name)
    else:
        print_result(result, site, flow_name or task)

    # Save result
    out_dir = Path("data/navigation")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{site}_{flow_name or 'task'}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    console.print(f"[dim]💾 Result saved → {out_path}[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())