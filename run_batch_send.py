"""
SalesNeuron — Batch Sender
=============================
Sends all emails from a completed run_batch.py output in one go,
instead of sending each company one at a time through
run_sequence_manager.py.

This is a SEPARATE file from run_sequence_manager.py by design — the
single-email send path (used by run_pipeline.py's interactive approval
and run_sequence_manager.py --send) is untouched. This script only
reads a batch JSON and reuses SequenceManager.send() directly; it does
not modify, import internals from, or share state with the single-send
CLI in any way that could change its behavior.

Workflow:
    1. python run_batch.py --companies ... --output data/batch_runs/mybatch.json
       (produces the JSON this script reads — nothing is sent yet)
    2. python run_batch_send.py --file data/batch_runs/mybatch.json --sender-email you@gmail.com
       (shows every company that's ready to send, asks ONE confirmation
       for the whole batch, then sends each with its own per-company
       follow-up schedule — same as a normal send)

Skips automatically:
    - Companies with no contact_email found
    - Companies where cold_email data wasn't saved (crashed during batch)
    - Companies below --min-confidence (default 0.70) UNLESS --force is passed

Usage:
    python run_batch_send.py --file data/batch_runs/batch_20260819.json \
        --sender "Suyash" --sender-email you@gmail.com

    python run_batch_send.py --file data/batch_runs/batch_20260819.json \
        --sender-email you@gmail.com --min-confidence 0.5

    python run_batch_send.py --file data/batch_runs/batch_20260819.json \
        --sender-email you@gmail.com --only razorpay.com,linear.app
"""

import asyncio
import json
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

from agents.sequence_manager import SequenceManager
from core.email_models import ColdEmail

console = Console()


def get_flag(args: list, flag: str, default: str = "") -> str:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


def load_batch(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)
    return json.loads(p.read_text())


def build_sendable_list(
    results: list[dict],
    min_confidence: float,
    only_domains: list[str] | None,
    force: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Split batch results into (sendable, skipped) based on:
      - has cold_email + contact_email
      - confidence >= min_confidence (unless --force)
      - matches --only filter if provided
    Each returned skipped entry carries a 'reason' for the summary table.
    """
    sendable, skipped = [], []

    for r in results:
        domain = r.get("url", "").replace("https://", "").replace("http://", "").split("/")[0]

        if only_domains and not any(d in domain for d in only_domains):
            continue  # silently excluded, not "skipped" — user didn't ask for it

        cold_email = r.get("cold_email")
        found_email = r.get("found_email")
        contact_email = (found_email or {}).get("email")
        confidence = (found_email or {}).get("confidence", 0.0)

        if not cold_email:
            skipped.append({**r, "reason": "no email drafted (research/personalize failed)"})
            continue
        if not contact_email:
            skipped.append({**r, "reason": "no contact email found"})
            continue
        if not force and confidence < min_confidence:
            skipped.append({**r, "reason": f"confidence {confidence:.0%} below {min_confidence:.0%} (use --force to include)"})
            continue

        sendable.append(r)

    return sendable, skipped


def print_preview(sendable: list[dict], skipped: list[dict]):
    table = Table(title="📬 Batch Send Preview", show_lines=True)
    table.add_column("Company", style="bold")
    table.add_column("Contact")
    table.add_column("Email")
    table.add_column("Confidence", justify="right")
    table.add_column("Subject")

    for r in sendable:
        cold_email = r["cold_email"]
        found_email = r["found_email"]
        table.add_row(
            r["company_name"],
            cold_email.get("contact_name") or "—",
            found_email["email"],
            f"{found_email.get('confidence', 0.0):.0%}",
            cold_email.get("subject", "—"),
        )

    console.print()
    console.print(table)

    if skipped:
        console.print("\n[yellow]Skipped (will NOT be sent):[/yellow]")
        for r in skipped:
            console.print(f"  • {r.get('company_name', r.get('url'))}: {r['reason']}")

    console.print(f"\n[bold]{len(sendable)} email(s) ready to send.[/bold]\n")


async def send_batch(
    sendable: list[dict],
    sender_email: str,
    sender_name: str,
) -> list[dict]:
    """Send each email via SequenceManager.send() — same underlying send
    path as everywhere else in the project. Continues past individual
    failures so one bad send doesn't stop the batch."""
    sm = SequenceManager()
    await sm.init()

    results = []
    for i, r in enumerate(sendable, 1):
        company = r["company_name"]
        cold_email = ColdEmail(**r["cold_email"])
        recipient = r["found_email"]["email"]

        console.print(f"[dim]({i}/{len(sendable)}) Sending to {company} → {recipient}...[/dim]")
        try:
            send_result = await sm.send(
                cold_email=cold_email,
                sender_email=sender_email,
                sender_name=sender_name,
                recipient_email=recipient,
                find_email_if_missing=False,  # already resolved in the batch run
            )
            status = send_result.get("status", "unknown")
            icon = "✅" if status == "sent" else "❌"
            console.print(f"  {icon} {company}: {status}")
            results.append({"company": company, "email": recipient, **send_result})
        except Exception as e:
            console.print(f"  ❌ {company}: send failed — {e}")
            results.append({"company": company, "email": recipient, "status": "failed", "error": str(e)})

    return results


async def main():
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        console.print(
            "[bold]SalesNeuron — Batch Sender[/bold]\n\n"
            "Usage:\n"
            "  python run_batch_send.py --file BATCH.json --sender-email you@gmail.com\n\n"
            "Options:\n"
            "  --sender NAME         Your name in the signature (default: from batch data)\n"
            "  --min-confidence N    Skip emails below this confidence (default 0.70)\n"
            "  --force               Send even low-confidence emails\n"
            "  --only domain1,domain2   Only send to these companies from the batch\n\n"
            "This does not touch run_sequence_manager.py — separate send path,\n"
            "same underlying SequenceManager.send().\n"
        )
        return

    file_path = get_flag(args, "--file")
    sender_email = get_flag(args, "--sender-email")
    sender_name = get_flag(args, "--sender", "Your Name")
    min_confidence = float(get_flag(args, "--min-confidence", "0.70"))
    force = "--force" in args
    only_flag = get_flag(args, "--only")
    only_domains = [d.strip() for d in only_flag.split(",")] if only_flag else None

    if not file_path:
        console.print("[red]--file is required[/red]")
        return
    if not sender_email:
        console.print("[red]--sender-email is required[/red]")
        return

    results = load_batch(file_path)
    sendable, skipped = build_sendable_list(results, min_confidence, only_domains, force)

    console.print(
        f"\n[bold green]SalesNeuron[/bold green] [dim]— Batch Sender[/dim]\n"
        f"[dim]Loaded {len(results)} companies from {file_path}[/dim]\n"
    )

    if not sendable:
        console.print("[yellow]Nothing ready to send.[/yellow]\n")
        print_preview(sendable, skipped)
        return

    print_preview(sendable, skipped)

    confirmed = Confirm.ask(
        f"Send all {len(sendable)} email(s) now?", default=False
    )
    if not confirmed:
        console.print("[dim]Cancelled — nothing sent.[/dim]\n")
        return

    send_results = await send_batch(sendable, sender_email, sender_name)

    sent_count = sum(1 for r in send_results if r.get("status") == "sent")
    console.print(
        f"\n[bold green]✅ {sent_count}/{len(sendable)} sent successfully[/bold green]\n"
    )

    out_dir = Path("data/batch_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    out_path = out_dir / f"batch_send_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(send_results, f, indent=2, default=str)
    console.print(f"[dim]💾 Send results saved → {out_path}[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())