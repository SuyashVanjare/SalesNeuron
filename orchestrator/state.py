"""
SalesNeuron — Orchestrator State
==================================
The shared state object that flows through every node in the LangGraph.
Each node reads what it needs and writes back updated fields.

Kept as a plain TypedDict (not Pydantic) because LangGraph merges
partial dict updates between nodes — TypedDict is the natural fit.
"""

from typing import TypedDict, Optional, Any
from typing_extensions import NotRequired


class PipelineState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    company_url: str
    force_refresh: bool
    sender_name: str
    sender_role: str
    sender_email: str
    auto_send: bool          # True = skip human approval, send immediately
    min_send_confidence: float  # email confidence threshold to auto-send

    # ── Research stage ─────────────────────────────────────────────
    profile: NotRequired[Optional[dict]]         # ProspectProfile as dict
    research_attempts: NotRequired[int]

    # ── Personalization stage ───────────────────────────────────────
    cold_email: NotRequired[Optional[dict]]       # ColdEmail as dict

    # ── Email discovery stage ───────────────────────────────────────
    found_email: NotRequired[Optional[dict]]      # EmailFinder result dict

    # ── Send stage ───────────────────────────────────────────────────
    send_result: NotRequired[Optional[dict]]

    # ── Control flow / observability ─────────────────────────────────
    status: NotRequired[str]      # running | awaiting_approval | sent |
                                   # skipped_no_email | failed
    error: NotRequired[Optional[str]]
    log: NotRequired[list[str]]


def new_state(
    company_url: str,
    sender_name: str = "Your Name",
    sender_role: str = "Founder, SalesNeuron",
    sender_email: str = "",
    force_refresh: bool = False,
    auto_send: bool = False,
    min_send_confidence: float = 0.70,
) -> PipelineState:
    """Build a fresh initial state for one pipeline run."""
    return PipelineState(
        company_url=company_url,
        force_refresh=force_refresh,
        sender_name=sender_name,
        sender_role=sender_role,
        sender_email=sender_email,
        auto_send=auto_send,
        min_send_confidence=min_send_confidence,
        profile=None,
        research_attempts=0,
        cold_email=None,
        found_email=None,
        send_result=None,
        status="running",
        error=None,
        log=[],
    )


def log_line(state: PipelineState, message: str) -> list[str]:
    """Append a log line, returning the new log list (immutable-style update)."""
    existing = state.get("log", []) or []
    return existing + [message]