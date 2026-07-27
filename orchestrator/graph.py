"""
SalesNeuron — LangGraph Orchestrator
=======================================
Wires every agent into a single autonomous pipeline:

    research → personalize → find_email → approval_gate → send

Design decisions (read before modifying):

1. RETRY-ON-LOW-CONFIDENCE instead of a separate "Navigator boost" node.
   The Researcher already uses the Site Knowledge Graph internally for
   deep page discovery. A second research pass with force_refresh=True
   re-runs the same (now-improved) discovery logic — it does not need
   its own graph node. Capped at 2 attempts to avoid burning LLM tokens
   on sites that are simply thin (e.g. very small companies).

2. HUMAN APPROVAL is implemented as a graph-exit + resume, not a
   LangGraph `interrupt`/checkpointer. Reasoning: checkpointer-based
   interrupts require a persistence backend (SQLite/Postgres saver)
   purely to pause a synchronous CLI session — added infrastructure
   for no real benefit here, since the "pause" already survives in the
   returned `state` dict and the CLI immediately decides whether to
   resume. `resume_and_send()` re-enters at the same point deterministically.
   If this becomes a web app with multiple users pausing across
   requests, swap this for a real checkpointer then — not before.

3. FOLLOW-UPS AND REPLY CHECKS are NOT graph nodes. They are scheduled,
   recurring maintenance operations independent of any single company's
   pipeline run (see orchestrator/scheduler.py). Cramming them into this
   graph would force one artificial "session" to own operations that
   naturally span all companies, all the time.

Usage:
    from orchestrator.graph import build_graph, run_pipeline

    result = await run_pipeline("https://stripe.com", sender_email="you@gmail.com")
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from orchestrator.state import PipelineState, new_state, log_line

from agents.researcher import ResearcherAgent
from agents.personalizer import PersonalizerAgent
from agents.sequence_manager import SequenceManager
from core.email_finder import email_finder
from core.models import ProspectProfile
from core.email_models import ColdEmail

logger = logging.getLogger(__name__)

MAX_RESEARCH_ATTEMPTS = 2


# ────────────────────────────────────────────────────────────────────
# Node 1 — Research
# ────────────────────────────────────────────────────────────────────

async def research_node(state: PipelineState) -> dict:
    """Run the Researcher Agent. Populates state['profile']."""
    attempts = state.get("research_attempts", 0) + 1
    logger.info(f"🔍 [Node: research] attempt {attempts} — {state['company_url']}")

    try:
        agent = ResearcherAgent()
        force = state["force_refresh"] or attempts > 1
        profile: ProspectProfile = await agent.research(
            state["company_url"], force_refresh=force
        )
        return {
            "profile": profile.model_dump(),
            "research_attempts": attempts,
            "log": log_line(state, f"Research attempt {attempts}: "
                                    f"confidence={profile.research_confidence}, "
                                    f"signals={len(profile.buying_signals)}"),
        }
    except Exception as e:
        logger.error(f"Research failed: {e}")
        return {
            "research_attempts": attempts,
            "status": "failed",
            "error": f"Research failed: {e}",
            "log": log_line(state, f"❌ Research failed: {e}"),
        }


def route_after_research(state: PipelineState) -> Literal["retry", "personalize", "end"]:
    """
    Conditional edge: retry research once if confidence is low,
    otherwise proceed. Caps retries at MAX_RESEARCH_ATTEMPTS.
    """
    if state.get("status") == "failed":
        return "end"

    profile = state.get("profile")
    if not profile:
        return "end"

    confidence = profile.get("research_confidence", "low")
    attempts = state.get("research_attempts", 0)

    if confidence == "low" and attempts < MAX_RESEARCH_ATTEMPTS:
        logger.info(f"⚠️  Low confidence ({confidence}) — retrying research fresh")
        return "retry"

    return "personalize"


# ────────────────────────────────────────────────────────────────────
# Node 2 — Personalize
# ────────────────────────────────────────────────────────────────────

async def personalize_node(state: PipelineState) -> dict:
    """Run the Personalizer Agent. Populates state['cold_email']."""
    logger.info(f"✉️  [Node: personalize] {state['company_url']}")

    try:
        profile = ProspectProfile(**state["profile"])
        agent = PersonalizerAgent()
        email: ColdEmail = await agent.personalize(
            profile=profile,
            sender_name=state["sender_name"],
            sender_role=state["sender_role"],
        )
        return {
            "cold_email": email.model_dump(),
            "log": log_line(
                state,
                f"Email drafted: '{email.subject}' "
                f"(personalization={email.personalization_score})",
            ),
        }
    except Exception as e:
        logger.error(f"Personalization failed: {e}")
        return {
            "status": "failed",
            "error": f"Personalization failed: {e}",
            "log": log_line(state, f"❌ Personalization failed: {e}"),
        }


# ────────────────────────────────────────────────────────────────────
# Node 3 — Find Email
# ────────────────────────────────────────────────────────────────────

async def find_email_node(state: PipelineState) -> dict:
    """
    Run the Email Finder using the contact the Personalizer already
    selected (ColdEmail.contact_name), so we don't re-derive it.
    """
    cold_email = state.get("cold_email")
    if not cold_email:
        return {"status": "failed", "error": "No cold_email to find contact for"}

    contact_name = cold_email.get("contact_name") or cold_email["company_name"]
    logger.info(f"📧 [Node: find_email] {contact_name} @ {cold_email['company_name']}")

    try:
        await email_finder.init()
        result = await email_finder.find(
            name=contact_name,
            company=cold_email["company_name"],
            website=cold_email["website"],
        )
        confidence = result.get("confidence", 0.0)
        email_addr = result.get("email")
        logger.info(f"📧 Found: {email_addr} (confidence={confidence})")

        return {
            "found_email": result,
            "log": log_line(
                state,
                f"Email found: {email_addr or 'NONE'} "
                f"(confidence={confidence:.0%}, source={result.get('source')})",
            ),
        }
    except Exception as e:
        logger.error(f"Email finding failed: {e}")
        return {
            "found_email": {"email": None, "confidence": 0.0},
            "log": log_line(state, f"⚠️  Email finding errored: {e}"),
        }


def route_after_find_email(state: PipelineState) -> Literal["approval_gate", "no_email"]:
    """No point asking for approval if we have no address to send to."""
    found = state.get("found_email") or {}
    if not found.get("email"):
        return "no_email"
    return "approval_gate"


# ────────────────────────────────────────────────────────────────────
# Node 4 — Approval Gate
# ────────────────────────────────────────────────────────────────────

async def approval_gate_node(state: PipelineState) -> dict:
    """
    Decide whether this email can be auto-sent or needs a human to
    review it first. See module docstring point 2 for why this is a
    graph-exit rather than a checkpointer interrupt.
    """
    found = state.get("found_email") or {}
    confidence = found.get("confidence", 0.0)
    threshold = state["min_send_confidence"]

    if state["auto_send"] and confidence >= threshold:
        logger.info(
            f"✅ Auto-send approved — confidence {confidence:.0%} ≥ {threshold:.0%}"
        )
        return {
            "status": "approved",
            "log": log_line(state, f"Auto-approved for send (confidence {confidence:.0%})"),
        }

    reason = (
        "auto_send disabled" if not state["auto_send"]
        else f"confidence {confidence:.0%} below threshold {threshold:.0%}"
    )
    logger.info(f"⏸️  Awaiting human approval — {reason}")
    return {
        "status": "awaiting_approval",
        "log": log_line(state, f"Paused for human approval ({reason})"),
    }


def route_after_approval(state: PipelineState) -> Literal["send", "end"]:
    return "send" if state.get("status") == "approved" else "end"


# ────────────────────────────────────────────────────────────────────
# Node 5 — Send
# ────────────────────────────────────────────────────────────────────

async def send_node(state: PipelineState) -> dict:
    """
    Send via the Sequence Manager. Passes recipient_email explicitly
    since we already ran Email Finder — avoids a redundant lookup
    inside SequenceManager.send().
    """
    cold_email = ColdEmail(**state["cold_email"])
    found = state.get("found_email") or {}
    recipient = found.get("email")

    logger.info(f"📬 [Node: send] {cold_email.company_name} → {recipient}")

    try:
        sm = SequenceManager()
        await sm.init()
        result = await sm.send(
            cold_email=cold_email,
            sender_email=state["sender_email"],
            sender_name=state["sender_name"],
            recipient_email=recipient,
            find_email_if_missing=False,  # already found above
        )
        return {
            "send_result": result,
            "status": "sent" if result.get("status") == "sent" else result.get("status", "failed"),
            "log": log_line(
                state,
                f"Send result: {result.get('status')} → {result.get('contact_email')}",
            ),
        }
    except Exception as e:
        logger.error(f"Send failed: {e}")
        return {
            "status": "failed",
            "error": f"Send failed: {e}",
            "log": log_line(state, f"❌ Send failed: {e}"),
        }


# ────────────────────────────────────────────────────────────────────
# Node — No email found (terminal, informational)
# ────────────────────────────────────────────────────────────────────

async def no_email_node(state: PipelineState) -> dict:
    logger.info("📭 No verified email found — pipeline stops here")
    return {
        "status": "skipped_no_email",
        "log": log_line(state, "No contact email found — cannot send"),
    }


# ────────────────────────────────────────────────────────────────────
# Graph assembly
# ────────────────────────────────────────────────────────────────────

def build_graph():
    """Compile the LangGraph state machine. Call once, reuse the compiled graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("research", research_node)
    graph.add_node("personalize", personalize_node)
    graph.add_node("find_email", find_email_node)
    graph.add_node("approval_gate", approval_gate_node)
    graph.add_node("send", send_node)
    graph.add_node("no_email", no_email_node)

    graph.set_entry_point("research")

    graph.add_conditional_edges(
        "research",
        route_after_research,
        {"retry": "research", "personalize": "personalize", "end": END},
    )
    graph.add_edge("personalize", "find_email")
    graph.add_conditional_edges(
        "find_email",
        route_after_find_email,
        {"approval_gate": "approval_gate", "no_email": "no_email"},
    )
    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {"send": "send", "end": END},
    )
    graph.add_edge("send", END)
    graph.add_edge("no_email", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Lazily compile and cache the graph — avoid recompiling per call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ────────────────────────────────────────────────────────────────────
# Public entry points
# ────────────────────────────────────────────────────────────────────

async def run_pipeline(
    company_url: str,
    sender_name: str = "Your Name",
    sender_role: str = "Founder, SalesNeuron",
    sender_email: str = "",
    force_refresh: bool = False,
    auto_send: bool = False,
    min_send_confidence: float = 0.70,
) -> PipelineState:
    """
    Run the full pipeline once for one company.
    If auto_send=False (default) and confidence is below threshold,
    stops at status='awaiting_approval' — call resume_and_send() to finish.
    """
    state = new_state(
        company_url=company_url,
        sender_name=sender_name,
        sender_role=sender_role,
        sender_email=sender_email,
        force_refresh=force_refresh,
        auto_send=auto_send,
        min_send_confidence=min_send_confidence,
    )
    graph = get_graph()
    result = await graph.ainvoke(state)
    return result


async def resume_and_send(state: PipelineState) -> PipelineState:
    """
    Resume a pipeline that stopped at 'awaiting_approval' after a human
    has reviewed the draft in state['cold_email'] and decided to send.
    Re-enters directly at the send step — no re-research, re-personalize,
    or re-email-lookup needed since that data is already in state.
    """
    if state.get("status") != "awaiting_approval":
        raise ValueError(
            f"resume_and_send() called on state with status="
            f"'{state.get('status')}' — expected 'awaiting_approval'"
        )
    approved_state = {**state, "status": "approved"}
    result = await send_node(approved_state)
    return {**approved_state, **result}