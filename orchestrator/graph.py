"""
SalesNeuron — LangGraph Orchestrator
=======================================
Full autonomous pipeline:

    explore → research → personalize → find_email → approval_gate → send

Node 0 — Explorer
  Learns website structure ONCE and stores it as a SiteGraph in SQLite.
  On subsequent runs for the same domain, this node is instant (cache hit).
  Researcher uses the stored graph to navigate directly to key pages
  instead of blindly guessing URLs every time.

Node 1 — Research
  Uses SiteGraph if available. Retries once on low confidence.

Node 2 — Personalize
  RAG + LLM → cold email grounded in real buying signals.

Node 3 — Find Email
  4-tier: person search → our free engine (WHOIS/GitHub/site/DNS) →
          Hunter domain search → skip.

Node 4 — Approval Gate
  Human review before sending (or auto-send if flag set).

Node 5 — Send
  Gmail API → Sequence Manager → auto-schedules follow-ups.
"""

import logging
from typing import Literal
from urllib.parse import urlparse

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


# ──────────────────────────────────────────────────────────────────
# Node 0 — Explorer (learn site structure once, reuse forever)
# ──────────────────────────────────────────────────────────────────

async def explore_node(state: PipelineState) -> dict:
    """
    Learn the website structure and store as SiteGraph in SQLite.
    Skips automatically if graph already exists for this domain.
    """
    url = state["company_url"].rstrip("/")
    domain = urlparse(url).netloc.replace("www.", "")

    try:
        from knowledge.graph_store import graph_store
        await graph_store.init()
        existing = await graph_store.get(domain)

        if existing:
            logger.info(f"🗺️  [Node: explore] Graph HIT — {domain} already learned ({len(existing.pages)} pages)")
            return {"log": log_line(state, f"Site graph loaded: {domain} ({len(existing.pages)} pages)")}

        logger.info(f"🗺️  [Node: explore] Learning {domain} for the first time...")
        from knowledge.explorer import SiteExplorer
        explorer = SiteExplorer()
        # BUG FIX: the method is learn(), not explore() — SiteExplorer
        # never had an .explore() attribute. This was silently caught by
        # the except block below and logged as "Explorer skipped:
        # 'SiteExplorer' object has no attribute 'explore'", so EVERY
        # first-time-seen domain fell through without a SiteGraph and
        # ResearcherAgent had to fall back to homepage-link discovery
        # from scratch every single run.
        graph = await explorer.learn(url)

        if graph:
            await graph_store.save(graph)
            logger.info(f"🗺️  [Node: explore] Learned {domain}: {len(graph.pages)} pages, {len(graph.flows)} flows")
            return {"log": log_line(state, f"Site graph built: {domain} ({len(graph.pages)} pages, {len(graph.flows)} flows)")}
        else:
            logger.warning(f"🗺️  [Node: explore] Could not learn {domain} — continuing without graph")
            return {"log": log_line(state, f"Site graph failed for {domain} — continuing anyway")}

    except Exception as e:
        logger.warning(f"🗺️  [Node: explore] Explorer failed: {e} — continuing")
        return {"log": log_line(state, f"Explorer skipped: {e}")}


# ──────────────────────────────────────────────────────────────────
# Node 1 — Research
# ──────────────────────────────────────────────────────────────────

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
            "log": log_line(state, f"Research attempt {attempts}: confidence={profile.research_confidence}, signals={len(profile.buying_signals)}"),
        }
    except Exception as e:
        logger.error(f"Research failed: {e}")
        return {
            "research_attempts": attempts,
            "status": "failed",
            "error": f"Research failed: {e}",
            "log": log_line(state, f"❌ Research failed: {e}"),
        }


def route_after_research(state):
    if state.get("status") == "failed":
        return "end"
    profile = state.get("profile")
    if not profile:
        return "end"
    confidence = profile.get("research_confidence", "low")
    attempts = state.get("research_attempts", 0)
    pages = len(profile.get("pages_scraped", []))
    signals = len(profile.get("buying_signals", []))
    if confidence == "low" and attempts < MAX_RESEARCH_ATTEMPTS and pages > 2 and signals == 0:
        return "retry"
    return "personalize"


# ──────────────────────────────────────────────────────────────────
# Node 2 — Personalize
# ──────────────────────────────────────────────────────────────────

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
            "log": log_line(state, f"Email drafted: '{email.subject}' (personalization={email.personalization_score})"),
        }
    except Exception as e:
        logger.error(f"Personalization failed: {e}")
        return {
            "status": "failed",
            "error": f"Personalization failed: {e}",
            "log": log_line(state, f"❌ Personalization failed: {e}"),
        }


# ──────────────────────────────────────────────────────────────────
# Node 3 — Find Email
# ──────────────────────────────────────────────────────────────────

async def find_email_node(state: PipelineState) -> dict:
    """
    4-tier email finding — works for ANY company:

    Tier 1 — Person search
      Uses contact_name from Personalizer or profile key_people.
      Full name → Hunter/Snov/Apollo → highest confidence.

    Tier 2a — Our free engine (find_by_domain)
      WHOIS → GitHub → site scrape → role emails + DNS verify.
      Zero API credits. Works even for brand new startups.

    Tier 2b — Hunter domain search
      Last paid resort. Uses 1 credit. Returns senior exec email.

    Tier 3 — Skip
      Nothing found anywhere. Pipeline stops cleanly.

    IMPORTANT — scraper wiring:
      email_finder.find() and find_by_domain() both accept an optional
      `scraper` argument that lets them read real emails directly off
      the company's contact/about/team pages (via
      EmailFinder._scrape_emails_from_site, which itself now prefers
      the real pages Explorer already discovered over guessed paths).
      Previously this node called both WITHOUT a scraper, silently
      skipping that entire step — meaning an email sitting in plain
      text on a page we'd already scraped (e.g. info@company.com on
      the Contact page) was never picked up, and the pipeline fell
      through to a low-confidence guessed "hello@domain" instead.
    """
    cold_email = state.get("cold_email")
    if not cold_email:
        return {"status": "failed", "error": "No cold_email"}

    contact_name = cold_email.get("contact_name")
    domain = cold_email.get("website", "").replace("https://", "").replace("http://", "").split("/")[0].replace("www.", "")

    # Pull names from profile key_people if personalizer missed it
    # Try ALL found people in sequence — if the first name fails
    # normalization (e.g. hyphenated "P-E Lallemant") or yields no
    # result, the next person is tried rather than giving up immediately.
    key_people_names: list[str] = []
    if not contact_name:
        profile = state.get("profile") or {}
        key_people = profile.get("key_people", []) or []
        for person in key_people:
            name = person.get("name", "") if isinstance(person, dict) else getattr(person, "name", "")
            if name and name.strip():
                key_people_names.append(name.strip())
        if key_people_names:
            contact_name = key_people_names[0]
            logger.info(f"📧 Using profile key people: {key_people_names[:3]}")

    try:
        await email_finder.init()

        from core.scraper import BrowserScraper
        async with BrowserScraper() as scraper:

            # Tier 1 — person search (try each found person in turn)
            candidates = []
            if contact_name and contact_name != cold_email.get("company_name"):
                candidates.append(contact_name)
            for name in key_people_names[1:]:
                if name not in candidates:
                    candidates.append(name)

            for person_name in candidates[:5]:  # cap at 5 attempts
                logger.info(f"📧 [Node: find_email] Person search: {person_name} @ {domain}")
                try:
                    result = await email_finder.find(
                        name=person_name,
                        company=cold_email["company_name"],
                        website=cold_email["website"],
                        scraper=scraper,
                    )
                except Exception as e:
                    logger.debug(f"📧 Person search failed for {person_name}: {e}")
                    continue
                if result and result.get("email"):
                    confidence = result.get("confidence", 0.0)
                    _inject_name(state, person_name)
                    logger.info(f"📧 Person search found: {result['email']} ({confidence:.0%})")
                    return {
                        "found_email": result,
                        "log": log_line(state, f"Email found: {result['email']} (confidence={confidence:.0%}, source={result.get('source')})"),
                    }

            # Tier 2a — our free engine
            if domain:
                logger.info(f"📧 [Node: find_email] Free engine for {domain}")
                domain_result = await email_finder.find_by_domain(domain, scraper=scraper)

                # Tier 2b — Hunter domain search
                if not domain_result or not domain_result.get("email"):
                    logger.info(f"📧 Free engine missed — trying Hunter domain search")
                    from core.providers import provider_waterfall
                    domain_result = await provider_waterfall.domain_search(domain)

                if domain_result and domain_result.get("email"):
                    confidence = domain_result.get("confidence", 0.0)
                    found_name = domain_result.get("contact_name")
                    if found_name:
                        _inject_name(state, found_name)
                        if not contact_name:
                            state["cold_email"]["contact_name"] = found_name
                    logger.info(f"📧 Domain engine found: {domain_result['email']} ({domain_result.get('contact_title', '')})")
                    return {
                        "found_email": domain_result,
                        "log": log_line(state, f"Email found via domain: {domain_result['email']} (confidence={confidence:.0%})"),
                    }

            # Tier 3 — nothing found
            logger.warning("📧 All email strategies exhausted")
            return {
                "found_email": {"email": None, "confidence": 0.0},
                "log": log_line(state, "⚠️  No contact email found via any method"),
            }

    except Exception as e:
        logger.error(f"Email finding failed: {e}")
        return {
            "found_email": {"email": None, "confidence": 0.0},
            "log": log_line(state, f"⚠️  Email finding errored: {e}"),
        }


def _inject_name(state: PipelineState, full_name: str):
    """
    Replace generic greeting with real person's first name.
    Fixes "Hi there" and "Hi [First Name]" in email body.
    """
    cold_email = state.get("cold_email")
    if not cold_email or not full_name:
        return
    first_name = full_name.strip().split()[0].title()
    body = cold_email.get("body", "") or ""
    for placeholder in ["Hi there,", "Hi there", "Hi [First Name],", "Hi [First Name]", "there,"]:
        if placeholder in body:
            body = body.replace(placeholder, f"Hi {first_name},", 1)
            break
    # Also fix subject line if it has placeholders
    subject = cold_email.get("subject", "") or ""
    subject = subject.replace("[First Name]", first_name)
    cold_email["body"] = body
    cold_email["subject"] = subject


def route_after_find_email(state: PipelineState) -> Literal["approval_gate", "no_email"]:
    found = state.get("found_email") or {}
    return "approval_gate" if found.get("email") else "no_email"


# ──────────────────────────────────────────────────────────────────
# Node 4 — Approval Gate
# ──────────────────────────────────────────────────────────────────

async def approval_gate_node(state: PipelineState) -> dict:
    found = state.get("found_email") or {}
    confidence = found.get("confidence", 0.0)
    threshold = state["min_send_confidence"]

    if state["auto_send"] and confidence >= threshold:
        logger.info(f"✅ Auto-send approved — confidence {confidence:.0%} ≥ {threshold:.0%}")
        return {"status": "approved", "log": log_line(state, f"Auto-approved (confidence {confidence:.0%})")}

    reason = "auto_send disabled" if not state["auto_send"] else f"confidence {confidence:.0%} below threshold {threshold:.0%}"
    logger.info(f"⏸️  Awaiting human approval — {reason}")
    return {"status": "awaiting_approval", "log": log_line(state, f"Paused for approval ({reason})")}


def route_after_approval(state: PipelineState) -> Literal["send", "end"]:
    return "send" if state.get("status") == "approved" else "end"


# ──────────────────────────────────────────────────────────────────
# Node 5 — Send
# ──────────────────────────────────────────────────────────────────

async def send_node(state: PipelineState) -> dict:
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
            find_email_if_missing=False,
        )
        return {
            "send_result": result,
            "status": "sent" if result.get("status") == "sent" else result.get("status", "failed"),
            "log": log_line(state, f"Send result: {result.get('status')} → {result.get('contact_email')}"),
        }
    except Exception as e:
        logger.error(f"Send failed: {e}")
        return {
            "status": "failed",
            "error": f"Send failed: {e}",
            "log": log_line(state, f"❌ Send failed: {e}"),
        }


# ──────────────────────────────────────────────────────────────────
# Terminal nodes
# ──────────────────────────────────────────────────────────────────

async def no_email_node(state: PipelineState) -> dict:
    logger.info("📭 No verified email found — pipeline stops here")
    return {"status": "skipped_no_email", "log": log_line(state, "No contact email found — cannot send")}


# ──────────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("explore", explore_node)
    graph.add_node("research", research_node)
    graph.add_node("personalize", personalize_node)
    graph.add_node("find_email", find_email_node)
    graph.add_node("approval_gate", approval_gate_node)
    graph.add_node("send", send_node)
    graph.add_node("no_email", no_email_node)

    graph.set_entry_point("explore")
    graph.add_edge("explore", "research")

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
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_pipeline(
    company_url: str,
    sender_name: str = "Your Name",
    sender_role: str = "Founder, SalesNeuron",
    sender_email: str = "",
    force_refresh: bool = False,
    auto_send: bool = False,
    min_send_confidence: float = 0.70,
) -> PipelineState:
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
    if state.get("status") != "awaiting_approval":
        raise ValueError(f"Expected status=awaiting_approval, got {state.get('status')}")
    approved_state = {**state, "status": "approved"}
    result = await send_node(approved_state)
    return {**approved_state, **result}