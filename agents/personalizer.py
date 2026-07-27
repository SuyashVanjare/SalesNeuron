"""
SalesNeuron — Personalizer Agent
===================================
Takes a ProspectProfile (from Researcher Agent) and generates a
hyper-personalized cold email using RAG over the product knowledge base.

Pipeline:
  1. SIGNAL SELECTION  — pick the strongest buying signal to hook on
  2. RAG RETRIEVAL     — find relevant product knowledge for that signal
  3. CONTACT SELECTION — pick the best person to email
  4. EMAIL GENERATION  — LLM writes the email grounded in real facts
  5. QUALITY CHECK     — score and validate the output

Input:  ProspectProfile
Output: ColdEmail (subject + body + metadata)

Usage:
    agent = PersonalizerAgent()
    email = await agent.personalize(profile)
    print(email.subject)
    print(email.body)
"""

import json
import logging
from typing import Optional

from core.llm import llm
from core.models import ProspectProfile
from core.email_models import ColdEmail
from core.knowledge_base import kb

logger = logging.getLogger(__name__)


class PersonalizerAgent:

    async def personalize(
        self,
        profile: ProspectProfile,
        sender_name: str = "Your Name",
        sender_role: str = "Founder, SalesNeuron",
        custom_cta: Optional[str] = None,
    ) -> ColdEmail:
        """
        Generate a personalized cold email for a prospect.

        Args:
            profile: ProspectProfile from ResearcherAgent
            sender_name: your name (appears in email signature)
            sender_role: your title
            custom_cta: override the default call-to-action
        """
        logger.info(f"\n{'='*55}")
        logger.info(f"✉️  PersonalizerAgent: {profile.company_name}")
        logger.info(f"{'='*55}")

        # ── Step 1: Initialize knowledge base ─────────────────────
        await kb.init()
        count = await self._kb_count()
        if count == 0:
            logger.warning(
                "⚠️  Knowledge base is empty! Building with default product knowledge..."
            )
            await kb.build()

        # ── Step 2: Select strongest buying signal ─────────────────
        logger.info("⚡ Step 1/4 — Selecting best buying signal...")
        best_signal = self._select_best_signal(profile)
        logger.info(
            f"   Signal: [{best_signal.get('strength', '?').upper()}] "
            f"{best_signal.get('signal_type')} — {best_signal.get('description', '')[:60]}"
        )

        # ── Step 3: RAG — find relevant product knowledge ──────────
        logger.info("🔍 Step 2/4 — RAG retrieval from knowledge base...")
        relevant_chunks = await kb.search_for_signals(profile.buying_signals)
        logger.info(f"   Retrieved {len(relevant_chunks)} relevant knowledge chunks")
        for chunk in relevant_chunks:
            logger.info(f"   → {chunk['title']}")

        # ── Step 4: Select best contact ────────────────────────────
        logger.info("👤 Step 3/4 — Selecting target contact...")
        contact = self._select_contact(profile)
        logger.info(
            f"   Contact: {contact['name']} ({contact['title']})"
            if contact["name"] != "there"
            else "   No specific contact found — using generic greeting"
        )

        # ── Step 5: Generate email ─────────────────────────────────
        logger.info("🧠 Step 4/4 — Generating personalized email...")
        email = await self._generate_email(
            profile=profile,
            best_signal=best_signal,
            relevant_chunks=relevant_chunks,
            contact=contact,
            sender_name=sender_name,
            sender_role=sender_role,
            custom_cta=custom_cta,
        )

        logger.info(f"✅ Email generated — personalization: {email.personalization_score.upper()}")
        return email

    # ──────────────────────────────────────────────────────────────
    # Step 1 — Select best buying signal
    # ──────────────────────────────────────────────────────────────

    def _select_best_signal(self, profile: ProspectProfile) -> dict:
        """
        Pick the single strongest buying signal to hook the email on.
        Priority: high → medium → low → fallback to recent news.
        """
        if not profile.buying_signals:
            # No signals — use recent news as hook
            if profile.recent_news:
                return {
                    "signal_type": "recent_news",
                    "description": profile.recent_news[0],
                    "strength": "medium",
                }
            return {
                "signal_type": "general",
                "description": f"{profile.company_name} is growing in {profile.industry}",
                "strength": "low",
            }

        # Sort by strength
        strength_order = {"high": 0, "medium": 1, "low": 2}
        sorted_signals = sorted(
            profile.buying_signals,
            key=lambda s: strength_order.get(
                s.strength if hasattr(s, "strength") else s.get("strength", "low"), 2
            ),
        )

        best = sorted_signals[0]
        if hasattr(best, "model_dump"):
            return best.model_dump()
        return best

    # ──────────────────────────────────────────────────────────────
    # Step 2 — Select contact
    # ──────────────────────────────────────────────────────────────

    def _select_contact(self, profile: ProspectProfile) -> dict:
        """
        Pick the best person to email from key_people.
        Priority: CEO/Founder > VP Sales > Head of Growth > CTO > first available.
        """
        if not profile.key_people:
            return {"name": "there", "title": ""}

        priority_titles = [
            "ceo", "founder", "co-founder", "president",
            "vp sales", "head of sales", "chief revenue",
            "vp growth", "head of growth", "vp marketing",
            "cto", "vp engineering", "head of product",
        ]

        for priority in priority_titles:
            for person in profile.key_people:
                title = (person.title if hasattr(person, "title") else person.get("title", "")).lower()
                name = person.name if hasattr(person, "name") else person.get("name", "")
                if priority in title and name:
                    return {
                        "name": name.split()[0],  # first name only
                        "title": person.title if hasattr(person, "title") else person.get("title", ""),
                    }

        # Fall back to first person
        first = profile.key_people[0]
        name = first.name if hasattr(first, "name") else first.get("name", "there")
        title = first.title if hasattr(first, "title") else first.get("title", "")
        return {"name": name.split()[0], "title": title}

    # ──────────────────────────────────────────────────────────────
    # Step 3 — Generate the email
    # ──────────────────────────────────────────────────────────────

    async def _generate_email(
        self,
        profile: ProspectProfile,
        best_signal: dict,
        relevant_chunks: list[dict],
        contact: dict,
        sender_name: str,
        sender_role: str,
        custom_cta: Optional[str],
    ) -> ColdEmail:
        """
        LLM generates the email grounded in:
        - Specific buying signal (real fact about this company)
        - Relevant product knowledge (from ChromaDB RAG)
        - Contact name and title
        """

        # Build RAG context from retrieved chunks
        rag_context = "\n\n".join([
            f"[{c['title']}]\n{c['content']}"
            for c in relevant_chunks
        ])

        chunk_titles = [c["title"] for c in relevant_chunks]

        cta = custom_cta or "Would you be open to a 15-minute call this week?"

        # Build the prompt
        signal_type = best_signal.get("signal_type", "general")
        signal_desc = best_signal.get("description", "")
        signal_strength = best_signal.get("strength", "low")

        prompt = (
            "You are an expert B2B cold email writer. Write a hyper-personalized cold email.\n\n"
            "RULES:\n"
            "- Under 150 words total (people skim emails)\n"
            "- First line MUST reference the specific buying signal below — a real fact\n"
            "- Second paragraph connects their situation to ONE specific product benefit\n"
            "- End with the CTA provided\n"
            "- NO generic opener like 'I hope this finds you well'\n"
            "- NO fluff, NO buzzwords\n"
            "- Sound like a human, not a robot\n"
            "- Use first name only in greeting\n\n"
            "PROSPECT:\n"
            f"  Company: {profile.company_name}\n"
            f"  Industry: {profile.industry}\n"
            f"  Description: {profile.description}\n"
            f"  Contact: {contact['name']} ({contact['title']})\n\n"
            "BUYING SIGNAL TO HOOK ON:\n"
            f"  Type: {signal_type}\n"
            f"  Detail: {signal_desc}\n"
            f"  Strength: {signal_strength}\n\n"
            "RELEVANT PRODUCT KNOWLEDGE (use this to match their pain to your solution):\n"
            f"{rag_context}\n\n"
            "SENDER:\n"
            f"  Name: {sender_name}\n"
            f"  Role: {sender_role}\n\n"
            f"CTA: {cta}\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "subject": "email subject line (under 8 words, specific, no clickbait)",\n'
            '  "body": "full email body with greeting and signature",\n'
            '  "product_angle": "which product feature/benefit you matched to this prospect",\n'
            '  "personalization_score": "high/medium/low"\n'
            "}\n"
            "No markdown, pure JSON only."
        )

        try:
            result = await llm.generate_structured(prompt, "{}", temperature=0.7)

            return ColdEmail(
                company_name=profile.company_name,
                website=profile.website,
                contact_name=contact["name"] if contact["name"] != "there" else None,
                contact_title=contact.get("title"),
                subject=result.get("subject", f"Quick question about {profile.company_name}"),
                body=result.get("body", ""),
                buying_signal_used=f"{signal_type}: {signal_desc}",
                product_angle=result.get("product_angle", ""),
                personalization_score=result.get("personalization_score", "medium"),
                knowledge_chunks_used=chunk_titles,
            )

        except Exception as e:
            logger.error(f"Email generation failed: {e}")
            # Fallback minimal email
            return ColdEmail(
                company_name=profile.company_name,
                website=profile.website,
                contact_name=contact["name"],
                subject=f"Idea for {profile.company_name}",
                body=(
                    f"Hi {contact['name']},\n\n"
                    f"Noticed {signal_desc}.\n\n"
                    f"We help companies like {profile.company_name} with sales automation.\n\n"
                    f"{cta}\n\n"
                    f"Best,\n{sender_name}"
                ),
                buying_signal_used=f"{signal_type}: {signal_desc}",
                product_angle="general sales automation",
                personalization_score="low",
                knowledge_chunks_used=[],
            )

    async def _kb_count(self) -> int:
        """Get count of items in knowledge base."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, kb._collection.count)
        except Exception:
            return 0