"""
SalesNeuron — Product Knowledge Base
======================================
This is YOUR product's information — what you're selling.
The Personalizer Agent uses RAG over this to match
prospect pain points to relevant features.

HOW TO USE:
  1. Edit PRODUCT_KNOWLEDGE below with your actual product info
  2. Run: python -c "from core.knowledge_base import kb; import asyncio; asyncio.run(kb.build())"
  3. ChromaDB stores embeddings locally in data/chromadb/
  4. Personalizer automatically searches this when writing emails

The knowledge base uses Google's free embedding API (no extra cost).
Falls back to ChromaDB's built-in embeddings if Gemini unavailable.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = str(Path(os.getenv("CHROMA_PATH", "data/chromadb")))
FEEDBACK_DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))

# ──────────────────────────────────────────────────────────────────
# ✏️  EDIT THIS — your product information
# ──────────────────────────────────────────────────────────────────

PRODUCT_KNOWLEDGE = [
    {
        "id": "product_overview",
        "category": "overview",
        "title": "What is SalesNeuron",
        "content": (
            "SalesNeuron is an autonomous AI sales intelligence platform that helps "
            "B2B sales teams research prospects, identify buying signals, and send "
            "hyper-personalized cold emails at scale. It replaces 80% of manual SDR work "
            "by automating prospect research, email personalization, and follow-up sequences."
        ),
    },
    {
        "id": "pain_prospect_research",
        "category": "pain_point",
        "title": "Problem: Manual prospect research takes hours",
        "content": (
            "SDR teams spend 3-4 hours per prospect manually researching company news, "
            "finding key contacts, checking tech stacks, and reading LinkedIn — "
            "just to write one cold email. SalesNeuron does this in 60 seconds by "
            "autonomously browsing the prospect's website, extracting buying signals, "
            "and building a structured profile ready for personalization."
        ),
    },
    {
        "id": "pain_generic_emails",
        "category": "pain_point",
        "title": "Problem: Generic cold emails get ignored",
        "content": (
            "Average cold email reply rates are 1-3% because they're generic mail-merge "
            "templates. SalesNeuron generates emails that reference specific recent events "
            "— a funding round, a product launch, a new hire — making each email feel "
            "hand-written. Personalized emails referencing specific events see 3-5x "
            "higher reply rates."
        ),
    },
    {
        "id": "feature_researcher",
        "category": "feature",
        "title": "Feature: Autonomous Prospect Research",
        "content": (
            "SalesNeuron's Researcher Agent uses browser automation to visit prospect "
            "websites, read news pages, check job postings, and extract structured "
            "intelligence including: company description, key decision makers, tech stack, "
            "recent funding, product launches, hiring trends, and buying signals. "
            "Fully automated, results in 60 seconds, cached for 7 days."
        ),
    },
    {
        "id": "feature_buying_signals",
        "category": "feature",
        "title": "Feature: Buying Signal Detection",
        "content": (
            "Automatically detects 7 types of buying signals: hiring surge (rapid team growth), "
            "recent funding (fresh budget), tech migration (moving off competitors), "
            "product launch (new tool needs), leadership change (new decision maker), "
            "geographic expansion (scaling pains), and pain point mentions "
            "(direct problems stated on their website). Each signal is scored high/medium/low."
        ),
    },
    {
        "id": "feature_personalizer",
        "category": "feature",
        "title": "Feature: AI Email Personalization",
        "content": (
            "The Personalizer Agent reads the prospect profile and buying signals, "
            "then crafts a cold email that: references a specific recent event at their company, "
            "connects their pain point to your product's solution, includes a clear CTA, "
            "and stays under 150 words for maximum readability. "
            "Uses RAG over your product knowledge base for accurate feature matching."
        ),
    },
    {
        "id": "feature_site_graph",
        "category": "feature",
        "title": "Feature: Site Knowledge Graph",
        "content": (
            "SalesNeuron builds a persistent knowledge graph of how websites work — "
            "mapping page types, interactive elements, and navigation flows. "
            "This means the agent learns a website once and reuses that knowledge forever, "
            "making subsequent visits instant. Similar to how StableBrowse works — "
            "but built open-source for sales automation."
        ),
    },
    {
        "id": "usecase_saas_sales",
        "category": "use_case",
        "title": "Use Case: SaaS Sales Teams",
        "content": (
            "SaaS companies with outbound sales teams use SalesNeuron to scale their "
            "SDR operations without hiring. One SDR can run 10x more outreach sequences "
            "because research and personalization are fully automated. "
            "Best fit: B2B SaaS companies with deal sizes above $5K ACV "
            "selling to engineering, product, or operations teams."
        ),
    },
    {
        "id": "usecase_hiring_surge",
        "category": "use_case",
        "title": "Use Case: Companies Experiencing Hiring Surge",
        "content": (
            "When a company is rapidly hiring — especially in sales, engineering, or operations — "
            "it signals growth pains and budget availability. SalesNeuron detects hiring surges "
            "from job posting data and uses this as the email hook: "
            "'Saw you're scaling your sales team — companies at your stage typically struggle with X...'"
        ),
    },
    {
        "id": "usecase_recent_funding",
        "category": "use_case",
        "title": "Use Case: Recently Funded Companies",
        "content": (
            "Companies that just raised funding have fresh budget and pressure to grow fast. "
            "SalesNeuron detects funding events and times outreach within 2 weeks of announcement — "
            "when the new budget is being allocated and vendors are being evaluated. "
            "Email hook: 'Congrats on the Series B — companies investing in growth at your stage often find...'"
        ),
    },
    {
        "id": "usecase_tech_migration",
        "category": "use_case",
        "title": "Use Case: Tech Migration Signals",
        "content": (
            "Job postings mentioning migration away from legacy tools, "
            "or blog posts about switching infrastructure, signal active vendor evaluation. "
            "SalesNeuron detects these migration signals and positions your product "
            "as the modern alternative they're looking for."
        ),
    },
    {
        "id": "competitive_advantage",
        "category": "differentiation",
        "title": "Why SalesNeuron vs Apollo / Clay / Outreach",
        "content": (
            "Apollo and Clay provide contact databases but not true AI research. "
            "Outreach and Salesloft are sequence tools, not intelligence tools. "
            "SalesNeuron is the only platform that: (1) autonomously browses live websites "
            "for real-time intelligence, (2) detects buying signals from actual website content, "
            "and (3) generates emails grounded in specific facts — not generic templates. "
            "No database subscription needed — it reads the live web."
        ),
    },
    {
        "id": "roi_metrics",
        "category": "roi",
        "title": "ROI: Time and Cost Savings",
        "content": (
            "SalesNeuron reduces prospect research time from 3 hours to 60 seconds per company. "
            "A team of 5 SDRs saves 60+ hours per week. "
            "Personalized emails achieve 3-5x higher reply rates vs generic templates. "
            "At $50K average SDR cost, automating 80% of research work saves $200K+ annually "
            "for a team of 5."
        ),
    },
    {
        "id": "usecase_hardware_robotics",
        "category": "use_case",
        "title": "Use Case: Hardware and Robotics Companies",
        "content": (
            "Hardware and robotics companies sell to data centers, manufacturers, and facility "
            "operators — all with very specific buying triggers: infrastructure expansion, "
            "facility upgrades, new compute deployments. SalesNeuron scans live web signals "
            "to find companies actively expanding physical AI compute infrastructure, posting "
            "jobs for facilities engineers, or announcing new data center builds — so your "
            "sales team pitches exactly when these prospects are ready to buy."
        ),
    },
    {
        "id": "usecase_web3_blockchain",
        "category": "use_case",
        "title": "Use Case: Blockchain and Web3 Consulting Firms",
        "content": (
            "Web3 and blockchain consulting firms win clients when companies are actively "
            "exploring blockchain adoption or digital transformation. SalesNeuron detects "
            "tech migration signals — companies mentioning legacy system limitations, posting "
            "jobs for blockchain engineers, or announcing tokenization initiatives. "
            "Reach prospects during their evaluation window instead of cold-pitching "
            "companies with no active need."
        ),
    },
    {
        "id": "usecase_ai_security",
        "category": "use_case",
        "title": "Use Case: AI Security and Compliance Companies",
        "content": (
            "AI security companies (agent guardrails, LLM safety, compliance tools) sell to "
            "companies actively deploying AI agents or LLMs in production. "
            "SalesNeuron detects these companies by scanning job postings for 'AI agents', "
            "'autonomous systems', 'LLM safety', product launches mentioning AI integration, "
            "and pain points like 'data access controls' or 'AI governance'. "
            "Pitch AI security tools exactly when companies are building systems that need them."
        ),
    },
    {
        "id": "usecase_early_stage_founders",
        "category": "use_case",
        "title": "Use Case: Early-Stage Founders Doing Their Own Sales",
        "content": (
            "Early-stage founders spend 10+ hours per week on manual outreach — "
            "researching prospects, writing personalized emails, following up. "
            "SalesNeuron automates this entire workflow so founders can focus on "
            "product and customer conversations instead of spreadsheets. "
            "Best fit: B2B founders doing their own sales before hiring an SDR, "
            "especially targeting technical buyers or other startups."
        ),
    },
    {
        "id": "usecase_computer_vision_ml",
        "category": "use_case",
        "title": "Use Case: Computer Vision and ML Tooling Companies",
        "content": (
            "Companies selling ML infrastructure, computer vision APIs, or AI developer tools "
            "need to find teams actively building with these technologies. "
            "SalesNeuron detects skill demand signals — companies posting jobs for ML engineers, "
            "computer vision specialists, or Python/PyTorch developers — and identifies "
            "tech migration signals when teams mention moving from legacy CV tools. "
            "Target buyers at the exact moment they are scaling their ML infrastructure."
        ),
    },
    # Add these chunks to PRODUCT_KNOWLEDGE:
    {
        "title": "Use Case: Hardware/Robotics Companies",
        "content": "Robotics and hardware companies sell to data centers, "
        "manufacturers, and facilities operators — all of whom have very "
        "specific buying triggers (infrastructure expansion, facility "
        "upgrades, new deployments). SalesNeuron scans live web signals "
        "to find these exact moments..."
    },
    {
        "title": "Use Case: Blockchain/Web3 Consulting",
        "content": "Web3 consulting firms win clients when companies are "
        "actively exploring blockchain adoption. SalesNeuron detects "
        "tech migration signals — companies mentioning legacy system "
        "limitations or digital transformation initiatives..."
    }
]


class KnowledgeBase:
    """
    ChromaDB-backed product knowledge base.
    Stores product info as vector embeddings for semantic search.
    The Personalizer Agent queries this to find relevant features
    for each prospect's specific pain points.
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._ready = False
        self._feedback_ready = False

    async def _ensure_feedback_db(self):
        """
        Create the chunk feedback table on first use. Tracks how often
        each knowledge chunk gets used in a sent email, and how often
        that led to a reply — data ColdEmail.knowledge_chunks_used and
        SequenceManager's reply detection were already collecting, but
        nothing was reading it back to actually improve retrieval.
        """
        if self._feedback_ready:
            return
        FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(FEEDBACK_DB_PATH) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS chunk_feedback (
                    title           TEXT PRIMARY KEY,
                    times_used      INTEGER NOT NULL DEFAULT 0,
                    times_replied   INTEGER NOT NULL DEFAULT 0,
                    updated_at      TEXT NOT NULL
                );
            """)
            await db.commit()
        self._feedback_ready = True

    async def record_feedback(self, chunk_titles: list[str], event_type: str):
        """
        Record that one or more knowledge chunks were used in an email,
        and (separately) whether that email got a reply.

        event_type: 'sent' — increments times_used for each chunk
                    'replied' — increments times_replied for each chunk
                    (call 'sent' when the email goes out, then 'replied'
                    later if/when SequenceManager.check_replies() finds
                    a reply on that sequence)
        """
        if not chunk_titles:
            return
        await self._ensure_feedback_db()
        now = datetime.now().isoformat()

        async with aiosqlite.connect(FEEDBACK_DB_PATH) as db:
            for title in chunk_titles:
                if event_type == "sent":
                    await db.execute(
                        """INSERT INTO chunk_feedback (title, times_used, times_replied, updated_at)
                           VALUES (?, 1, 0, ?)
                           ON CONFLICT(title) DO UPDATE SET
                               times_used = times_used + 1,
                               updated_at = excluded.updated_at""",
                        (title, now),
                    )
                elif event_type == "replied":
                    await db.execute(
                        """INSERT INTO chunk_feedback (title, times_used, times_replied, updated_at)
                           VALUES (?, 0, 1, ?)
                           ON CONFLICT(title) DO UPDATE SET
                               times_replied = times_replied + 1,
                               updated_at = excluded.updated_at""",
                        (title, now),
                    )
            await db.commit()

        logger.debug(f"📚 Feedback recorded ({event_type}): {chunk_titles}")

    async def _get_reply_rates(self) -> dict:
        """Return {title: reply_rate} for all chunks with recorded feedback."""
        await self._ensure_feedback_db()
        async with aiosqlite.connect(FEEDBACK_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT title, times_used, times_replied FROM chunk_feedback WHERE times_used > 0"
            ) as cur:
                rows = await cur.fetchall()
        return {
            r["title"]: r["times_replied"] / r["times_used"]
            for r in rows
        }

    async def init(self):
        """Initialize ChromaDB and load existing collection if available."""
        import chromadb
        from chromadb.config import Settings

        Path(DB_PATH).mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(
            None,
            lambda: chromadb.PersistentClient(path=DB_PATH)
        )

        # Get or create collection
        self._collection = await loop.run_in_executor(
            None,
            lambda: self._client.get_or_create_collection(
                name="product_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
        )

        count = await loop.run_in_executor(None, self._collection.count)
        if count > 0:
            logger.info(f"📚 Knowledge base loaded — {count} chunks in ChromaDB")
        else:
            logger.info("📚 Knowledge base empty — run build() to populate")

        self._ready = True

    async def build(self):
        """
        Index all PRODUCT_KNOWLEDGE into ChromaDB.
        Run this once after editing PRODUCT_KNOWLEDGE above.
        Safe to re-run — clears and rebuilds.
        """
        if not self._ready:
            await self.init()

        loop = asyncio.get_running_loop()

        # Clear existing data
        existing = await loop.run_in_executor(None, self._collection.count)
        if existing > 0:
            await loop.run_in_executor(
                None,
                lambda: self._collection.delete(
                    ids=[item["id"] for item in PRODUCT_KNOWLEDGE]
                )
            )

        # Add all documents
        documents = [item["content"] for item in PRODUCT_KNOWLEDGE]
        metadatas = [
            {"category": item["category"], "title": item["title"]}
            for item in PRODUCT_KNOWLEDGE
        ]
        ids = [item["id"] for item in PRODUCT_KNOWLEDGE]

        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        )

        count = await loop.run_in_executor(None, self._collection.count)
        logger.info(f"📚 Knowledge base built — {count} chunks indexed in ChromaDB")
        logger.info(f"   Location: {DB_PATH}")

    async def search(
        self,
        query: str,
        n_results: int = 4,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over the knowledge base.
        Returns most relevant chunks for a given prospect pain point.

        Args:
            query: the prospect's pain point or buying signal description
            n_results: how many chunks to return
            category: filter by 'feature', 'pain_point', 'use_case', etc.
        """
        if not self._ready:
            await self.init()

        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, self._collection.count)

        where = {"category": category} if category else None

        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_texts=[query],
                n_results=min(n_results * 2, count),  # over-fetch, re-rank below
                where=where,
            )
        )

        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "content": doc,
                    "title": results["metadatas"][0][i]["title"],
                    "category": results["metadatas"][0][i]["category"],
                    "distance": results["distances"][0][i] if "distances" in results else 0,
                })

        # Re-rank using reply-rate feedback, blended with vector distance.
        # A chunk with a proven track record of leading to replies should
        # outrank one that's merely semantically closer but has never
        # actually worked — this is the data ColdEmail.knowledge_chunks_used
        # + SequenceManager's reply detection were already collecting,
        # just never fed back into retrieval until now.
        reply_rates = await self._get_reply_rates()
        if reply_rates and chunks:
            max_distance = max((c["distance"] for c in chunks), default=1) or 1

            def _rank_score(chunk):
                # Lower distance = better match (normalize to 0-1, invert)
                distance_score = 1 - (chunk["distance"] / max_distance)
                reply_boost = reply_rates.get(chunk["title"], 0.0)
                # Blend: relevance still dominates, reply-rate breaks ties
                # and can promote a proven chunk over a marginally closer
                # but unproven one.
                return (distance_score * 0.75) + (reply_boost * 0.25)

            chunks.sort(key=_rank_score, reverse=True)

        return chunks[:n_results]

    async def search_for_signals(self, buying_signals: list) -> list[dict]:
        """
        Given a list of BuyingSignal objects, find relevant product knowledge
        for each signal. Returns combined relevant chunks.
        """
        if not buying_signals:
            return await self.search("general sales outreach B2B", n_results=3)

        all_chunks = []
        seen_ids = set()

        for signal in buying_signals[:3]:  # top 3 signals
            description = (
                signal.description
                if hasattr(signal, "description")
                else signal.get("description", "")
            )
            signal_type = (
                signal.signal_type
                if hasattr(signal, "signal_type")
                else signal.get("signal_type", "")
            )

            query = f"{signal_type}: {description}"
            chunks = await self.search(query, n_results=2)

            for chunk in chunks:
                if chunk["title"] not in seen_ids:
                    seen_ids.add(chunk["title"])
                    all_chunks.append(chunk)

        return all_chunks[:5]  # max 5 chunks total

    def is_ready(self) -> bool:
        return self._ready


# Singleton
kb = KnowledgeBase()