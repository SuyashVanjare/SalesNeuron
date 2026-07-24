"""
SalesNeuron — Site Graph Store (SQLite)
=========================================
Saves and retrieves SiteGraph objects from SQLite.
Works alongside core/memory.py (which stores prospect profiles).

Tables:
  site_graphs    — one row per domain, full graph as JSON
  site_flows     — normalized flows for fast "what can I do on X?" queries

Usage:
  from knowledge.graph_store import graph_store

  await graph_store.init()
  await graph_store.save(graph)

  graph = await graph_store.get("amazon.com")
  flow  = await graph_store.get_flow("amazon.com", "search_product")
  all   = await graph_store.all_sites()
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiosqlite

from knowledge.models import SiteGraph, NavigationFlow

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "data/salesneuron.db"))
GRAPH_CACHE_DAYS = int(os.getenv("GRAPH_CACHE_DAYS", "30"))  # graphs stay fresh longer


class GraphStore:
    """
    Persistent store for site knowledge graphs.
    Shares the same SQLite DB as the prospect memory.
    """

    def __init__(self):
        self._db_path = DB_PATH
        self._ready = False

    async def init(self):
        """Create tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS site_graphs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain          TEXT UNIQUE NOT NULL,
                    base_url        TEXT NOT NULL,
                    site_type       TEXT,
                    description     TEXT,
                    pages_explored  INTEGER DEFAULT 0,
                    flows_count     INTEGER DEFAULT 0,
                    edges_count     INTEGER DEFAULT 0,
                    graph_json      TEXT NOT NULL,
                    learned_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS site_flows (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain          TEXT NOT NULL,
                    flow_name       TEXT NOT NULL,
                    description     TEXT,
                    variables       TEXT,
                    steps_count     INTEGER DEFAULT 0,
                    times_used      INTEGER DEFAULT 0,
                    success_rate    REAL DEFAULT 1.0,
                    flow_json       TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    UNIQUE(domain, flow_name),
                    FOREIGN KEY (domain) REFERENCES site_graphs(domain)
                );

                CREATE INDEX IF NOT EXISTS idx_graphs_domain
                    ON site_graphs(domain);
                CREATE INDEX IF NOT EXISTS idx_flows_domain
                    ON site_flows(domain);
                CREATE INDEX IF NOT EXISTS idx_flows_name
                    ON site_flows(flow_name);
            """)
            await db.commit()

        self._ready = True
        logger.info(f"🗄️  GraphStore initialized → {self._db_path.resolve()}")

    # ──────────────────────────────────────────────────────────────
    # READ
    # ──────────────────────────────────────────────────────────────

    async def get(self, domain: str) -> Optional[SiteGraph]:
        """
        Return the site graph for a domain if it exists and is fresh.
        Returns None if not found or stale.
        """
        self._ensure_ready()
        domain = self._normalize_domain(domain)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT graph_json, learned_at FROM site_graphs WHERE domain = ?",
                (domain,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            logger.info(f"🗺️  Graph MISS — {domain} not in DB")
            return None

        age = datetime.now() - datetime.fromisoformat(row["learned_at"])
        if age > timedelta(days=GRAPH_CACHE_DAYS):
            logger.info(f"🗺️  Graph STALE — {domain} is {age.days} days old, will re-learn")
            return None

        logger.info(f"🗺️  Graph HIT — {domain} (learned {age.days}d ago)")
        try:
            return SiteGraph(**json.loads(row["graph_json"]))
        except Exception as e:
            logger.warning(f"Failed to deserialize graph: {e}")
            return None

    async def get_flow(self, domain: str, flow_name: str) -> Optional[NavigationFlow]:
        """
        Get a specific named flow for a domain.
        This is what the Navigator Agent calls to execute a task.
        """
        self._ensure_ready()
        domain = self._normalize_domain(domain)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT flow_json FROM site_flows WHERE domain = ? AND flow_name = ?",
                (domain, flow_name),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        try:
            return NavigationFlow(**json.loads(row["flow_json"]))
        except Exception as e:
            logger.warning(f"Failed to deserialize flow: {e}")
            return None

    async def list_flows(self, domain: str) -> list[dict]:
        """
        List all known flows for a domain.
        Lets the agent ask: "what can I do on amazon.com?"
        """
        self._ensure_ready()
        domain = self._normalize_domain(domain)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT flow_name, description, variables, steps_count,
                          times_used, success_rate
                   FROM site_flows WHERE domain = ?
                   ORDER BY times_used DESC""",
                (domain,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [dict(r) for r in rows]

    async def all_sites(self) -> list[dict]:
        """Summary of all sites in the graph store."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT domain, base_url, site_type, description,
                          pages_explored, flows_count, edges_count, updated_at
                   FROM site_graphs ORDER BY updated_at DESC"""
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────────

    async def save(self, graph: SiteGraph):
        """Save a complete SiteGraph to the database."""
        self._ensure_ready()
        domain = self._normalize_domain(graph.domain)
        now = datetime.now().isoformat()
        graph_json = graph.model_dump_json()

        async with aiosqlite.connect(self._db_path) as db:
            # Upsert main graph
            await db.execute(
                """
                INSERT INTO site_graphs
                    (domain, base_url, site_type, description,
                     pages_explored, flows_count, edges_count,
                     graph_json, learned_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    site_type     = excluded.site_type,
                    description   = excluded.description,
                    pages_explored = excluded.pages_explored,
                    flows_count   = excluded.flows_count,
                    edges_count   = excluded.edges_count,
                    graph_json    = excluded.graph_json,
                    updated_at    = excluded.updated_at
                """,
                (
                    domain,
                    graph.base_url,
                    graph.site_type,
                    graph.description,
                    graph.pages_explored,
                    len(graph.flows),
                    len(graph.edges),
                    graph_json,
                    now,
                    now,
                ),
            )

            # Upsert each flow separately for fast lookup
            for flow in graph.flows:
                await db.execute(
                    """
                    INSERT INTO site_flows
                        (domain, flow_name, description, variables,
                         steps_count, flow_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(domain, flow_name) DO UPDATE SET
                        description  = excluded.description,
                        variables    = excluded.variables,
                        steps_count  = excluded.steps_count,
                        flow_json    = excluded.flow_json
                    """,
                    (
                        domain,
                        flow.flow_name,
                        flow.description,
                        json.dumps(flow.variables),
                        len(flow.steps),
                        flow.model_dump_json(),
                        now,
                    ),
                )

            await db.commit()

        logger.info(
            f"🗺️  Graph saved — {domain} | "
            f"{len(graph.pages)} pages | "
            f"{len(graph.flows)} flows | "
            f"{len(graph.edges)} edges"
        )

    async def record_flow_use(self, domain: str, flow_name: str, success: bool):
        """
        Called by the Navigator Agent after executing a flow.
        Updates usage count and success rate — makes the graph smarter over time.
        """
        self._ensure_ready()
        domain = self._normalize_domain(domain)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT times_used, success_rate FROM site_flows WHERE domain=? AND flow_name=?",
                (domain, flow_name),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                times_used = row[0] + 1
                # Exponential moving average for success rate
                new_rate = (row[1] * 0.9) + (1.0 if success else 0.0) * 0.1
                await db.execute(
                    "UPDATE site_flows SET times_used=?, success_rate=? WHERE domain=? AND flow_name=?",
                    (times_used, new_rate, domain, flow_name),
                )
                await db.commit()

    async def stats(self) -> dict:
        """Return graph store statistics."""
        self._ensure_ready()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM site_graphs") as c:
                total_sites = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM site_flows") as c:
                total_flows = (await c.fetchone())[0]
            async with db.execute("SELECT SUM(times_used) FROM site_flows") as c:
                total_uses = (await c.fetchone())[0] or 0

        return {
            "total_sites": total_sites,
            "total_flows": total_flows,
            "total_flow_executions": total_uses,
            "cache_days": GRAPH_CACHE_DAYS,
            "db_path": str(self._db_path.resolve()),
        }

    async def delete(self, domain: str):
        """Remove a site graph — forces re-learning next time."""
        self._ensure_ready()
        domain = self._normalize_domain(domain)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM site_graphs WHERE domain = ?", (domain,))
            await db.execute("DELETE FROM site_flows WHERE domain = ?", (domain,))
            await db.commit()
        logger.info(f"🗺️  Deleted graph for {domain}")

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _normalize_domain(self, domain: str) -> str:
        """Strip www. and trailing slashes for consistent keys."""
        domain = domain.replace("https://", "").replace("http://", "")
        domain = domain.replace("www.", "").rstrip("/")
        return domain.lower()

    def _ensure_ready(self):
        if not self._ready:
            raise RuntimeError(
                "GraphStore not initialized. Call `await graph_store.init()` first."
            )


# Singleton
graph_store = GraphStore()