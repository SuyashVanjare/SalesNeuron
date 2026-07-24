"""
SalesNeuron — Core Data Models
All agent outputs are typed Pydantic models so downstream agents
(Personalizer, Sequence Manager) can rely on a stable schema.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class FundingRound(BaseModel):
    round_type: str = Field(description="e.g. Seed, Series A, Series B")
    amount_usd: Optional[str] = Field(None, description="e.g. '$5M', '$2.3M'")
    date: Optional[str] = Field(None, description="e.g. 'March 2024'")
    investors: list[str] = Field(default_factory=list)


class KeyPerson(BaseModel):
    name: str
    title: str
    linkedin_url: Optional[str] = None


class BuyingSignal(BaseModel):
    signal_type: str = Field(
        description="One of: hiring_surge, skill_demand, recent_funding, tech_migration, "
                    "product_launch, leadership_change, geographic_expansion, "
                    "pain_point_mention, rapid_growth"
    )
    description: str = Field(description="Specific, factual detail of the signal")
    strength: str = Field(description="high / medium / low")
    source_url: Optional[str] = None


class TechStack(BaseModel):
    category: str = Field(description="e.g. CRM, Analytics, Cloud, Communication")
    tools: list[str] = Field(description="e.g. ['Salesforce', 'Tableau']")


class ProspectProfile(BaseModel):
    """
    The structured output of the Researcher Agent.
    This is what gets passed to the Personalizer Agent.
    """
    # Identity
    company_name: str
    website: str
    industry: str
    company_size: Optional[str] = Field(None, description="e.g. '50-200 employees'")
    founded_year: Optional[str] = None
    headquarters: Optional[str] = None
    description: str = Field(description="2-3 sentence company summary")

    # People
    key_people: list[KeyPerson] = Field(default_factory=list)

    # Recent news & signals
    recent_news: list[str] = Field(
        default_factory=list,
        description="Top 3-5 recent events (funding, launches, hires, expansions)"
    )
    funding_history: list[FundingRound] = Field(default_factory=list)
    buying_signals: list[BuyingSignal] = Field(default_factory=list)

    # Tech & product
    tech_stack: list[TechStack] = Field(default_factory=list)
    products_services: list[str] = Field(default_factory=list)

    # Hiring intelligence
    open_job_roles: list[dict] = Field(
        default_factory=list,
        description=(
            "Structured job roles currently open. Each dict has: "
            "title, team, location, type, skills (list), description_snippet"
        )
    )
    hiring_velocity: Optional[str] = Field(
        None,
        description="Summary of hiring activity e.g. '15 open roles, heavy ML focus'"
    )
    hiring_locations: list[str] = Field(
        default_factory=list,
        description="Unique locations where the company is actively hiring"
    )

    # Research metadata
    pages_scraped: list[str] = Field(default_factory=list)
    research_confidence: str = Field(
        description="high / medium / low — based on data completeness"
    )
    researched_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    raw_text_summary: str = Field(
        description="Free-text synthesis of all scraped content, used by Personalizer"
    )