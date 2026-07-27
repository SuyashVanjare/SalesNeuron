"""
SalesNeuron — Email Models
Typed output of the Personalizer Agent.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ColdEmail(BaseModel):
    """
    The structured output of the Personalizer Agent.
    Passed to the Sequence Manager for sending + tracking.
    """
    # Target
    company_name: str
    website: str
    contact_name: Optional[str] = None
    contact_title: Optional[str] = None

    # Email content
    subject: str
    body: str

    # Personalization metadata
    buying_signal_used: str = Field(
        description="The specific buying signal this email hooks on"
    )
    product_angle: str = Field(
        description="Which product feature/use case was matched to this prospect"
    )
    personalization_score: str = Field(
        description="high/medium/low — how personalized is this email?"
    )

    # RAG metadata
    knowledge_chunks_used: list[str] = Field(
        default_factory=list,
        description="Titles of knowledge base chunks used to write this email"
    )

    # Tracking
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    status: str = "draft"  # draft / sent / replied / bounced