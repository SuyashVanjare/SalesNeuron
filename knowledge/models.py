"""
SalesNeuron — Site Knowledge Graph Models
==========================================
A website is represented as a directed graph:

  Node  = a page (URL + what's on it + interactive elements)
  Edge  = an action that moves from one page to another
  Flow  = a named sequence of edges to accomplish a goal
           e.g. "search_product", "login", "fill_contact_form"

This is exactly how StableBrowse works internally —
they store site structure so agents never re-explore the same site.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class InteractiveElement(BaseModel):
    """One clickable / typeable / selectable element on a page."""
    element_type: str       # button, input, link, select, form
    selector: str           # CSS selector to find this element
    label: str              # human-readable label (button text, placeholder, aria-label)
    purpose: str            # what does clicking/typing here do?
    required: bool = False  # is this required to proceed?


class PageNode(BaseModel):
    """
    A single page in the site graph.
    Stores everything an agent needs to understand this page.
    """
    url: str
    url_pattern: str = Field(
        description="Regex pattern for this page type. "
                    "e.g. 'https://amazon.com/dp/.+' matches any product page"
    )
    page_type: str = Field(
        description="homepage / search_results / product_page / "
                    "cart / checkout / login / form / listing / article / other"
    )
    title: str
    purpose: str            # what is this page for?
    elements: list[InteractiveElement] = Field(default_factory=list)
    outgoing_urls: list[str] = Field(default_factory=list)  # pages this links to
    requires_auth: bool = False
    discovered_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class NavigationEdge(BaseModel):
    """
    An action that moves from one page to another.
    e.g. "click Add to Cart on product page → goes to cart page"
    """
    from_url_pattern: str   # source page pattern
    to_url_pattern: str     # destination page pattern
    action_type: str        # click / type / select / navigate / submit / hover
    selector: str           # CSS selector of the element to interact with
    input_value: Optional[str] = None   # for type actions — use {variable} for dynamic
    description: str        # human readable: "click the search button"
    wait_for: Optional[str] = None      # selector to wait for after action
    confidence: float = 1.0            # how reliable is this edge? (0-1)


class NavigationFlow(BaseModel):
    """
    A named multi-step workflow to accomplish a goal on this site.
    e.g. "search_and_find_product" or "fill_contact_form"
    """
    flow_name: str          # machine name: search_product, add_to_cart, login
    description: str        # human readable goal
    steps: list[NavigationEdge]
    variables: list[str] = Field(
        default_factory=list,
        description="Variables this flow needs e.g. ['query', 'email', 'password']"
    )
    success_indicator: Optional[str] = Field(
        None,
        description="CSS selector that appears when flow succeeds"
    )
    times_used: int = 0
    success_rate: float = 1.0

    # ── Auth marking (generic — works for any site) ─────────────────
    is_auth_flow: bool = Field(
        False,
        description="True if this flow's purpose IS authentication itself "
                    "(logging in or signing up), as opposed to a flow that "
                    "merely requires the user to already be authenticated."
    )
    auth_flow_type: Optional[str] = Field(
        None,
        description="'login' or 'signup' — only set when is_auth_flow=True"
    )
    requires_auth: bool = Field(
        False,
        description="True if this flow needs an authenticated session to "
                    "succeed. Navigator will auto-login first if credentials "
                    "are stored for this domain."
    )


class SiteGraph(BaseModel):
    """
    Complete knowledge graph for one website.
    Stored in SQLite, loaded instantly when agent needs to navigate the site.
    """
    domain: str                         # e.g. "amazon.com"
    base_url: str                       # e.g. "https://amazon.com"
    site_type: str = Field(
        description="ecommerce / saas / news / social / corporate / "
                    "job_board / directory / other"
    )
    description: str                    # what is this site for?

    pages: list[PageNode] = Field(default_factory=list)
    edges: list[NavigationEdge] = Field(default_factory=list)
    flows: list[NavigationFlow] = Field(default_factory=list)

    pages_explored: int = 0
    learned_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    explorer_version: str = "1.0"