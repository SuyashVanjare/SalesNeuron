"""
Manually seed correct Internshala flows into the graph store.
Run: python data/seed_internshala.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from dotenv import load_dotenv
load_dotenv()

from knowledge.graph_store import graph_store
from knowledge.models import SiteGraph, PageNode, NavigationEdge, NavigationFlow, InteractiveElement

async def main():
    await graph_store.init()

    graph = SiteGraph(
        domain="internshala.com",
        base_url="https://internshala.com",
        site_type="job_board",
        description="India's largest internship and job platform.",
        pages=[
            PageNode(
                url="https://internshala.com/internships",
                url_pattern="https://internshala\\.com/internships.*",
                page_type="search_results",
                title="Internships",
                purpose="Browse and search internships",
                elements=[
                    InteractiveElement(
                        element_type="input",
                        selector="input#search-field",
                        label="Search internships",
                        purpose="Type internship keyword to search"
                    ),
                    InteractiveElement(
                        element_type="button",
                        selector="button[type='submit']",
                        label="Search",
                        purpose="Submit search query"
                    ),
                ]
            ),
            PageNode(
                url="https://internshala.com/jobs",
                url_pattern="https://internshala\\.com/jobs.*",
                page_type="search_results",
                title="Jobs",
                purpose="Browse and search jobs",
                elements=[
                    InteractiveElement(
                        element_type="input",
                        selector="input#search-field",
                        label="Search jobs",
                        purpose="Type job keyword to search"
                    ),
                ]
            ),
        ],
        flows=[
            NavigationFlow(
                flow_name="search_internships",
                description="Search for internships by keyword (pass a hyphenated slug, e.g. 'python' or 'data-science')",
                variables=["query"],
                success_indicator=".internship_list_container",
                steps=[
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com",
                        to_url_pattern="https://internshala\\.com/internships/.*-internship.*",
                        action_type="navigate",
                        selector="https://internshala.com/internships/{query}-internship/",
                        description="Navigate directly to the category search results page",
                        wait_for=".internship_list_container",
                        confidence=0.9,
                    ),
                ]
            ),
            NavigationFlow(
                flow_name="search_jobs",
                description="Search for jobs by keyword (pass a hyphenated slug, e.g. 'python' or 'sales')",
                variables=["query"],
                success_indicator=".jobs-container",
                steps=[
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com",
                        to_url_pattern="https://internshala\\.com/jobs/.*-jobs.*",
                        action_type="navigate",
                        selector="https://internshala.com/jobs/{query}-jobs/",
                        description="Navigate directly to the category jobs results page",
                        wait_for=".jobs-container",
                        confidence=0.9,
                    ),
                ]
            ),
            # ── Login flow — reference example for the generic auth layer ──
            # Selectors below are illustrative; if Internshala's real login
            # form differs, run the explorer against /login and let it
            # discover the actual selectors, or correct these by hand.
            NavigationFlow(
                flow_name="login",
                description="Log in to Internshala with email and password",
                variables=["email", "password"],
                is_auth_flow=True,
                auth_flow_type="login",
                success_indicator="#header_middle_container",
                steps=[
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com.*",
                        to_url_pattern="https://internshala\\.com/login.*",
                        action_type="navigate",
                        selector="https://internshala.com/login/student",
                        description="Go to student login page",
                        confidence=0.9,
                    ),
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com/login.*",
                        to_url_pattern="https://internshala\\.com/login.*",
                        action_type="type",
                        selector="#email",
                        input_value="{email}",
                        description="Type email address",
                        confidence=0.85,
                    ),
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com/login.*",
                        to_url_pattern="https://internshala\\.com/login.*",
                        action_type="type",
                        selector="#password",
                        input_value="{password}",
                        description="Type password",
                        confidence=0.85,
                    ),
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com/login.*",
                        to_url_pattern="https://internshala\\.com/student/dashboard.*",
                        action_type="click",
                        selector="#login_submit",
                        description="Click login submit button",
                        wait_for="#header_middle_container",
                        confidence=0.85,
                    ),
                ],
            ),
            NavigationFlow(
                flow_name="search_python_internships",
                description="Search specifically for Python internships",
                variables=[],
                success_indicator=".internship_list_container",
                steps=[
                    NavigationEdge(
                        from_url_pattern="https://internshala\\.com",
                        to_url_pattern="https://internshala\\.com/internships/.*",
                        action_type="navigate",
                        selector="https://internshala.com/internships/python-django-internship",
                        description="Navigate to Python internships page",
                        wait_for=".internship_list_container",
                        confidence=0.95,
                    ),
                ]
            ),
        ],
        pages_explored=2,
    )

    # Delete old graph and save new one
    await graph_store.delete("internshala.com")
    await graph_store.save(graph)
    print("[SUCCESS] Internshala flows seeded successfully")

    flows = await graph_store.list_flows("internshala.com")
    for f in flows:
        print(f"  -> {f['flow_name']}: {f['description']}")

    print(
        "\nTo use the login flow, store real credentials with:\n"
        "  python run_navigator.py --save-credentials internshala.com\n"
    )

asyncio.run(main())