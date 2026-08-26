"""
Cognex reference backend — data model.

This is the same shape as the client-side prototype (PERSONAS / DECISIONS / GOALS),
ported to Python so the permission and retrieval logic has something real to run
against. In production this file becomes: PERSONAS -> your identity provider (Okta /
Azure AD / Workspace, synced via SCIM), DECISIONS and GOALS -> rows in a real graph
store (Neo4j, or a relational table with an adjacency/edge table) plus a vector index
over the free-text fields for semantic search.

Nothing here talks to Claude. This module is intentionally dumb: it's the "memory"
half of the architecture, decoupled from the "model" half in agent.py.
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Clearance levels. Higher number = broader access. In production this would
# be a real role/attribute system (department + seniority + explicit grants),
# not a single ladder — but the ladder keeps this reference implementation
# legible and maps 1:1 onto the front-end prototype.
# ---------------------------------------------------------------------------
LEVEL = {
    "intern": 1,
    "employee": 2,
    "manager": 3,
    "director": 4,
    "exec": 5,
    "ceo": 6,
}
LEVEL_LABEL = {
    1: "Company-wide",
    2: "Employee+",
    3: "Manager+",
    4: "Director+",
    5: "Executive",
    6: "CEO only",
}


@dataclass
class Persona:
    id: str
    name: str
    title: str
    level: int
    dept: str


@dataclass
class Decision:
    id: str
    title: str
    decided: str
    owner: str
    visibility: int          # minimum clearance level required to see the RAW record
    derived_level: int       # minimum clearance level required to see the DERIVED summary
    why: list[str]
    alternatives: list[str]
    assumptions: list[str]
    risks: list[str]
    review: str
    result: str
    tags: list[str]
    derived: str              # pre-approved, lower-clearance-safe summary (see derived_context.py)
    contributor: Optional[str] = None
    via_story: bool = False


@dataclass
class Goal:
    id: str
    depth: int
    kicker: str
    title: str
    parent: Optional[str]
    dept: Optional[str]
    owner: Optional[str] = None  # persona id, for individual-work leaf goals


PERSONAS: dict[str, Persona] = {
    p.id: p for p in [
        Persona("ceo", "Elena Vasquez", "CEO", LEVEL["ceo"], "Executive"),
        Persona("cfo", "Marcus Chen", "CFO", LEVEL["exec"], "Finance"),
        Persona("director", "Priya Nair", "Director of Product", LEVEL["director"], "Product"),
        Persona("manager", "Sam Okafor", "Sales Manager", LEVEL["manager"], "Sales"),
        Persona("employee", "Jordan Lee", "Marketing Associate", LEVEL["employee"], "Marketing"),
        Persona("intern", "Riya Patel", "Marketing Intern", LEVEL["intern"], "Marketing"),
    ]
}

DECISIONS: list[Decision] = [
    Decision(
        id="d-launch-x",
        title="Launch Product X in Q4",
        decided="2026-06-02",
        owner="CEO + Product Director",
        visibility=LEVEL["employee"],
        derived_level=LEVEL["intern"],
        why=[
            "Customer demand is increasing — three major enterprise customers requested this product directly.",
            "Manufacturing capacity is expected to increase in Q4.",
            "Projected gross margin of 35% is attractive relative to the rest of the portfolio.",
        ],
        alternatives=["Launch in Q3", "Delay until Q1 2027", "Launch in another market first"],
        assumptions=["Manufacturing capacity increase lands on schedule",
                     "Demand signal from the three customers converts to committed orders"],
        risks=["Capacity slip pushes launch past the holiday buying window",
               "Margin compresses if input costs rise before Q4"],
        review="Q1 2027",
        result="TBD",
        tags=["product x", "launch", "q4", "margin", "manufacturing capacity", "customer demand"],
        derived="Product X is launching in Q4, driven by customer demand and an expected manufacturing capacity increase.",
    ),
    Decision(
        id="d-india-vietnam",
        title="Manufacture Product X in India instead of Vietnam",
        decided="2026-03-14",
        owner="COO + Supply Chain Lead",
        visibility=LEVEL["director"],
        derived_level=LEVEL["employee"],
        why=[
            "Manufacturing costs are 14% lower in India than the Vietnam alternative.",
            "Supplier reliability scored higher in India across the last two sourcing audits.",
        ],
        alternatives=["Manufacture in Vietnam", "Split production across both"],
        assumptions=["India supplier maintains current reliability under higher volume"],
        risks=["Single-country concentration risk if India supply is disrupted"],
        review="Scheduled for 2028 (two years post-decision) — not yet completed",
        result="In effect; no formal review has occurred",
        tags=["india", "vietnam", "manufacturing", "supplier", "cost", "sourcing"],
        derived="We manufacture in India rather than Vietnam for cost and supplier-reliability reasons.",
    ),
    Decision(
        id="d-no-germany",
        title="Do not enter the Germany market",
        decided="2025-11-20",
        owner="CEO + Board",
        visibility=LEVEL["exec"],
        derived_level=LEVEL["manager"],
        why=[
            "Regulatory complexity in Germany was assessed as high relative to expected near-term return.",
            "Market was judged saturated by two entrenched incumbents.",
            "UAE expansion offered a better return on the same investment window.",
        ],
        alternatives=["Enter Germany directly in 2025", "Enter via a local distributor partnership"],
        assumptions=["UAE opportunity remains available on a similar timeline"],
        risks=["A competitor establishes an uncontested position in Germany"],
        review="2028",
        result="Resources reallocated to UAE market entry",
        tags=["germany", "market entry", "expansion", "uae", "regulatory"],
        derived="Germany market entry was evaluated and declined in favor of the UAE opportunity.",
    ),
    Decision(
        id="d-acquire-northwind",
        title="Acquire Northwind Logistics",
        decided="2026-07-08",
        owner="CEO",
        visibility=LEVEL["ceo"],
        derived_level=LEVEL["intern"],
        why=[
            "Acquisition accelerates last-mile fulfillment capability ahead of the UAE launch.",
            "Northwind's Gulf-region warehouse network removes an 18-month build-out.",
        ],
        alternatives=["Build fulfillment capability in-house", "Partner with a regional 3PL instead of acquiring"],
        assumptions=["Northwind's warehouse leases transfer cleanly", "Integration completes before the UAE launch date"],
        risks=["Deal does not close in time for the launch window", "Integration runs slower than planned"],
        review="Q3 2027",
        result="Approved — deal signed 2026-08-01",
        tags=["acquisition", "northwind", "logistics", "fulfillment", "uae", "confidential"],
        derived=(
            "The company is expanding its logistics and fulfillment capabilities. As a result, "
            "Product and Sales goals tied to the UAE launch now assume faster fulfillment timelines — "
            "priorities for the next two quarters have been updated accordingly."
        ),
    ),
]

GOALS: list[Goal] = [
    Goal("g-uae", 0, "Company goal", "Enter the UAE market in 2027", None, None),
    Goal("g-marketing", 1, "Marketing goal", "Build brand awareness among target Gulf-region customers", "g-uae", "Marketing"),
    Goal("g-sales", 1, "Sales goal", "Develop 100 qualified enterprise leads in UAE by Q2 2027", "g-uae", "Sales"),
    Goal("g-product", 1, "Product goal", "Localize Product X for Gulf compliance & Arabic UI", "g-uae", "Product"),
    Goal("g-brand-team", 2, "Team goal", "Produce localized campaign assets for the UAE launch", "g-marketing", "Marketing"),
    Goal("g-sales-team", 2, "Team goal", "Outbound prospecting into UAE logistics & retail verticals", "g-sales", "Sales"),
    Goal("g-product-team", 2, "Team goal", "Ship Arabic localization by Q4 2026", "g-product", "Product"),
    Goal("g-jordan", 3, "Individual work · Jordan Lee", "Create UAE-specific product one-pagers and social assets", "g-brand-team", "Marketing", owner="employee"),
    Goal("g-riya", 3, "Individual work · Riya Patel", "Localize social captions & translate the launch FAQ into Arabic", "g-brand-team", "Marketing", owner="intern"),
    Goal("g-sam", 3, "Individual work · Sam Okafor", "Build target account list for the UAE logistics vertical", "g-sales-team", "Sales", owner="manager"),
]


def goal_by_id(goal_id: str) -> Optional[Goal]:
    return next((g for g in GOALS if g.id == goal_id), None)


def goal_chain(goal: Goal) -> list[Goal]:
    """Walk a goal up to the company-level root. Same logic as goalChain() in the JS prototype."""
    chain = [goal]
    cur = goal
    while cur.parent:
        cur = goal_by_id(cur.parent)
        if not cur:
            break
        chain.insert(0, cur)
    return chain
