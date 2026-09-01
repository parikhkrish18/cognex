"""
Engine/session setup for the real persistence layer (see models.py), plus a
one-time seed of the "Cognex Labs" demo company so a fresh database behaves
exactly like the old hardcoded JS seed did — same 6 personas, same 5
decisions, same 10 goals, byte-for-byte ported from the frontend's
`seedCognexLabs()`. This only runs if the `companies` table is empty, so it
never overwrites real data on a redeploy.

DATABASE_URL is what Railway's DATABASE_URL variable is set to point at the
Postgres service (see the persistence build-log entry for how that's wired).
If it's not set at all — a local dev shell, or a redeploy that hasn't picked
up the variable yet — this falls back to a local SQLite file, so the rest of
the app (and the offline test suite) keeps working rather than hard-crashing
on import. That fallback is NOT a substitute for real Postgres in
production — it's the same "don't break the demo if the real backend piece
isn't configured yet" philosophy already used everywhere else in this app
(Ask Cognex, Complete the Story, Vantage all degrade gracefully instead of
hard-failing), applied to persistence.
"""

import os

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from models import Base, Company, Decision, Goal, Persona

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cognex_persistence.db")
# Railway (and some other hosts) hand out `postgres://`, but SQLAlchemy 2.x
# wants the explicit `postgresql://` (or `postgresql+psycopg://`) scheme.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Fail fast instead of hanging forever if Postgres is unreachable or still
# initializing. Without this, a misconfigured or momentarily-down Postgres
# makes psycopg's TCP connect block with NO timeout — the FastAPI startup
# hook (see server.py) then hangs on "Waiting for application startup." with
# no further log output and no exception, which looks exactly like a silent
# crash/restart loop from the outside. This is what actually happened on
# 2026-08-27 (see the build log entry) — the fix is this timeout, not just
# the one-off infra repair, so the same misconfiguration class can't cause
# an indefinite hang again even if it recurs.
_connect_args = {"connect_timeout": 8} if DATABASE_URL.startswith("postgresql") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session() -> Session:
    return SessionLocal()


AVATAR_COLORS = [
    {"bg": "#e8f0fe", "fg": "#0071e3"},
    {"bg": "#e6f9ec", "fg": "#1f8a4c"},
    {"bg": "#fff1e0", "fg": "#b45c09"},
    {"bg": "#f5e9fc", "fg": "#7a2fb0"},
    {"bg": "#ffe9f0", "fg": "#c2185b"},
    {"bg": "#e6f9fb", "fg": "#0f7a8c"},
]
LEVEL = {"intern": 1, "employee": 2, "manager": 3, "director": 4, "exec": 5, "ceo": 6}


def _initials_of(name: str) -> str:
    return "".join(part[0] for part in name.split() if part)[:2].upper()


def init_db():
    Base.metadata.create_all(engine)
    _migrate_missing_columns()
    _seed_cognex_labs_if_empty()


def _sql_default_literal(column):
    """Turn a mapped_column's Python-side `default=` into a SQL literal we
    can put in an ADD COLUMN ... DEFAULT clause, or None if it isn't a
    plain scalar (a callable/JSON default can't be expressed this way --
    those columns get added nullable, with existing rows left NULL)."""
    default = column.default
    if default is None or not getattr(default, "is_scalar", False):
        return None
    value = default.arg
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return None


def _migrate_missing_columns():
    """create_all() only creates tables that don't exist yet -- it never
    adds a new column to a table that's already there. That's exactly what
    happened on 2026-09-01: the Brain Board redesign added `domain` and
    `parent_id` to the Decision model, the code deployed cleanly, but the
    live `decisions` table (already existing from before that change)
    never picked up the new columns. Every company-load after that --
    including the one login itself does right after checking the password
    -- started throwing `UndefinedColumn: decisions.domain`, which surfaced
    to users as a login/"couldn't load that company" failure that had
    nothing to do with their password or being offline.

    This is the fix, and the guard against the same class of bug next time
    a column gets added to an existing table: on every startup, diff each
    model's columns against what the live table actually has, and add
    whatever's missing via a plain idempotent `ADD COLUMN IF NOT EXISTS`.
    Where the model has a simple scalar Python default (e.g. domain's
    "other"), that default is included in the ALTER so existing rows are
    backfilled immediately instead of coming back NULL — matching what the
    build log already assumed would happen. Postgres-only (`IF NOT EXISTS`
    on ADD COLUMN is a Postgres extension); SQLite dev/test databases are
    always created fresh via create_all() with every current column
    already in place, so they never hit this gap."""
    if not DATABASE_URL.startswith("postgresql"):
        return
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue  # brand new table -- create_all() already built it with every column
            existing_cols = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_cols:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                default_literal = _sql_default_literal(column)
                clause = f' ADD COLUMN IF NOT EXISTS "{column.name}" {col_type}'
                if default_literal is not None:
                    clause += f" DEFAULT {default_literal}"
                    if not column.nullable:
                        clause += " NOT NULL"
                conn.execute(text(f'ALTER TABLE "{table.name}"' + clause))
                print(f"[migrate] added missing column {table.name}.{column.name}", flush=True)


def _seed_cognex_labs_if_empty():
    with get_session() as db:
        if db.execute(select(Company.id)).first() is not None:
            return  # already has data — never overwrite on redeploy

        db.add(Company(id="cognex-labs", name="Cognex Labs", industry="Enterprise AI", seeded=True))

        persona_rows = [
            ("ceo", "Elena Vasquez", "CEO", LEVEL["ceo"], "Executive", "elena@cognexlabs.com", None, True),
            ("cfo", "Marcus Chen", "CFO", LEVEL["exec"], "Finance", "marcus@cognexlabs.com", "ceo", False),
            ("director", "Priya Nair", "Director of Product", LEVEL["director"], "Product", "priya@cognexlabs.com", "ceo", False),
            ("manager", "Sam Okafor", "Sales Manager", LEVEL["manager"], "Sales", "sam@cognexlabs.com", "ceo", False),
            ("employee", "Jordan Lee", "Marketing Associate", LEVEL["employee"], "Marketing", "jordan@cognexlabs.com", "ceo", False),
            ("intern", "Riya Patel", "Marketing Intern", LEVEL["intern"], "Marketing", "riya@cognexlabs.com", "employee", False),
        ]
        for i, (pid, name, title, level, dept, email, reports_to, is_admin) in enumerate(persona_rows):
            color = AVATAR_COLORS[i % len(AVATAR_COLORS)]
            db.add(Persona(
                company_id="cognex-labs", id=pid, name=name, title=title, level=level, dept=dept,
                email=email, reports_to=reports_to, is_org_admin=is_admin,
                initials=_initials_of(name), bg=color["bg"], fg=color["fg"],
            ))

        decisions = [
            dict(
                id="d-launch-x", title="Launch Product X in Q4", decided="2026-06-02", owner="CEO + Product Director",
                visibility=LEVEL["employee"], derived_level=LEVEL["intern"],
                why=[
                    "Customer demand is increasing — three major enterprise customers requested this product directly.",
                    "Manufacturing capacity is expected to increase in Q4.",
                    "Projected gross margin of 35% is attractive relative to the rest of the portfolio.",
                ],
                alternatives=["Launch in Q3", "Delay until Q1 2027", "Launch in another market first"],
                assumptions=["Manufacturing capacity increase lands on schedule", "Demand signal from the three customers converts to committed orders"],
                risks=["Capacity slip pushes launch past the holiday buying window", "Margin compresses if input costs rise before Q4"],
                review="Q1 2027", result="TBD",
                tags=["product x", "launch", "q4", "margin", "manufacturing capacity", "customer demand"],
                derived="Product X is launching in Q4, driven by customer demand and an expected manufacturing capacity increase.",
            ),
            dict(
                id="d-india-vietnam", title="Manufacture Product X in India instead of Vietnam", decided="2026-03-14", owner="COO + Supply Chain Lead",
                visibility=LEVEL["director"], derived_level=LEVEL["employee"],
                why=["Manufacturing costs are 14% lower in India than the Vietnam alternative.", "Supplier reliability scored higher in India across the last two sourcing audits."],
                alternatives=["Manufacture in Vietnam", "Split production across both"],
                assumptions=["India supplier maintains current reliability under higher volume"],
                risks=["Single-country concentration risk if India supply is disrupted"],
                review="Scheduled for 2028 (two years post-decision) — not yet completed", result="In effect; no formal review has occurred",
                tags=["india", "vietnam", "manufacturing", "supplier", "cost", "sourcing"],
                derived="We manufacture in India rather than Vietnam for cost and supplier-reliability reasons.",
            ),
            dict(
                id="d-no-germany", title="Do not enter the Germany market", decided="2025-11-20", owner="CEO + Board",
                visibility=LEVEL["exec"], derived_level=LEVEL["manager"],
                why=[
                    "Regulatory complexity in Germany was assessed as high relative to expected near-term return.",
                    "Market was judged saturated by two entrenched incumbents.",
                    "UAE expansion offered a better return on the same investment window.",
                ],
                alternatives=["Enter Germany directly in 2025", "Enter via a local distributor partnership"],
                assumptions=["UAE opportunity remains available on a similar timeline"],
                risks=["A competitor establishes an uncontested position in Germany"],
                review="2028", result="Resources reallocated to UAE market entry",
                tags=["germany", "market entry", "expansion", "uae", "regulatory"],
                derived="Germany market entry was evaluated and declined in favor of the UAE opportunity.",
            ),
            dict(
                id="d-acquire-northwind", title="Acquire Northwind Logistics", decided="2026-07-08", owner="CEO",
                visibility=LEVEL["ceo"], derived_level=LEVEL["intern"],
                why=["Acquisition accelerates last-mile fulfillment capability ahead of the UAE launch.", "Northwind's Gulf-region warehouse network removes an 18-month build-out."],
                alternatives=["Build fulfillment capability in-house", "Partner with a regional 3PL instead of acquiring"],
                assumptions=["Northwind's warehouse leases transfer cleanly", "Integration completes before the UAE launch date"],
                risks=["Deal does not close in time for the launch window", "Integration runs slower than planned"],
                review="Q3 2027", result="Approved — deal signed 2026-08-01",
                tags=["acquisition", "northwind", "logistics", "fulfillment", "uae", "confidential"],
                derived="The company is expanding its logistics and fulfillment capabilities. As a result, Product and Sales goals tied to the UAE launch now assume faster fulfillment timelines — priorities for the next two quarters have been updated accordingly.",
            ),
            dict(
                id="d-vendor-cloudsecure", title="Renew CloudSecure vendor contract", decided="2026-02-10", owner="IT Manager",
                visibility=LEVEL["employee"], derived_level=LEVEL["intern"],
                why=["CloudSecure remains the only vendor meeting our compliance certification requirements.", "Switching vendors mid-year would have disrupted the security audit already in progress."],
                alternatives=["Switch to an alternative vendor", "Renegotiate a shorter-term contract"],
                assumptions=["Pricing stays within 10% of the prior term"],
                risks=["Renewal terms lock in for another 12 months without a competitive re-bid"],
                review="Q2 2026", result="TBD",
                tags=["vendor", "contract", "cloudsecure", "compliance", "renewal"],
                derived="The CloudSecure vendor contract was renewed in early 2026; a scheduled review of the renewal terms is now overdue.",
            ),
        ]
        for d in decisions:
            db.add(Decision(company_id="cognex-labs", **d))

        goals = [
            dict(id="g-uae", depth=0, kicker="Company goal", title="Enter the UAE market in 2027", parent=None, dept=None),
            dict(id="g-marketing", depth=1, kicker="Marketing goal", title="Build brand awareness among target Gulf-region customers", parent="g-uae", dept="Marketing"),
            dict(id="g-sales", depth=1, kicker="Sales goal", title="Develop 100 qualified enterprise leads in UAE by Q2 2027", parent="g-uae", dept="Sales"),
            dict(id="g-product", depth=1, kicker="Product goal", title="Localize Product X for Gulf compliance & Arabic UI", parent="g-uae", dept="Product"),
            dict(id="g-brand-team", depth=2, kicker="Team goal", title="Produce localized campaign assets for the UAE launch", parent="g-marketing", dept="Marketing"),
            dict(id="g-sales-team", depth=2, kicker="Team goal", title="Outbound prospecting into UAE logistics & retail verticals", parent="g-sales", dept="Sales"),
            dict(id="g-product-team", depth=2, kicker="Team goal", title="Ship Arabic localization by Q4 2026", parent="g-product", dept="Product"),
            dict(id="g-jordan", depth=3, kicker="Individual work · Jordan Lee", title="Create UAE-specific product one-pagers and social assets", parent="g-brand-team", dept="Marketing", owner="employee"),
            dict(id="g-riya", depth=3, kicker="Individual work · Riya Patel", title="Localize social captions & translate the launch FAQ into Arabic", parent="g-brand-team", dept="Marketing", owner="intern"),
            dict(id="g-sam", depth=3, kicker="Individual work · Sam Okafor", title="Build target account list for the UAE logistics vertical", parent="g-sales-team", dept="Sales", owner="manager"),
        ]
        for g in goals:
            g.setdefault("owner", None)
            db.add(Goal(company_id="cognex-labs", **g))

        db.commit()
