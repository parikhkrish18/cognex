"""
SQLAlchemy models for Cognex's real, durable persistence layer.

Why this exists: every feature built before this point (Decision Memory,
Goal Alignment, the roster, chat threads, Vantage, handoffs, the access
ledger) lived entirely in one JS object in the browser tab — it reset on
every reload and every Railway redeploy. The one exception was the Slack
integration's own SQLite tables (see db.py). This module is the real
database for the core objects: companies, people, decisions, and goals.

Originally scoped to exclude chat threads, Vantage gaps/plans, handoff
records, and the access ledger — see the 2026-08-27 "Real persistence" entry
in the project's build log for the reasoning on why companies/personas/
decisions/goals came first (they're the data that actually matters — the
decisions and goals ARE the product — and the roster/company shell has to
exist server-side before anything else can hang off it). Chat threads were
added 2026-08-29 (see ChatThread below), and handoffs/the access ledger were
added 2026-08-31 (see Handoff/LedgerEntry below) after an audit found both
were being actively sold — Handoff Capture, Access Ledger & audit exports —
while still living only in browser memory. Vantage gaps/plans remain
session-only; they're not sold as a standalone persisted feature the way
those two are.

Uses plain `JSON` (not Postgres-specific `JSONB`) for list/dict columns so
the exact same models also work against SQLite in tests — no behavior
difference for our purposes (we never query inside these JSON blobs, only
read/write them whole), and it means the offline test suite doesn't need a
real Postgres to verify this layer's logic.
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, ForeignKeyConstraint, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    industry: Mapped[str] = mapped_column(String, default="—")
    seeded: Mapped[bool] = mapped_column(Boolean, default=False)


class Persona(Base):
    """A member of a company's roster. `id` is only unique WITHIN a company
    (e.g. "ceo", "cfo" for the seed data, or a slug for real members) —
    the real primary key is the (company_id, id) pair, same as every other
    table here. `reports_to` is a plain string referencing another
    persona's `id` within the same company, not a DB-level foreign key
    (mirrors how the frontend already modeled the chain of command)."""

    __tablename__ = "personas"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, default="")
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    dept: Mapped[str] = mapped_column(String, default="")
    email: Mapped[str] = mapped_column(String, nullable=False)
    reports_to: Mapped[str | None] = mapped_column(String, nullable=True)
    is_org_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    initials: Mapped[str] = mapped_column(String, default="")
    bg: Mapped[str] = mapped_column(String, default="#e8e8ed")
    fg: Mapped[str] = mapped_column(String, default="#1d1d1f")

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class Decision(Base):
    __tablename__ = "decisions"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    decided: Mapped[str] = mapped_column(String, default="")
    owner: Mapped[str] = mapped_column(String, default="")
    visibility: Mapped[int] = mapped_column(Integer, default=1)
    derived_level: Mapped[int] = mapped_column(Integer, default=1)
    why: Mapped[list] = mapped_column(JSON, default=list)
    alternatives: Mapped[list] = mapped_column(JSON, default=list)
    assumptions: Mapped[list] = mapped_column(JSON, default=list)
    risks: Mapped[list] = mapped_column(JSON, default=list)
    review: Mapped[str] = mapped_column(String, default="")
    result: Mapped[str] = mapped_column(String, default="TBD")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    derived: Mapped[str] = mapped_column(Text, default="")
    contributor: Mapped[str | None] = mapped_column(String, nullable=True)
    via_story: Mapped[bool] = mapped_column(Boolean, default=False)
    via_slack: Mapped[bool] = mapped_column(Boolean, default=False)
    via_chat: Mapped[bool] = mapped_column(Boolean, default=False)
    via_vantage: Mapped[bool] = mapped_column(Boolean, default=False)
    # Added 2026-08-31 for the department-hub Brain Board redesign (direct
    # founder ask: core business departments that "strand" into smaller,
    # sharper parts as the company brain grows). `domain` is the broad
    # department name that decides which colored arc of the ring a node
    # sits under. Originally a fixed six-value set baked into live.py; per
    # 2026-09-01 founder feedback ("the company brain should smartly grow
    # using AI and gain new departments as required") it's now just a plain
    # free-text string Claude assigns per topic (see live.py's
    # consolidate_memory / _normalize_domain) -- no enum or fixed list
    # anywhere, including at this column's own default below, which is only
    # ever a defensive fallback for a row nothing has classified yet.
    # `parent_id` makes a node a "branch"
    # of another node already on the board (null = a "trunk", attached
    # directly to its domain's arc) instead of every node sitting in one
    # flat ring -- populated either by consolidate_memory's "new_branch"
    # action, or by the split endpoint below when a trunk has absorbed too
    # many genuinely distinct sub-topics and gets split into sharper
    # children. Self-referential within the same company -- deliberately
    # just a plain string column rather than a real FK constraint, matching
    # how Goal.parent already works one table over, so a parent can be
    # deleted (cascading to its children, mirroring Goal's own cascading
    # delete) without fighting a DB-level constraint.
    domain: Mapped[str] = mapped_column(String, default="other")
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class Goal(Base):
    __tablename__ = "goals"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    kicker: Mapped[str] = mapped_column(String, default="")
    depth: Mapped[int] = mapped_column(Integer, default=0)
    parent: Mapped[str | None] = mapped_column(String, nullable=True)
    dept: Mapped[str | None] = mapped_column(String, nullable=True)
    owner: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class Handoff(Base):
    """A departing employee's captured handoff -- who's leaving, who it goes
    to, the Q&A that produced it, the extracted report, and the delegation
    checklist derived from it. Added 2026-08-31: this used to live only in
    the browser tab's `company.handoffs` array (explicitly flagged in this
    file's module docstring above as one of the pieces "NOT modeled here
    yet"), even though Handoff Capture is a feature actively sold on the
    Business tier -- a customer relying on it would find every handoff gone
    after a Railway redeploy or a page reload. `id` is only unique within a
    company, same convention as every other table here. `report` and
    `delegation_items` are stored as whole JSON blobs (not normalized),
    matching this file's existing convention for structured-but-never-
    queried-into fields (see Decision.why, ChatThread.turns)."""

    __tablename__ = "handoffs"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    from_persona_id: Mapped[str] = mapped_column(String, default="")
    from_name: Mapped[str] = mapped_column(String, default="")
    from_title: Mapped[str] = mapped_column(String, default="")
    from_dept: Mapped[str] = mapped_column(String, default="")
    to_persona_id: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String, default="")
    qa_pairs: Mapped[list] = mapped_column(JSON, default=list)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    offline: Mapped[bool] = mapped_column(Boolean, default=False)
    delegation_items: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="pending")

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class LedgerEntry(Base):
    """One row of the Access Ledger -- who saw what, at what access tier,
    via which surface (Decision Memory tab, Ask Cognex). Added 2026-08-31
    for the same reason as Handoff above: sold as an Enterprise-tier
    "Access Ledger & audit exports" feature while living only in browser
    memory -- empty for a compliance reviewer the moment anyone reloaded
    the page or the server redeployed, the opposite of what an audit trail
    is for. `id` includes a client-generated random suffix (matching the
    frontend's existing id scheme) since many entries can be created in the
    same millisecond (logDecisionMemoryAccess logs one per visible decision
    in a single loop)."""

    __tablename__ = "ledger_entries"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[str] = mapped_column(String, default="")
    persona_id: Mapped[str] = mapped_column(String, default="")
    persona_name: Mapped[str] = mapped_column(String, default="")
    persona_title: Mapped[str] = mapped_column(String, default="")
    decision_id: Mapped[str] = mapped_column(String, default="")
    decision_title: Mapped[str] = mapped_column(String, default="")
    access: Mapped[str] = mapped_column(String, default="")
    via: Mapped[str] = mapped_column(String, default="")

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class CompanyUsage(Base):
    """Cumulative ESTIMATED Anthropic/OpenAI spend for one company in one
    calendar-month `period` ("YYYY-MM"), added 2026-08-31 to back a basic
    per-company cost cap (see live.py's _check_cost_cap/_record_cost) after
    an audit found nothing stopped one user -- or anyone who had the shared
    demo password -- from running up an unbounded bill via unlimited Ask
    Cognex/image-generation calls. Deliberately coarse: a running cents
    total computed from real token/image usage at fixed published per-unit
    prices, not a reconciled invoice. This exists to be a circuit breaker
    against runaway spend, not a billing system."""

    __tablename__ = "company_usage"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    period: Mapped[str] = mapped_column(String, primary_key=True)
    estimated_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class Document(Base):
    """A company's uploaded document library (added 2026-08-31, direct
    founder feedback: "I am not able to upload files"). The founder's own
    explicit choice on scope was the biggest of three options offered: a
    real persistent, company-wide, searchable library -- uploaded once,
    referenceable across every future chat -- not a one-off per-message
    chat attachment.

    Deliberately keyed by a single globally-unique `id`, NOT the
    (company_id, id) composite key every other table in this file uses.
    Downloads resolve through the existing global GET /api/files/{file_id}
    route in live.py (extended with a "doc-" prefix branch) the exact same
    way an OpenAI-generated image ("img-" prefix) or an Anthropic Files API
    upload already does -- and that route has no company_id in its URL path
    at all, so a document id has to be globally resolvable on its own.
    `company_id` is still a plain filterable column here (not part of the
    primary key) for listing/scoping a company's own library and for the
    search_documents/get_document tools in live.py.

    File bytes live in this table (`content_bytes`), not on local disk --
    matching this whole app's one-Postgres-instance architecture, and
    avoiding exactly the "gone on every Railway redeploy" gap the 2026-08-31
    Handoff Capture / Access Ledger fix above closed for those two features:
    Railway's filesystem isn't guaranteed to survive a redeploy, Postgres is.
    There's a per-file size cap enforced in persistence.py's upload route
    (a database column, not an object-storage tier, so this is deliberately
    modest -- fine for the kind of reference docs a small company uploads,
    not a general file-storage product).

    `text_content` is the extracted plain text used for search_documents'
    keyword matching (same approach as Decision.why -- a flat blob, not a
    real search index). `extraction_status` is "indexed" (txt/md/csv read
    directly, or pdf/docx text pulled via pypdf/python-docx), "not_indexed"
    (a file type this app doesn't know how to extract text from -- still
    stored and downloadable, just not searchable by content), or "failed"
    (a recognized type whose extraction itself errored, e.g. a corrupt or
    password-protected PDF)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    company_id: Mapped[str] = mapped_column(String, index=True)
    filename: Mapped[str] = mapped_column(String, default="")
    mime_type: Mapped[str] = mapped_column(String, default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_by: Mapped[str] = mapped_column(String, default="")
    uploaded_at: Mapped[str] = mapped_column(String, default="")
    extraction_status: Mapped[str] = mapped_column(String, default="not_indexed")
    text_content: Mapped[str] = mapped_column(Text, default="")
    content_bytes: Mapped[bytes] = mapped_column(LargeBinary, default=b"")

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )


class ChatThread(Base):
    """An Ask Cognex conversation. Added 2026-08-29 in direct response to a
    founder bug report ("the multiple chats thing is not working, no saved
    chats at all") and the immediate follow-up asking Cognex to "remember
    all its memory" — until now, chat threads lived only in the browser
    tab's `state.chatByKey` JS object (see the 2026-08-27 persistence
    entry's explicit scope note: "Deliberately NOT covered by this phase...
    Ask Cognex chat threads"), so every reload — and `index.html` is served
    with Cache-Control: no-store on purpose (see that same day's entry) —
    silently wiped every conversation.

    `turns` stores the whole turn list as one JSON blob per thread rather
    than a normalized per-turn table: nothing else in this app ever needs
    to query into an individual turn (search/grounding already runs over
    Decision Memory, not raw chat), and every other write in this
    persistence layer already follows the same "send the whole current
    object, upsert it" pattern (see persistNewDecision in the frontend) —
    consistent, not a shortcut. Composite primary key is (company_id,
    persona_id, id): a thread only belongs to one persona, and persona ids
    are themselves only unique within a company (see Persona's own
    docstring), so this mirrors that same nesting exactly."""

    __tablename__ = "chat_threads"

    company_id: Mapped[str] = mapped_column(String, primary_key=True)
    persona_id: Mapped[str] = mapped_column(String, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, default="New chat")
    turns: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[str] = mapped_column(String, default="")

    __table_args__ = (
        ForeignKeyConstraint(["company_id"], ["companies.id"]),
    )
