"""
SQLAlchemy models for Cognex's real, durable persistence layer.

Why this exists: every feature built before this point (Decision Memory,
Goal Alignment, the roster, chat threads, Vantage, handoffs, the access
ledger) lived entirely in one JS object in the browser tab — it reset on
every reload and every Railway redeploy. The one exception was the Slack
integration's own SQLite tables (see db.py). This module is the real
database for the core objects: companies, people, decisions, and goals.

Deliberately scoped: chat threads, Vantage gaps/plans, handoff records, and
the access ledger are NOT modeled here yet — they stay session-only for now.
See the 2026-08-27 "Real persistence" entry in the project's build log for
the reasoning on why this slice first (it's the data that actually matters —
the decisions and goals ARE the product — and the roster/company shell has
to exist server-side before anything else can hang off it).

Uses plain `JSON` (not Postgres-specific `JSONB`) for list/dict columns so
the exact same models also work against SQLite in tests — no behavior
difference for our purposes (we never query inside these JSON blobs, only
read/write them whole), and it means the offline test suite doesn't need a
real Postgres to verify this layer's logic.
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, ForeignKeyConstraint, Integer, String, Text
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
