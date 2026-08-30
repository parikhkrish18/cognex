"""
The real persistence API — mounted at /api/v2. This is the first phase of
closing the "nothing is really saved" gap: companies, the roster, decisions,
and goals now live in Postgres (see models.py / db_postgres.py) instead of
only in the browser tab.

Deliberately NOT covered by this phase (still session/browser-only, exactly
as before): Vantage gaps/plans, handoff records, and the access ledger. See
the persistence build-log entry for why this slice first and what's next.

Ask Cognex chat threads (see the ChatThread endpoints near the bottom of
this file) were added to real persistence on 2026-08-29, one phase later
than the rest — see that day's build-log entry for why.

Auth note, read together with server.py's existing one on the reference
endpoints: login here still checks a shared demo password (`DEMO_PASSWORD`)
— the same illustrative, not-real authentication the app has always had.
What changed is WHERE the account data comes from (a real database instead
of a hardcoded JS object), not how identity is verified. Real
authentication (a third-party provider was the founder's stated direction)
is the deliberately separate next phase — swapping this login endpoint's
password check for real session/token verification shouldn't require
touching the data model built here at all.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from datetime import datetime, timezone

from db_postgres import get_session
from models import ChatThread, Company, Decision, Goal, Persona

router = APIRouter(prefix="/api/v2")

DEMO_PASSWORD = "cognex"
STAFF_EMAIL = "admin@cognex.ai"
# Staff sign-in gets its own password, separate from the shared per-company
# demo password above — the frontend's staff-sign-in modal never displays it
# (no prefilled prompt, no footnote), unlike the regular login screen, which
# does show DEMO_PASSWORD on screen as an explicit "this demo isn't really
# authenticated" disclosure.
STAFF_PASSWORD = "Admin@Cognex"


# ---------------------------------------------------------------------------
# Serialization — matches the exact camelCase shape static/index.html already
# expects (it used to build these objects itself; now it fetches them).
# ---------------------------------------------------------------------------
def _persona_json(p: Persona) -> dict:
    return {
        "id": p.id, "name": p.name, "title": p.title, "level": p.level, "dept": p.dept,
        "email": p.email, "reportsTo": p.reports_to, "isOrgAdmin": p.is_org_admin,
        "initials": p.initials, "bg": p.bg, "fg": p.fg,
    }


def _decision_json(d: Decision) -> dict:
    out = {
        "id": d.id, "title": d.title, "decided": d.decided, "owner": d.owner,
        "visibility": d.visibility, "derivedLevel": d.derived_level,
        "why": d.why or [], "alternatives": d.alternatives or [], "assumptions": d.assumptions or [],
        "risks": d.risks or [], "review": d.review, "result": d.result,
        "tags": d.tags or [], "derived": d.derived,
    }
    if d.contributor:
        out["contributor"] = d.contributor
    if d.via_story:
        out["viaStory"] = True
    if d.via_slack:
        out["viaSlack"] = True
    if d.via_chat:
        out["viaChat"] = True
    if d.via_vantage:
        out["viaVantage"] = True
    return out


def _goal_json(g: Goal) -> dict:
    out = {"id": g.id, "title": g.title, "kicker": g.kicker, "depth": g.depth, "parent": g.parent, "dept": g.dept}
    if g.owner:
        out["owner"] = g.owner
    return out


def _company_snapshot(db, company: Company) -> dict:
    personas = db.execute(select(Persona).where(Persona.company_id == company.id)).scalars().all()
    decisions = db.execute(select(Decision).where(Decision.company_id == company.id)).scalars().all()
    goals = db.execute(select(Goal).where(Goal.company_id == company.id)).scalars().all()
    return {
        "id": company.id, "name": company.name, "industry": company.industry, "seeded": company.seeded,
        "personas": {p.id: _persona_json(p) for p in personas},
        "decisions": [_decision_json(d) for d in decisions],
        "goals": [_goal_json(g) for g in goals],
    }


def _get_company_or_404(db, company_id: str) -> Company:
    company = db.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail=f"No company '{company_id}'.")
    return company


def _slugify(name: str) -> str:
    base = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in base:
        base = base.replace("--", "-")
    return base or "company"


# ---------------------------------------------------------------------------
# Company snapshot + login + staff console listing
# ---------------------------------------------------------------------------
@router.get("/companies/{company_id}")
def get_company(company_id: str):
    with get_session() as db:
        company = _get_company_or_404(db, company_id)
        return _company_snapshot(db, company)


@router.get("/platform/companies")
def list_companies():
    with get_session() as db:
        companies = db.execute(select(Company)).scalars().all()
        rows = []
        for c in companies:
            personas = db.execute(select(Persona).where(Persona.company_id == c.id)).scalars().all()
            rows.append({
                "id": c.id, "name": c.name, "industry": c.industry, "seeded": c.seeded,
                "memberCount": len(personas),
                "adminCount": sum(1 for p in personas if p.is_org_admin),
            })
        return {"companies": rows}


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    if email == STAFF_EMAIL:
        if req.password != STAFF_PASSWORD:
            raise HTTPException(status_code=401, detail="Incorrect password for Cognex staff.")
        return {"type": "platform"}

    with get_session() as db:
        persona = db.execute(select(Persona).where(Persona.email == email)).scalars().first()
        if not persona or req.password != DEMO_PASSWORD:
            raise HTTPException(status_code=401, detail="No matching account, or the password's wrong.")
        company = _get_company_or_404(db, persona.company_id)
        return {"type": "company", "companyId": company.id, "personaId": persona.id, "company": _company_snapshot(db, company)}


# ---------------------------------------------------------------------------
# Company creation (staff onboarding) — mirrors the wizard: exactly one
# founding Team Admin, created together with the company.
# ---------------------------------------------------------------------------
class FounderIn(BaseModel):
    name: str
    title: str = "CEO"
    email: str


class CompanyCreateRequest(BaseModel):
    name: str
    industry: str = "—"
    founder: FounderIn


@router.post("/companies")
def create_company(req: CompanyCreateRequest):
    email = req.founder.email.strip().lower()
    with get_session() as db:
        existing = db.execute(select(Persona).where(Persona.email == email)).scalars().first()
        if existing:
            raise HTTPException(status_code=409, detail="That email is already in use by another account.")
        base_id = _slugify(req.name)
        company_id, n = base_id, 1
        while db.get(Company, company_id):
            n += 1
            company_id = f"{base_id}-{n}"

        company = Company(id=company_id, name=req.name, industry=req.industry or "—", seeded=False)
        db.add(company)
        founder_id = "founder"
        db.add(Persona(
            company_id=company_id, id=founder_id, name=req.founder.name, title=req.founder.title or "CEO",
            level=6, dept="Executive", email=email, reports_to=None, is_org_admin=True,
            initials="".join(w[0] for w in req.founder.name.split() if w)[:2].upper(),
            bg="#e8f0fe", fg="#0071e3",
        ))
        db.add(Goal(company_id=company_id, id="g-root", depth=0, kicker="Company goal", title="Define your first company goal", parent=None, dept=None))
        db.commit()
        return _company_snapshot(db, company)


# ---------------------------------------------------------------------------
# Roster (personas)
# ---------------------------------------------------------------------------
class PersonaIn(BaseModel):
    id: Optional[str] = None
    name: str
    title: str = ""
    level: int
    dept: str = ""
    email: str
    reportsTo: Optional[str] = None
    isOrgAdmin: bool = False


@router.post("/companies/{company_id}/personas")
def add_persona(company_id: str, req: PersonaIn):
    with get_session() as db:
        company = _get_company_or_404(db, company_id)
        pid = req.id or _slugify(req.name)
        if db.get(Persona, (company_id, pid)):
            base, n = pid, 1
            while db.get(Persona, (company_id, pid)):
                n += 1
                pid = f"{base}-{n}"
        color_idx = len(db.execute(select(Persona).where(Persona.company_id == company_id)).scalars().all())
        colors = [{"bg": "#e8f0fe", "fg": "#0071e3"}, {"bg": "#e6f9ec", "fg": "#1f8a4c"}, {"bg": "#fff1e0", "fg": "#b45c09"},
                  {"bg": "#f5e9fc", "fg": "#7a2fb0"}, {"bg": "#ffe9f0", "fg": "#c2185b"}, {"bg": "#e6f9fb", "fg": "#0f7a8c"}]
        color = colors[color_idx % len(colors)]
        p = Persona(
            company_id=company_id, id=pid, name=req.name, title=req.title, level=req.level, dept=req.dept,
            email=req.email.strip().lower(), reports_to=req.reportsTo, is_org_admin=req.isOrgAdmin,
            initials="".join(w[0] for w in req.name.split() if w)[:2].upper(), bg=color["bg"], fg=color["fg"],
        )
        db.add(p)
        db.commit()
        return _persona_json(p)


@router.put("/companies/{company_id}/personas/{persona_id}")
def update_persona(company_id: str, persona_id: str, req: PersonaIn):
    with get_session() as db:
        p = db.get(Persona, (company_id, persona_id))
        if not p:
            raise HTTPException(status_code=404, detail="No such persona.")
        p.name, p.title, p.level, p.dept = req.name, req.title, req.level, req.dept
        p.email, p.reports_to, p.is_org_admin = req.email.strip().lower(), req.reportsTo, req.isOrgAdmin
        db.commit()
        return _persona_json(p)


@router.delete("/companies/{company_id}/personas/{persona_id}")
def delete_persona(company_id: str, persona_id: str, acting_persona_id: Optional[str] = None):
    with get_session() as db:
        p = db.get(Persona, (company_id, persona_id))
        if not p:
            raise HTTPException(status_code=404, detail="No such persona.")
        if acting_persona_id == persona_id:
            raise HTTPException(status_code=400, detail="You can't remove your own account while signed in as them.")
        if p.is_org_admin:
            admin_count = len(db.execute(select(Persona).where(Persona.company_id == company_id, Persona.is_org_admin == True)).scalars().all())  # noqa: E712
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="Can't remove the last team admin. Promote someone else first.")
        db.delete(p)
        db.commit()
        return {"ok": True}


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
class DecisionIn(BaseModel):
    id: Optional[str] = None
    title: str
    decided: str = ""
    owner: str = ""
    visibility: int = 1
    derivedLevel: int = 1
    why: list = []
    alternatives: list = []
    assumptions: list = []
    risks: list = []
    review: str = "Not yet scheduled"
    result: str = "TBD"
    tags: list = []
    derived: str = ""
    contributor: Optional[str] = None
    viaStory: bool = False
    viaSlack: bool = False
    viaChat: bool = False
    viaVantage: bool = False


@router.post("/companies/{company_id}/decisions")
def add_decision(company_id: str, req: DecisionIn):
    with get_session() as db:
        _get_company_or_404(db, company_id)
        did = req.id or f"d-{_slugify(req.title)}"
        if db.get(Decision, (company_id, did)):
            did = f"{did}-{len(db.execute(select(Decision).where(Decision.company_id == company_id)).scalars().all()) + 1}"
        d = Decision(
            company_id=company_id, id=did, title=req.title, decided=req.decided, owner=req.owner,
            visibility=req.visibility, derived_level=req.derivedLevel, why=req.why, alternatives=req.alternatives,
            assumptions=req.assumptions, risks=req.risks, review=req.review, result=req.result, tags=req.tags,
            derived=req.derived, contributor=req.contributor, via_story=req.viaStory, via_slack=req.viaSlack,
            via_chat=req.viaChat, via_vantage=req.viaVantage,
        )
        db.add(d)
        db.commit()
        return _decision_json(d)


class DecisionUpdateIn(BaseModel):
    result: Optional[str] = None


@router.put("/companies/{company_id}/decisions/{decision_id}")
def update_decision(company_id: str, decision_id: str, req: DecisionUpdateIn):
    with get_session() as db:
        d = db.get(Decision, (company_id, decision_id))
        if not d:
            raise HTTPException(status_code=404, detail="No such decision.")
        if req.result is not None:
            d.result = req.result
        db.commit()
        return _decision_json(d)


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------
class GoalIn(BaseModel):
    id: Optional[str] = None
    title: str
    kicker: str = ""
    depth: int = 0
    parent: Optional[str] = None
    dept: Optional[str] = None
    owner: Optional[str] = None


@router.post("/companies/{company_id}/goals")
def add_goal(company_id: str, req: GoalIn):
    with get_session() as db:
        _get_company_or_404(db, company_id)
        gid = req.id or f"g-{_slugify(req.title)}"
        if db.get(Goal, (company_id, gid)):
            gid = f"{gid}-{len(db.execute(select(Goal).where(Goal.company_id == company_id)).scalars().all()) + 1}"
        g = Goal(company_id=company_id, id=gid, title=req.title, kicker=req.kicker, depth=req.depth,
                  parent=req.parent, dept=req.dept, owner=req.owner)
        db.add(g)
        db.commit()
        return _goal_json(g)


class GoalUpdateIn(BaseModel):
    owner: Optional[str] = None


@router.put("/companies/{company_id}/goals/{goal_id}")
def update_goal(company_id: str, goal_id: str, req: GoalUpdateIn):
    with get_session() as db:
        g = db.get(Goal, (company_id, goal_id))
        if not g:
            raise HTTPException(status_code=404, detail="No such goal.")
        if req.owner is not None:
            g.owner = req.owner
        db.commit()
        return _goal_json(g)


# ---------------------------------------------------------------------------
# Chat threads (Ask Cognex conversations) — added 2026-08-29, see ChatThread's
# docstring in models.py for why this exists and why `turns` is one JSON blob
# per thread rather than a normalized per-turn table.
# ---------------------------------------------------------------------------
def _thread_json(t: ChatThread) -> dict:
    return {"id": t.id, "title": t.title, "turns": t.turns or [], "updatedAt": t.updated_at}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/companies/{company_id}/personas/{persona_id}/threads")
def list_threads(company_id: str, persona_id: str):
    """Fetched once on login/session start so a persona's chat history is
    there from the first render, instead of every session starting with an
    empty sidebar the way it did before this endpoint existed."""
    with get_session() as db:
        rows = db.execute(
            select(ChatThread)
            .where(ChatThread.company_id == company_id, ChatThread.persona_id == persona_id)
            .order_by(ChatThread.updated_at.desc())
        ).scalars().all()
        return {"threads": [_thread_json(t) for t in rows]}


class ThreadUpsertIn(BaseModel):
    id: Optional[str] = None
    title: str = "New chat"
    turns: list = []


@router.put("/companies/{company_id}/personas/{persona_id}/threads/{thread_id}")
def upsert_thread(company_id: str, persona_id: str, thread_id: str, req: ThreadUpsertIn):
    """Upsert, not separate create/update — the frontend always knows the
    thread id it's writing (it mints one client-side the moment a new chat
    is started, same as it already does for decisions/goals), so one
    idempotent call handles both "first turn in a brand-new thread" and
    "another turn in an existing one" without the frontend having to track
    which case it's in. Called fire-and-forget after every turn completes
    and on rename/thread creation, mirroring persistNewDecision's pattern —
    the browser-local state.chatByKey stays the source of truth for
    rendering, this is a best-effort mirror so a reload has something real
    to restore from."""
    with get_session() as db:
        _get_company_or_404(db, company_id)
        t = db.get(ChatThread, (company_id, persona_id, thread_id))
        if not t:
            t = ChatThread(company_id=company_id, persona_id=persona_id, id=thread_id)
            db.add(t)
        t.title = req.title or "New chat"
        t.turns = req.turns
        t.updated_at = _now_iso()
        db.commit()
        return _thread_json(t)


@router.delete("/companies/{company_id}/personas/{persona_id}/threads/{thread_id}")
def delete_thread(company_id: str, persona_id: str, thread_id: str):
    with get_session() as db:
        t = db.get(ChatThread, (company_id, persona_id, thread_id))
        if not t:
            return {"ok": True}  # already gone — deleting twice isn't an error
        db.delete(t)
        db.commit()
        return {"ok": True}
