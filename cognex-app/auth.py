"""
Shared session-token authentication for every route that touches a specific
company's data — /api/v2 (persistence.py), /api (live.py), and
/api/integrations/slack (slack_integration.py) all import from here instead
of each keeping their own copy.

Added 2026-09-03. Extends the 2026-08-31 fix, which added a session-token
check to exactly ONE route (persistence.py's GET /companies/{id}, after an
audit found it leaking a company's full confidential Decision Memory to
anyone who could guess its id). The 2026-09-02 full-app audit found that
fix never generalized: 46 of 48 routes across the app — every write route,
every Slack route, the global file-download route — still took nothing but
a client-supplied company_id, no credential of any kind. This module is the
single place that issues, validates, and expires tokens, so that gap can't
reopen route-by-route the way it did the first time.

Still NOT real authentication: there is one shared password per company
(persistence.DEMO_PASSWORD) and one for Cognex staff
(persistence.STAFF_PASSWORD), so this doesn't stop someone who already has
that shared password from reading anything at all — it stops the
zero-credential drive-by (anyone who never signed in at all) and stops a
session issued for one company from reading or writing a DIFFERENT
company's data. Real per-user authentication (a third-party identity
provider, per the founder's own stated direction) is the deliberately
separate next phase; swapping this module's token issuance for a real IdP
session shouldn't require touching any of the ~50 call sites that now call
`require_session`/`require_any_session` — they only care about the
company_id/persona_id/kind a valid session resolves to, not how it was
issued.

Transport: the `Authorization: Bearer <token>` header, read by
`token_from_request`. A couple of GET routes used to accept `?token=...` as
a query string instead; that's been removed everywhere a fetch/XHR call
site could instead send a header (query strings land in access and proxy
logs — see the audit's medium finding on this). The one deliberate
exception is Slack's OAuth `/authorize` redirect (slack_integration.py),
which the browser navigates to directly via `window.open` rather than
`fetch` — a top-level navigation can't set a header, so that one route
still reads an explicit `?token=` query parameter. Everywhere else, no
token ever appears in a URL.

An in-memory dict is deliberately sufficient here, matching this codebase's
existing "illustrative, not real infrastructure" conventions (see
persistence.py's _SESSIONS comment this replaces, live.py's _rate_state,
etc.) — these tokens gate access for the lifetime of one server process,
not across a redeploy or a second instance. That's a real, tracked
limitation (see the "sell this professionally" audit section on reliability
& operations), not something this module tries to paper over.
"""

import secrets
import time
from typing import Optional

from fastapi import HTTPException, Request

# token -> {"kind": "company"|"platform", "company_id": str|None,
#           "persona_id": str|None, "issued_at": float, "expires_at": float}
_SESSIONS: dict = {}

# How long an issued token stays valid. Session tokens never expired at all
# before this pass (audit medium finding #17) — a stolen or copy-pasted
# token worked forever, with no logout endpoint able to actually invalidate
# it either (the frontend's old "logout" only cleared local storage, never
# told the server). 12 hours balances "don't make people re-login mid-
# workday" against "a leaked token isn't valid indefinitely."
SESSION_TTL_SECONDS = 12 * 60 * 60

# Basic login lockout (audit medium finding #18: no brute-force protection
# on a login that, for staff, unlocks every company's data via one shared
# password). Keyed by the lowercased email being attempted, not by caller
# IP — this demo has no reliable client IP behind Railway's proxy, and
# per-account lockout is what actually stops a password-guessing loop
# against one target account, which is the realistic threat here.
_LOGIN_ATTEMPTS: dict = {}  # email -> list[float] (unix timestamps of recent failures)
LOGIN_MAX_FAILURES = 8
LOGIN_LOCKOUT_WINDOW_SECONDS = 15 * 60


def issue_token(kind: str, company_id: Optional[str] = None, persona_id: Optional[str] = None) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    _SESSIONS[token] = {
        "kind": kind, "company_id": company_id, "persona_id": persona_id,
        "issued_at": now, "expires_at": now + SESSION_TTL_SECONDS,
    }
    return token


def revoke_token(token: Optional[str]) -> None:
    if token:
        _SESSIONS.pop(token, None)


def _session_for(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None
    session = _SESSIONS.get(token)
    if not session:
        return None
    if session["expires_at"] < time.time():
        _SESSIONS.pop(token, None)
        return None
    return session


def token_from_request(request: Request, query_token: Optional[str] = None) -> Optional[str]:
    """Prefers the Authorization header (`Bearer <token>`); falls back to an
    explicit query-string token only when the caller passed one (the Slack
    /authorize redirect — see the module docstring)."""
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return query_token


def require_session(token: Optional[str], company_id: Optional[str] = None) -> dict:
    """Validates a session token, returning the session dict, or raises 401.
    A platform (staff) token is always accepted regardless of company_id —
    staff legitimately need to read/support any company, mirroring the
    existing "Enter as support" flow. A company token must match the
    specific company_id being accessed, so signing in to one company never
    lets that session touch a different company's data by guessing its id."""
    session = _session_for(token)
    if not session:
        raise HTTPException(status_code=401, detail="Sign in required.")
    if session["kind"] == "platform":
        return session
    if company_id is not None and session.get("company_id") != company_id:
        raise HTTPException(status_code=401, detail="This session isn't signed in to that company.")
    return session


def require_any_session(token: Optional[str]) -> dict:
    """For routes with no single company_id to scope against — several of
    live.py's AI-processing endpoints (memory consolidation, Complete the
    Story question generation, offboarding drafting, Vantage scanning) take
    the persona/decisions/goals they operate on directly in the request
    body rather than looking anything up server-side by company. Per-company
    scoping doesn't apply there, but SOME valid, signed-in session still
    should — this is what stops those endpoints from being a fully
    anonymous, unauthenticated way to burn the server's Claude/OpenAI
    budget."""
    session = _session_for(token)
    if not session:
        raise HTTPException(status_code=401, detail="Sign in required.")
    return session


def require_persona_session(token: Optional[str], company_id: str, persona_id: str) -> dict:
    """Like require_session, but for data that's private to one specific
    person within a company — Ask Cognex chat threads, specifically. Any
    signed-in member of a company could otherwise read or overwrite a
    DIFFERENT employee's private chat history just by knowing their persona
    id, which company-level scoping alone doesn't catch. Platform (staff)
    sessions are exempt, same as everywhere else — support needs to be able
    to look at any persona's threads."""
    session = require_session(token, company_id=company_id)
    if session["kind"] == "platform":
        return session
    if session.get("persona_id") != persona_id:
        raise HTTPException(status_code=403, detail="You can only access your own chat threads.")
    return session


def check_login_lockout(email: str) -> None:
    """Raises 429 if this email has failed to log in too many times
    recently. Call BEFORE checking the submitted password."""
    now = time.time()
    attempts = [t for t in _LOGIN_ATTEMPTS.get(email, []) if now - t < LOGIN_LOCKOUT_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[email] = attempts
    if len(attempts) >= LOGIN_MAX_FAILURES:
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts for this account. Try again in a few minutes.",
        )


def record_login_failure(email: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(email, []).append(time.time())


def clear_login_failures(email: str) -> None:
    _LOGIN_ATTEMPTS.pop(email, None)
