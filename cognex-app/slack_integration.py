"""
Slack integration — OAuth connect, ingestion, and Claude-powered decision
extraction from real Slack history.

Architecture, and why it's shaped this way:

  1. OAuth (authorize/callback) gets a bot token for the customer's Slack
     workspace, stored in SQLite (db.py) keyed by company_id — the only
     piece of this app that needs to survive a redeploy to be useful. The
     "Connect Slack" button opens this flow in a POPUP window, not the main
     tab: this app has no session persistence yet (COMPANIES resets on
     reload, see DEPLOY.md), so navigating the main tab away to Slack and
     back would silently lose the admin's in-memory session. The popup
     completes OAuth, shows a "connected, you can close this tab" page, and
     the main tab picks up the new connection by polling /status.

  2. Sync pulls recent channel history via Slack's Web API and stores raw
     messages. This is polling, not push — no Events API subscription in
     this pass. Simpler, and enough to prove the pipeline; a real production
     build would add Events API webhooks for near-real-time capture instead
     of a manual "Sync now" button.

  3. Extract runs a forced-tool-call Claude pass (same pattern as
     extract.py / live.py) over unprocessed messages, grouped by channel,
     looking for decision-shaped moments — NOT a summary of every message,
     a filter for the minority that actually constitute something worth
     remembering. Results are written to slack_candidates with
     status='pending' — never directly into Decision Memory. Same principle
     Complete the Story already established: a human confirms before
     company memory changes.

  4. Permission mapping: each candidate's suggested visibility is computed
     from the ACTUAL Slack channel membership at sync time, matched to
     Cognex personas by email, defaulting to the LOWEST clearance level
     among members who were in the channel — because those people already
     saw the raw conversation in Slack; Cognex restricting it further than
     that wouldn't add security, only make Cognex less useful than Slack
     itself for them. This is a suggestion a human can override before
     confirming, never an auto-applied final answer.

  What this deliberately does NOT do in this pass: real-time sync (polling
  only), full historical backfill (most recent ~150 messages per channel),
  or continuous re-sync of channel membership after the fact (a candidate's
  suggested visibility reflects membership AT SYNC TIME — if someone leaves
  a channel later, already-extracted candidates don't retroactively change).
  These are honest, known limitations of a first real pass at the hardest
  part of this product: turning raw content into curated memory.
"""

import os
import json
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from anthropic import Anthropic

from auth import require_session, token_from_request
from db import get_conn

router = APIRouter(prefix="/api/integrations/slack")

# ---------------------------------------------------------------------------
# OAuth CSRF protection — added 2026-09-03 (full-app-audit-2026-09-02 high
# finding). /authorize used to pass the plaintext company_id straight
# through as the OAuth `state` param with no server-side nonce at all: an
# attacker could run their own OAuth grant against their own Slack
# workspace, then call OUR /callback with state=<victim-company-id>,
# silently overwriting that victim's real Slack connection with the
# attacker's token — a complete hijack with zero interaction from the
# victim. Now /authorize mints a random, single-use, short-lived nonce
# mapped to the company_id it was actually issued for; /callback looks the
# nonce up and rejects anything it doesn't recognize, so a state value has
# to have come from a real, freshly-issued /authorize call for that exact
# company. Same in-memory-dict trade-off as the rest of this app's session
# state (see auth.py) — sufficient for a flow that's only ever alive for
# the few minutes between clicking "Connect Slack" and the popup closing.
# ---------------------------------------------------------------------------
_OAUTH_STATE: dict = {}  # nonce -> {"company_id": str, "expires_at": float}
_OAUTH_STATE_TTL_SECONDS = 10 * 60


def _issue_oauth_state(company_id: str) -> str:
    nonce = secrets.token_urlsafe(24)
    _OAUTH_STATE[nonce] = {"company_id": company_id, "expires_at": time.time() + _OAUTH_STATE_TTL_SECONDS}
    return nonce


def _consume_oauth_state(nonce: str) -> Optional[str]:
    """One-time use: a nonce is removed as soon as it's looked up, whether
    or not it turns out to be valid, so a captured callback URL can't be
    replayed."""
    entry = _OAUTH_STATE.pop(nonce, None)
    if not entry or entry["expires_at"] < time.time():
        return None
    return entry["company_id"]

SLACK_CLIENT_ID = os.environ.get("SLACK_CLIENT_ID", "")
SLACK_CLIENT_SECRET = os.environ.get("SLACK_CLIENT_SECRET", "")
SLACK_REDIRECT_URI = os.environ.get("SLACK_REDIRECT_URI", "")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# Bot scopes only — this reads workspace content on behalf of the company,
# not on behalf of an individual user. users:read.email is what lets sync()
# map Slack members to Cognex personas' emails for the visibility default.
BOT_SCOPES = "channels:history,channels:read,groups:history,groups:read,users:read,users:read.email,team:read"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _configured():
    return bool(SLACK_CLIENT_ID and SLACK_CLIENT_SECRET and SLACK_REDIRECT_URI)


@router.get("/status")
def status(company_id: str, request: Request, token: Optional[str] = None):
    require_session(token_from_request(request, token), company_id=company_id)
    # "configured" = the server has Slack app credentials set up at all (gates
    # whether "Connect" can start a new OAuth flow). "connected" is read from
    # the DB independent of that — a company that already has a stored token
    # should still show connected even if the env vars were unset later.
    conn = get_conn()
    row = conn.execute(
        "SELECT team_name, connected_at FROM slack_connections WHERE company_id=?", (company_id,)
    ).fetchone()
    if not row:
        conn.close()
        return {"configured": _configured(), "connected": False}
    msg_count = conn.execute(
        "SELECT COUNT(*) c FROM slack_messages WHERE company_id=?", (company_id,)
    ).fetchone()["c"]
    pending_count = conn.execute(
        "SELECT COUNT(*) c FROM slack_candidates WHERE company_id=? AND status='pending'", (company_id,)
    ).fetchone()["c"]
    conn.close()
    return {
        "configured": _configured(),
        "connected": True,
        "team_name": row["team_name"],
        "connected_at": row["connected_at"],
        "message_count": msg_count,
        "pending_candidates": pending_count,
    }


@router.get("/authorize")
def authorize(company_id: str, request: Request, token: Optional[str] = None):
    # This route is reached by a top-level browser navigation (slackConnect's
    # window.open popup, not a fetch/XHR call), so it can't carry an
    # Authorization header the way every other route in this app now does —
    # ?token= here is the one deliberate exception (see auth.py's module
    # docstring). Still required: without a valid session scoped to this
    # exact company_id, starting the OAuth flow (and therefore being able to
    # connect to any company's Slack) isn't allowed at all.
    require_session(token_from_request(request, token), company_id=company_id)
    if not _configured():
        raise HTTPException(status_code=503, detail="Slack is not configured on the server yet.")
    state = _issue_oauth_state(company_id)
    url = (
        "https://slack.com/oauth/v2/authorize"
        f"?client_id={SLACK_CLIENT_ID}&scope={BOT_SCOPES}"
        f"&redirect_uri={SLACK_REDIRECT_URI}&state={state}"
    )
    return RedirectResponse(url)


@router.get("/callback")
def callback(code: str = "", state: str = "", error: str = ""):
    page = lambda body: HTMLResponse(
        "<html><body style='font-family:sans-serif;padding:40px;text-align:center;'>"
        f"<p>{body}</p>"
        "<script>setTimeout(()=>window.close(), 2500)</script>"
        "</body></html>"
    )
    if error or not code:
        return page(f"Slack connection failed: {error or 'no code returned'}. Close this tab and try again.")
    # Resolves the real company_id from the nonce /authorize issued —
    # NEVER trusts `state` itself as a company_id (see the CSRF comment
    # above this file's _OAUTH_STATE). An unrecognized/expired/already-used
    # nonce means this callback didn't originate from a /authorize call this
    # server actually issued moments ago, so it's rejected outright rather
    # than connecting Slack to whatever company an attacker named.
    company_id = _consume_oauth_state(state)
    if not company_id:
        return page("This Slack connection link has expired or was already used. Close this tab and click Connect Slack again.")
    resp = httpx.post(
        "https://slack.com/api/oauth.v2.access",
        data={
            "client_id": SLACK_CLIENT_ID,
            "client_secret": SLACK_CLIENT_SECRET,
            "code": code,
            "redirect_uri": SLACK_REDIRECT_URI,
        },
        timeout=15,
    )
    data = resp.json()
    if not data.get("ok"):
        return page(f"Slack rejected the connection: {data.get('error')}. Close this tab and try again.")

    conn = get_conn()
    conn.execute(
        "INSERT INTO slack_connections (company_id, team_id, team_name, access_token, connected_by, connected_at) "
        "VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(company_id) DO UPDATE SET team_id=excluded.team_id, team_name=excluded.team_name, "
        "access_token=excluded.access_token, connected_at=excluded.connected_at",
        (
            company_id,
            data["team"]["id"],
            data["team"]["name"],
            data["access_token"],
            data.get("authed_user", {}).get("id"),
            _now(),
        ),
    )
    conn.commit()
    conn.close()
    return page(f"Slack connected to {data['team']['name']}. You can close this tab and go back to Cognex.")


@router.post("/disconnect")
def disconnect(company_id: str, request: Request, token: Optional[str] = None):
    require_session(token_from_request(request, token), company_id=company_id)
    conn = get_conn()
    conn.execute("DELETE FROM slack_connections WHERE company_id=?", (company_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


def _slack_get(token, method, **params):
    r = httpx.get(
        f"https://slack.com/api/{method}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=15,
    )
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error on {method}: {data.get('error')}")
    return data


class SyncRequest(BaseModel):
    company_id: str


@router.post("/sync")
def sync(req: SyncRequest, request: Request):
    require_session(token_from_request(request), company_id=req.company_id)
    conn = get_conn()
    row = conn.execute(
        "SELECT access_token FROM slack_connections WHERE company_id=?", (req.company_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="No Slack connection for this company yet.")
    token = row["access_token"]

    try:
        channels_data = _slack_get(token, "conversations.list", types="public_channel,private_channel", limit=100)
    except RuntimeError as e:
        conn.close()
        raise HTTPException(status_code=502, detail=str(e))

    email_by_user = {}
    ingested = 0
    channel_summaries = []

    for ch in channels_data.get("channels", []):
        if not ch.get("is_member", True):
            continue
        try:
            members_data = _slack_get(token, "conversations.members", channel=ch["id"], limit=200)
        except RuntimeError:
            continue
        member_ids = members_data.get("members", [])
        member_emails = []
        for uid in member_ids:
            if uid not in email_by_user:
                try:
                    u = _slack_get(token, "users.info", user=uid)
                    email_by_user[uid] = (u["user"].get("profile", {}) or {}).get("email", "")
                except RuntimeError:
                    email_by_user[uid] = ""
            if email_by_user[uid]:
                member_emails.append(email_by_user[uid])

        try:
            history = _slack_get(token, "conversations.history", channel=ch["id"], limit=150)
        except RuntimeError:
            continue

        for m in history.get("messages", []):
            if m.get("subtype") or not m.get("text"):
                continue  # skip joins/leaves/bot edits/empty messages
            mid = f"{ch['id']}-{m['ts']}"
            user_email = email_by_user.get(m.get("user", ""), "")
            cur = conn.execute("SELECT 1 FROM slack_messages WHERE id=?", (mid,)).fetchone()
            if cur:
                continue
            conn.execute(
                "INSERT INTO slack_messages (id, company_id, channel_id, channel_name, user_email, text, ts, ingested_at, extracted) "
                "VALUES (?,?,?,?,?,?,?,?,0)",
                (mid, req.company_id, ch["id"], ch.get("name", ""), user_email, m["text"], m["ts"], _now()),
            )
            ingested += 1
        channel_summaries.append({"id": ch["id"], "name": ch.get("name", ""), "member_count": len(member_emails)})

    conn.commit()
    conn.close()
    return {"channels_synced": len(channel_summaries), "messages_ingested": ingested, "channels": channel_summaries}


EXTRACT_CANDIDATES_SCHEMA = {
    "name": "submit_decision_candidates",
    "description": (
        "Submit any decision-shaped moments found in this Slack channel's recent messages. "
        "Return an empty list if none of the messages describe an actual decision being made."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short title, e.g. 'Switch analytics vendor to X'."},
                        "why": {"type": "array", "items": {"type": "string"}, "description": "1-3 bullets on why, grounded only in what was actually said."},
                        "alternatives": {"type": "array", "items": {"type": "string"}, "description": "Alternatives that were mentioned, if any."},
                        "risks": {"type": "array", "items": {"type": "string"}, "description": "Risks or concerns that were mentioned, if any."},
                    },
                    "required": ["title", "why", "alternatives", "risks"],
                },
            }
        },
        "required": ["candidates"],
    },
}


def extract_candidates_from_channel(channel_name, messages):
    client = Anthropic()
    transcript = "\n".join(f"[{m['user_email'] or 'unknown'}] {m['text']}" for m in messages)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "You scan a Slack channel's recent messages for moments where the team actually decided "
            "something — not casual chat, status updates, or open questions still being debated. Most "
            "batches this size will contain zero or one real decision; do not force it. Stay strictly "
            "grounded in what was actually written — never invent reasoning, names, or numbers that "
            "weren't in the messages."
        ),
        tools=[EXTRACT_CANDIDATES_SCHEMA],
        tool_choice={"type": "tool", "name": "submit_decision_candidates"},
        messages=[{"role": "user", "content": f"Channel: #{channel_name}\n\n{transcript}"}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_decision_candidates":
            return block.input.get("candidates", [])
    return []


class ExtractRequest(BaseModel):
    company_id: str
    personas: list = []  # [{email, level}, ...] supplied by the frontend, mirroring live.py's pattern


@router.post("/extract")
def extract(req: ExtractRequest, request: Request):
    require_session(token_from_request(request), company_id=req.company_id)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server.")

    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM slack_messages WHERE company_id=? AND extracted=0 ORDER BY channel_id, ts",
        (req.company_id,),
    ).fetchall()

    by_channel = {}
    for r in rows:
        by_channel.setdefault((r["channel_id"], r["channel_name"]), []).append(dict(r))

    if not by_channel:
        conn.close()
        return {"channels_scanned": 0, "candidates_created": 0}

    persona_by_email = {p.get("email", "").lower(): p for p in req.personas if p.get("email")}
    created = 0

    for (channel_id, channel_name), msgs in by_channel.items():
        try:
            candidates = extract_candidates_from_channel(channel_name, msgs)
        except Exception as e:
            conn.commit()
            conn.close()
            raise HTTPException(status_code=502, detail=f"Model call failed on #{channel_name}: {e}")

        member_emails = {(m["user_email"] or "").lower() for m in msgs if m.get("user_email")}
        member_levels = [persona_by_email[e]["level"] for e in member_emails if e in persona_by_email]
        suggested_visibility = min(member_levels) if member_levels else 1

        for c in candidates:
            cid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO slack_candidates (id, company_id, channel_name, title, why, alternatives, risks, "
                "suggested_visibility, source_message_ids, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,'pending',?)",
                (
                    cid,
                    req.company_id,
                    channel_name,
                    c["title"],
                    json.dumps(c.get("why", [])),
                    json.dumps(c.get("alternatives", [])),
                    json.dumps(c.get("risks", [])),
                    suggested_visibility,
                    json.dumps([m["id"] for m in msgs]),
                    _now(),
                ),
            )
            created += 1
        for m in msgs:
            conn.execute("UPDATE slack_messages SET extracted=1 WHERE id=?", (m["id"],))

    conn.commit()
    conn.close()
    return {"channels_scanned": len(by_channel), "candidates_created": created}


@router.get("/candidates")
def list_candidates(company_id: str, request: Request, token: Optional[str] = None):
    require_session(token_from_request(request, token), company_id=company_id)
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM slack_candidates WHERE company_id=? AND status='pending' ORDER BY created_at DESC",
        (company_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "channel_name": r["channel_name"],
            "title": r["title"],
            "why": json.loads(r["why"]),
            "alternatives": json.loads(r["alternatives"]),
            "risks": json.loads(r["risks"]),
            "suggested_visibility": r["suggested_visibility"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


class CandidateActionRequest(BaseModel):
    company_id: str


@router.post("/candidates/{candidate_id}/dismiss")
def dismiss_candidate(candidate_id: str, req: CandidateActionRequest, request: Request):
    require_session(token_from_request(request), company_id=req.company_id)
    conn = get_conn()
    conn.execute(
        "UPDATE slack_candidates SET status='dismissed' WHERE id=? AND company_id=?",
        (candidate_id, req.company_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.post("/candidates/{candidate_id}/confirm")
def confirm_candidate(candidate_id: str, req: CandidateActionRequest, request: Request):
    require_session(token_from_request(request), company_id=req.company_id)
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM slack_candidates WHERE id=? AND company_id=?", (candidate_id, req.company_id)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Candidate not found.")
    conn.execute("UPDATE slack_candidates SET status='confirmed' WHERE id=?", (candidate_id,))
    conn.commit()
    conn.close()
    return {
        "title": row["title"],
        "why": json.loads(row["why"]),
        "alternatives": json.loads(row["alternatives"]),
        "risks": json.loads(row["risks"]),
        "suggested_visibility": row["suggested_visibility"],
        "channel_name": row["channel_name"],
    }
