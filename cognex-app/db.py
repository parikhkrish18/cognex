"""
Minimal persistence layer.

Everything else in this app is deliberately in-memory / client-side (see
REFERENCE_BACKEND.md and DEPLOY.md's persistence note) — that was a reasonable
simplification right up until a Slack OAuth token needed to survive a
redeploy. A token that resets every time Railway rebuilds the container is
useless, so this is the first real server-side persistence in the app:
SQLite, at the path in DB_PATH (a Railway volume in production, so it
survives restarts and redeploys — a bare filename locally, which is fine for
development).

Deliberately NOT stored here: Decision Memory, goals, personas, or anything
else that already lives in the frontend's COMPANIES object — that stays
exactly as it was. The Slack integration writes CANDIDATE decisions here for
a human to review; only once someone confirms one does it become a real
Decision Memory entry, written back into the frontend's own state exactly
like Complete the Story already does. This file stores the Slack connection
itself, raw ingested messages, and unconfirmed candidates — nothing that
bypasses "a human approves before company memory changes."
"""

import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "cognex_local.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS slack_connections (
            company_id TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            team_name TEXT,
            access_token TEXT NOT NULL,
            connected_by TEXT,
            connected_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS slack_messages (
            id TEXT PRIMARY KEY,
            company_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            channel_name TEXT,
            user_email TEXT,
            text TEXT,
            ts TEXT,
            ingested_at TEXT NOT NULL,
            extracted INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS slack_candidates (
            id TEXT PRIMARY KEY,
            company_id TEXT NOT NULL,
            channel_name TEXT,
            title TEXT,
            why TEXT,
            alternatives TEXT,
            risks TEXT,
            suggested_visibility INTEGER,
            source_message_ids TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()
