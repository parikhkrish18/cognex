# Deploying Cognex

This folder is one deployable service: FastAPI serves both the app (`GET /`)
and the API it calls (`POST /api/ask`, `POST /api/complete-story/extract`).
No separate frontend host needed.

## What's real here vs. what's still a prototype

**Real:** Ask Cognex and Complete the Story call the actual Claude API,
server-side, with permission filtering enforced in Python before anything
reaches the model (see `live.py`). This has been tested locally end to end,
including the fallback path — see the "not yet tested live" note below.

**Still a prototype:** company/roster/decision/goal data lives in each
browser tab's memory, not a database. Reload the page or redeploy the
service and a company's data resets to the seed demo. Login is a shared
demo password, not real auth. This was already true before this deploy —
see the in-app footer and `REFERENCE_BACKEND.md` for the full picture. The
straightforward next step, when you're ready, is swapping the frontend's
`COMPANIES` object and this backend's per-request payload for a real
database (Postgres is the obvious choice) behind a couple of CRUD
endpoints — everything else (the Claude integration, the permission
filtering) stays as-is.

**Not yet tested live:** nobody has run this against a real
`ANTHROPIC_API_KEY` yet — there wasn't one available while building it. The
code is written against the installed `anthropic` SDK's real method
signatures, and the "no key configured" path (503 → frontend falls back to
the local rule-based demo engine, clearly labeled "Offline demo mode") was
verified with Playwright. The live Claude call itself needs a real key to
confirm — do that first thing after deploying, before relying on it in
front of anyone. `python3 demo_cli.py` locally is the fastest way to check.

## 1. Push this to GitHub

```bash
cd cognex-app   # this folder
git init
git add -A
git commit -m "Cognex: merged frontend + live Claude backend"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Create the empty repo on GitHub first — no README/license/gitignore, so
there's nothing to conflict with this push.)

## 2. Deploy to Railway

Easiest path — Railway's dashboard, no CLI needed:

1. railway.app → **New Project** → **Deploy from GitHub repo** → pick the
   repo you just pushed.
2. Railway auto-detects this as a Python app (via `requirements.txt` +
   `Procfile`) and builds it. No Dockerfile needed.
3. Once it's deployed, open the service → **Variables** and add:
   - `ANTHROPIC_API_KEY` = your key from console.anthropic.com
     (**set it here, in Railway's UI — never paste it into a chat or commit
     it to the repo**; `.env` is already gitignored)
   - `CLAUDE_MODEL` (optional — omit to use the default in `.env.example`)
4. Railway gives you a `*.up.railway.app` URL immediately. Open it — you
   should see the Cognex login screen.

## 3. Point your domain at it

In the Railway service → **Settings** → **Networking** → **Custom Domain**,
add your domain (or a subdomain like `app.yourdomain.com`). Railway shows
you a CNAME record (or an A/ALIAS for a root domain) to add. Add that
record at wherever your domain is registered/managed. Propagation is
usually minutes, sometimes longer.

## 4. Sanity-check the live deploy

```bash
curl https://your-domain-or-railway-url/api/health
# should return {"ok":true,"api_key_configured":true}
```

Then open the site, sign in as any demo account, ask something in "Ask
Cognex" and confirm you see the **Live · Claude** badge (not "Offline demo
mode") on the answer — that's the real integration confirmed end to end.

## Rolling back / redeploying

Railway redeploys automatically on every push to the connected branch.
Since there's no database yet, a redeploy resets any demo companies created
through onboarding back to just the seeded "Cognex Labs" — expected for now,
see the persistence note above.
