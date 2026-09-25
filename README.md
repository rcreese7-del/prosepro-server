# ProSe Pro API — Server (deploy bundle)

`app.py` is the whole FastAPI server in one file (analysis, drafting, and
adversary red-teaming), plus this `Dockerfile` and `requirements.txt`.
Upload all files in this folder to one GitHub repo, then deploy that repo
on Render.

## Deploy on Render (free tier, no credit card)

1. On **github.com**, create a new repo (e.g. `prosepro-server`) and upload
   every file from this folder via **Add file → Upload files**.
2. On **render.com**, sign up free (no card needed; GitHub sign-in is fine).
3. **New → Web Service** → connect the repo.
   - Runtime: **Docker**
   - Instance type: **Free**
4. Under **Environment**, add variable:
   - Key `ANTHROPIC_API_KEY`, value your Anthropic API key.
   Without this secret, analysis/drafting/adversary calls fail.
5. **Create Web Service** and wait for the deploy to finish. The public URL
   looks like `https://prosepro-server.onrender.com`.

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /analyze` — multipart form upload, field name **`file`**; accepts
  `.pdf`, `.docx`, `.txt`. Returns the analysis JSON contract.
- `POST /draft` — JSON body; drafts a motion/response from a document
  analysis plus an objective.
- `POST /adversary/round` — the adversary attacks the draft (cap: 10 rounds).
- `POST /adversary/harden` — revises the draft against the previous attack.

Notes:
- Render's free tier sleeps after 15 minutes of inactivity; the first
  request after a nap takes ~30 seconds to wake the service. Normal.
- The container disk is wiped on redeploys — per-session round history is
  temporary. Durable history needs an external store later.

General legal information only — not legal advice.
