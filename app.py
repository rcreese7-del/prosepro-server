"""ProSe Pro API — single-file flattened build for manual upload.

Generated from server/app/*.py. Identical behavior; intra-package
imports resolved. Run with: uvicorn app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

# ============================================================
# module: sessions.py (flattened)
# ============================================================
"""JSON file session store for the drafting + adversary loop.

Sessions live under <server>/data/sessions/<session_id>.json and hold the
full round history: every draft version, every attack, every hardening
report, with timestamps and round numbers.

There is no git repo here; the data/ directory is runtime state and must
never be copied into a commit. Atomic writes (temp file + rename) avoid
half-written sessions if the server restarts mid-write.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "sessions"  # flat layout: app.py is at repo root


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str) -> Path:
    return DATA_DIR / f"{session_id}.json"


def _blank_session(session_id: str) -> dict:
    now = _utcnow()
    return {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "round": 0,
        "objective": None,
        "jurisdiction_state": None,
        "court": None,
        "language": None,
        "model": None,
        "events": [],
    }


def get_or_create(session_id: str | None) -> dict:
    """Return the stored session, or create a new one (with the given id,
    or a fresh uuid4 when session_id is None). Never returns None."""
    if session_id:
        path = _session_path(session_id)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt file -> start fresh under the same id
        session = _blank_session(session_id)
        save(session)
        return session
    return new_session()


def new_session() -> dict:
    session = _blank_session(str(uuid.uuid4()))
    save(session)
    return session


def save(session: dict) -> None:
    """Persist a session dict atomically."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session["session_id"])
    tmp = path.with_suffix(".json.tmp")
    session["updated_at"] = _utcnow()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def add_event(session: dict, event_type: str, round: int, data: dict) -> None:
    """Append one round-history event and persist."""
    session["events"].append(
        {
            "ts": _utcnow(),
            "type": event_type,
            "round": round,
            "data": data,
        }
    )
    session["round"] = round
    save(session)

# ============================================================
# module: extract.py (flattened)
# ============================================================
"""Text extraction for supported document types."""

import io


class ExtractionError(Exception):
    """Raised when a document yields no usable text."""


def _extract_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def _extract_docx(raw: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(raw))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    return "\n".join(parts)


def _extract_txt(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}


def extract_text(ext: str, raw: bytes) -> str:
    """Extract plain text from ``raw`` bytes of type ``ext``.

    Raises ExtractionError when nothing usable can be extracted
    (empty file, scanned-image PDF with no text layer, corrupt file, ...).
    """
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ExtractionError(f"Unsupported file type '{ext}'.")
    if not raw:
        raise ExtractionError("The uploaded file is empty.")
    try:
        text = extractor(raw)
    except Exception:
        raise ExtractionError(
            "Could not extract text from the document. "
            "It may be corrupt or in an unsupported format."
        )
    text = (text or "").strip()
    if not text:
        raise ExtractionError(
            "No readable text found in the document. "
            "Scanned images without a text layer are not supported."
        )
    return text

# ============================================================
# module: ai.py (flattened)
# ============================================================
"""Shared helpers for calling Anthropic from the new draft/adversary endpoints.

Auth — two paths, chosen at request time:
  1. ANTHROPIC_API_KEY env var set (e.g. Hugging Face Space): the key is
     sent directly as an ``x-api-key`` header to api.anthropic.com. The key
     is never logged, echoed, or written anywhere.
  2. Env var absent (local sandbox dev): the same surrogate-credential
     mechanism as the ~/workspace/skills/anthropic/bin/anthropic CLI — the
     helpers in /opt/hatch/skills/skill-creator/bin/dynamic_credentials.py
     attach an ``hsurr:*`` surrogate that authd swaps for the real key at
     request time. Only api.anthropic.com is ever contacted, and no API key
     is handled here, ever. (See ~/workspace/skills/anthropic/SKILL.md.)

Why not just shell out to the CLI here? The CLI hardcodes max_tokens=1024
with no flag to raise it, and draft/adversary JSON reliably exceeds that,
so outputs truncate mid-object and fail JSON parsing. This module keeps
the CLI's model default and request shape but raises max_tokens
(PROSEPRO_MAX_TOKENS, default 4096) for the long-form endpoints.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

CREDENTIAL_NAME = "custom.anthropic"
ALLOWED_HOSTS = ["api.anthropic.com"]
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 180

MAX_ADVERSARY_ROUNDS = 10

# Sandbox-only path for the surrogate helpers; imported lazily so the
# module loads fine on hosts (e.g. Hugging Face) where this file
# does not exist. Only used when ANTHROPIC_API_KEY is absent.
_SURROGATE_IMPORT_DIR = "/opt/hatch/skills/skill-creator/bin"


class AIServiceError(Exception):
    """Raised when the AI step fails (safe to surface as HTTP 502)."""


def resolve_model(explicit: str | None = None) -> str:
    """Model resolution order: explicit param -> PROSEPRO_MODEL env -> default."""
    return explicit or os.environ.get("PROSEPRO_MODEL") or DEFAULT_MODEL


def _resolve_max_tokens() -> int:
    try:
        return max(1024, int(os.environ.get("PROSEPRO_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS


def _api_key_from_env() -> str | None:
    """Return ANTHROPIC_API_KEY when set and non-blank, else None.

    The value is only ever passed as an HTTP header; it is never logged,
    echoed, or returned in any response.
    """
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _add_surrogate_auth(req: urllib.request.Request) -> None:
    """Attach the sandbox surrogate credential to an outgoing request.

    Lazily imported: only used on the local dev path (no ANTHROPIC_API_KEY),
    so the module imports cleanly on hosts where the helpers don't exist.
    """
    if _SURROGATE_IMPORT_DIR not in sys.path:
        sys.path.insert(0, _SURROGATE_IMPORT_DIR)
    from dynamic_credentials import add_surrogate_to_request

    add_surrogate_to_request(req, CREDENTIAL_NAME, allowed_hosts=ALLOWED_HOSTS)


def run_prompt(prompt: str, model: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Send one prompt to the Anthropic Messages API and return the reply text.

    Auth path is chosen at request time: ANTHROPIC_API_KEY when set
    (Hugging Face / production), sandbox surrogate auth otherwise.
    """
    data = {
        "model": resolve_model(model),
        "max_tokens": _resolve_max_tokens(),
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    api_key = _api_key_from_env()
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(
        MESSAGES_URL,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    if not api_key:
        _add_surrogate_auth(req)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise AIServiceError(
                "AI service is temporarily rate limited. Please try again shortly."
            )
        detail = e.read().decode("utf-8", "replace")[:300]
        raise AIServiceError(f"AI service returned HTTP {e.code}: {detail}")
    except TimeoutError:
        raise AIServiceError("AI service timed out. Please try again.")
    except Exception as e:
        raise AIServiceError(f"AI service request failed: {type(e).__name__}.")

    parts = []
    for block in payload.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    text = "".join(parts).strip()
    stop = payload.get("stop_reason")
    if not text:
        raise AIServiceError(
            "AI service returned an empty response. Please try again."
        )
    if stop == "max_tokens":
        raise AIServiceError(
            "AI service response was cut off (output limit). Try a shorter "
            "request or raise PROSEPRO_MAX_TOKENS."
        )
    return text


def strip_code_fences(raw: str) -> str:
    """Tolerate the model wrapping JSON in ```json ... ``` fences."""
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def parse_json_response(raw: str, required_keys: tuple[str, ...], what: str) -> dict:
    """Parse tolerant JSON from the model and check required keys are present."""
    try:
        payload = json.loads(strip_code_fences(raw))
    except json.JSONDecodeError:
        raise AIServiceError(
            f"AI service returned an unreadable response for {what}. Please try again."
        )
    if not isinstance(payload, dict):
        raise AIServiceError(
            f"AI service returned a non-JSON-object response for {what}."
        )
    for key in required_keys:
        if key not in payload:
            raise AIServiceError(
                f"AI service response for {what} was missing the key '{key}'."
            )
    return payload

# ============================================================
# module: analysis.py (flattened)
# ============================================================
"""AI analysis of extracted document text.

Two auth paths, chosen at request time:
  1. ANTHROPIC_API_KEY env var set (e.g. Hugging Face Space): call the
     Anthropic API directly via app.ai.run_prompt. The key is never logged.
  2. Env var absent (local sandbox dev): shell out to the CLI at
     ~/workspace/skills/anthropic/bin/anthropic, which uses surrogate auth
     on this machine — no API key is handled here, ever.
"""

import json
import os
import re
import subprocess


CLI_PATH = "/home/hatch/workspace/skills/anthropic/bin/anthropic"

# Model override: explicit argument wins, then PROSEPRO_MODEL, then the
# CLI's own default. The CLI exposes no max-tokens flag (it hardcodes
# max_tokens=1024), so outputs must stay compact by prompt design.
def _resolve_model(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("PROSEPRO_MODEL") or None

ANALYSIS_REQUIRED_KEYS = (
    "document_type",
    "parties",
    "key_claims",
    "deadlines",
    "weaknesses",
    "summary",
)


class AnalysisError(Exception):
    """Raised when the analysis step fails (safe to surface as HTTP 502)."""


def build_prompt(document_text: str) -> str:
    return (
        "You are a legal-document analyst assistant. You provide GENERAL "
        "LEGAL INFORMATION ONLY — you are not a lawyer, this is not legal "
        "advice, and no attorney-client relationship is created.\n\n"
        "Analyze the document below and return STRICT JSON with EXACTLY these "
        'keys and no others: "document_type" (string), "parties" (array of '
        'strings), "key_claims" (array of strings), "deadlines" (array of '
        'strings, empty array if none are mentioned), "weaknesses" (array of '
        'strings: missing elements, procedural risks, ambiguities), '
        '"summary" (string, 3-5 sentences). No markdown, no code fences, no '
        "commentary outside the JSON object.\n\n"
        "Rules: never reproduce copyrighted text verbatim; when referring to "
        "published sources, cite them by name and edition only. Keep values "
        "concise.\n\n"
        "DOCUMENT:\n"
        f"{document_text}"
    )


def _strip_code_fences(raw: str) -> str:
    """Tolerate the model wrapping JSON in ```json ... ``` fences."""
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def _validate_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise AnalysisError("Analysis service returned a non-JSON-object response.")
    for key in ANALYSIS_REQUIRED_KEYS:
        if key not in payload:
            raise AnalysisError(
                f"Analysis service response was missing the key '{key}'."
            )
    if not isinstance(payload["parties"], list):
        raise AnalysisError("Analysis service returned 'parties' as a non-list.")
    for key in ("key_claims", "deadlines", "weaknesses"):
        if not isinstance(payload[key], list):
            raise AnalysisError(
                f"Analysis service returned '{key}' as a non-list."
            )
    return {key: payload[key] for key in ANALYSIS_REQUIRED_KEYS}


def analyze_document_text(document_text: str, model: str | None = None) -> dict:
    """Run the analysis and return the validated result dict.

    `model` is optional; falls back to the PROSEPRO_MODEL env var, then the
    default model. Omitting it keeps /analyze behavior identical to before.
    """
    prompt = build_prompt(document_text)
    # Cloud path (e.g. Hugging Face Space): ANTHROPIC_API_KEY is set, so call
    # the Anthropic API directly instead of shelling out to the sandbox-only
    # CLI, which does not exist on the Space.
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            raw = run_prompt(prompt, model)
        except AIServiceError as e:
            raise AnalysisError(str(e))
        try:
            payload = json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError:
            raise AnalysisError(
                "Analysis service returned an unreadable response. Please try again."
            )
        return _validate_payload(payload)

    resolved = _resolve_model(model)
    cmd = [CLI_PATH, "message", prompt]
    if resolved:
        cmd += ["--model", resolved]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise AnalysisError("Analysis service is unavailable (helper not found).")
    except subprocess.TimeoutExpired:
        raise AnalysisError("Analysis service timed out. Please try again.")

    if proc.returncode != 0:
        detail = (proc.stdout or proc.stderr or "").strip()
        safe = detail[:300] if detail else "no output"
        raise AnalysisError(
            f"Analysis service returned an error (exit {proc.returncode}): {safe}"
        )

    try:
        payload = json.loads(_strip_code_fences(proc.stdout))
    except json.JSONDecodeError:
        raise AnalysisError(
            "Analysis service returned an unreadable response. Please try again."
        )

    return _validate_payload(payload)

# ============================================================
# module: draft.py (flattened)
# ============================================================
"""Draft generation for the ProSe Pro drafting + adversary loop.

The model is framed strictly as a legal drafting assistant giving GENERAL
INFORMATION ONLY — never a lawyer, never legal advice. Output is strict
JSON with a fixed contract so thin clients can render it.
"""


# Endpoint-facing alias; same shared service exception underneath.
DraftError = AIServiceError

DRAFT_REQUIRED_KEYS = ("title", "sections", "full_text", "needs_user_input")


def build_draft_prompt(
    analysis: dict,
    objective: str,
    jurisdiction_state: str,
    court: str,
    facts: str | None,
    language: str,
) -> str:
    facts_block = facts.strip() if facts and facts.strip() else "(none provided)"

    def _lines(values) -> str:
        if not values:
            return "(none)"
        return "\n".join(f"- {v}" for v in values)

    return (
        "You are a legal drafting assistant. You provide GENERAL LEGAL "
        "INFORMATION ONLY — you are not a lawyer, this is not legal advice, "
        "no attorney-client relationship is created, and the person using "
        "this output should have a licensed attorney review it before doing "
        "anything with it. Do not claim to be a lawyer. Do not file or send "
        "anything anywhere; this is a DRAFT only.\n\n"
        f"Draft the following document. OBJECTIVE: {objective}\n"
        f"Jurisdiction: {jurisdiction_state}. Court: {court}.\n"
        f"Language: write the entire draft in language code '{language}' "
        "(if the code is not 'en', still keep JSON keys in English).\n\n"
        "ANALYSIS OF THE SOURCE DOCUMENT:\n"
        f"Document type: {analysis.get('document_type', '')}\n"
        f"Parties: {_lines(analysis.get('parties', []))}\n"
        f"Key claims: {_lines(analysis.get('key_claims', []))}\n"
        f"Deadlines: {_lines(analysis.get('deadlines', []))}\n"
        f"Weaknesses: {_lines(analysis.get('weaknesses', []))}\n"
        f"Summary: {analysis.get('summary', '')}\n\n"
        f"USER-SUPPLIED FACTS:\n{facts_block}\n\n"
        "RULES:\n"
        "- Never reproduce copyrighted text verbatim. When referring to "
        "published legal sources, cite them by name and edition only "
        "(e.g. 'Black's Law Dictionary, 11th ed.') and link-free; never "
        "quote or reprint their content.\n"
        "- Do not invent case citations, docket numbers, statutes with "
        "specific section numbers, or quotations from authorities. If a "
        "point needs authority the user has not supplied, mark it "
        "needs_user_input instead of inventing one.\n"
        "- Where a fact needed for the draft is missing, write a visible "
        "marker in the body like [USER INPUT NEEDED: describe the missing "
        "fact] and list each missing fact in needs_user_input.\n"
        "- Keep the draft compact: 4-7 sections, each body 2-5 sentences. "
        "full_text is the whole document as one string with blank lines "
        "between sections.\n\n"
        "Return STRICT JSON with EXACTLY these keys and no others:\n"
        '"title" (string), "sections" (array of objects with "heading" and '
        '"body" strings), "full_text" (string), "needs_user_input" (array '
        'of strings, empty array if nothing is missing). No markdown, no '
        "code fences, no commentary outside the JSON object."
    )


def _validate_draft(payload: dict) -> dict:
    sections = payload["sections"]
    if not isinstance(sections, list) or not sections:
        raise AIServiceError("Draft response contained no sections.")
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            raise AIServiceError(f"Draft section {i} is not an object.")
        for key in ("heading", "body"):
            if key not in section or not isinstance(section[key], str):
                raise AIServiceError(
                    f"Draft section {i} is missing a string '{key}'."
                )
    if not isinstance(payload["full_text"], str) or not payload["full_text"].strip():
        raise AIServiceError("Draft response contained an empty full_text.")
    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise AIServiceError("Draft response contained an empty title.")
    needs = payload["needs_user_input"]
    if not isinstance(needs, list) or any(not isinstance(x, str) for x in needs):
        raise AIServiceError("Draft 'needs_user_input' must be an array of strings.")
    return {
        "title": payload["title"],
        "sections": [
            {"heading": s["heading"], "body": s["body"]} for s in sections
        ],
        "full_text": payload["full_text"],
        "needs_user_input": needs,
    }


def draft_document(
    analysis: dict,
    objective: str,
    jurisdiction_state: str,
    court: str,
    facts: str | None = None,
    language: str = "en",
    model: str | None = None,
) -> dict:
    """Generate a draft document; returns the validated draft dict."""
    prompt = build_draft_prompt(
        analysis, objective, jurisdiction_state, court, facts, language
    )
    raw = run_prompt(prompt, model=resolve_model(model))
    return _validate_draft(parse_json_response(raw, DRAFT_REQUIRED_KEYS, "draft"))

# ============================================================
# module: adversary.py (flattened)
# ============================================================
"""Adversary red-teaming and hardening for the ProSe Pro loop.

Two model calls:
  1. attack_draft() — role-plays the other side (opposing counsel,
     prosecutor, landlord, ...) attacking a draft: weakest arguments,
     missing elements, counter-authority, procedural defects, formatting,
     ambiguity.
  2. harden_draft() — revises the draft against a previous attack and
     produces a per-objection hardening report.

General-information framing is repeated in both prompts; nothing here is
legal advice.
"""


# Endpoint-facing alias; same shared service exception underneath.
AdversaryError = AIServiceError

ATTACK_KEYS = ("objections",)
REVISED_KEYS = ("revised_draft", "hardening_report")

SEVERITIES = {"high", "medium", "low"}
CATEGORIES = {
    "weakest argument",
    "missing element",
    "counter-authority",
    "procedural defect",
    "formatting",
    "ambiguity",
}
STATUSES = {"fixed", "needs_user_input", "not_applicable"}

GENERAL_FRAME = (
    "You are assisting with a legal drafting exercise. Everything you "
    "produce is GENERAL LEGAL INFORMATION ONLY — you are not a lawyer, "
    "this is not legal advice, no attorney-client relationship is created, "
    "and a licensed attorney should review any resulting document before "
    "it is used. This is a DRAFT exercise; nothing is filed or sent."
)


def build_attack_prompt(
    draft_text: str, adversary_role: str, jurisdiction_state: str, court: str
) -> str:
    return (
        f"{GENERAL_FRAME}\n\n"
        f"You are now role-playing as the ADVERSE party: {adversary_role}. "
        f"Your job is to attack the draft below as that adversary would in "
        f"{jurisdiction_state}, {court} — find every weakness so the draft "
        "can be strengthened.\n\n"
        "Attack across these categories where applicable: weakest argument, "
        "missing element, counter-authority, procedural defect, formatting, "
        "ambiguity.\n\n"
        "RULES:\n"
        "- Never reproduce copyrighted text verbatim; cite published sources "
        "by name and edition only.\n"
        "- Do not invent case citations, docket numbers, or quotations from "
        "authorities. If you would normally cite authority the user has not "
        "supplied, say so in the objection instead of inventing it.\n"
        "- Be specific: quote the exact draft language you are attacking.\n"
        "- Return 3-8 objections, ordered most damaging first. Keep each "
        "objection and argument to 2-4 sentences.\n\n"
        "DRAFT TO ATTACK:\n"
        f"{draft_text}\n\n"
        "Return STRICT JSON with EXACTLY this shape and no others:\n"
        '{"objections": [{"severity": "high|medium|low", "category": '
        '"weakest argument|missing element|counter-authority|procedural '
        'defect|formatting|ambiguity", "objection": "one-sentence statement", '
        '"argument": "the adversary\'s supporting argument"}]}. No markdown, '
        "no code fences, no commentary outside the JSON object."
    )


def _validate_attack(payload: dict) -> dict:
    objections = payload["objections"]
    if not isinstance(objections, list) or not objections:
        raise AIServiceError("Attack response contained no objections.")
    cleaned = []
    for i, o in enumerate(objections):
        if not isinstance(o, dict):
            raise AIServiceError(f"Attack objection {i} is not an object.")
        severity = o.get("severity")
        category = o.get("category")
        if severity not in SEVERITIES:
            raise AIServiceError(
                f"Attack objection {i} has invalid severity '{severity}'."
            )
        if category not in CATEGORIES:
            raise AIServiceError(
                f"Attack objection {i} has invalid category '{category}'."
            )
        for key in ("objection", "argument"):
            if key not in o or not isinstance(o[key], str) or not o[key].strip():
                raise AIServiceError(
                    f"Attack objection {i} is missing a non-empty '{key}'."
                )
        cleaned.append(
            {
                "severity": severity,
                "category": category,
                "objection": o["objection"],
                "argument": o["argument"],
            }
        )
    return {"objections": cleaned}


def attack_draft(
    draft_text: str,
    adversary_role: str,
    jurisdiction_state: str,
    court: str,
    model: str | None = None,
) -> dict:
    """Run one adversary round; returns the validated attack dict."""
    prompt = build_attack_prompt(draft_text, adversary_role, jurisdiction_state, court)
    raw = run_prompt(prompt, model=resolve_model(model))
    return _validate_attack(parse_json_response(raw, ATTACK_KEYS, "adversary attack"))


def build_harden_prompt(draft_text: str, attack: dict) -> str:
    lines = []
    for i, o in enumerate(attack.get("objections", []), start=1):
        lines.append(
            f"{i}. [{o.get('severity', '?')}/{o.get('category', '?')}] "
            f"{o.get('objection', '')} — {o.get('argument', '')}"
        )
    objections_block = "\n".join(lines) if lines else "(none)"

    return (
        f"{GENERAL_FRAME}\n\n"
        "You are now the drafter again. Revise the draft below to address "
        "EACH of the adversary's objections, one by one. Then report exactly "
        "what you changed.\n\n"
        "DRAFT TO REVISE:\n"
        f"{draft_text}\n\n"
        "ADVERSARY OBJECTIONS:\n"
        f"{objections_block}\n\n"
        "RULES:\n"
        "- Never reproduce copyrighted text verbatim; cite published sources "
        "by name and edition only.\n"
        "- Do not invent case citations, docket numbers, statutes with "
        "specific section numbers, or quotations. If an objection can only "
        "be fixed with a fact or authority the user has not supplied, keep "
        "the draft as is on that point, mark the hardening status "
        '"needs_user_input", and say exactly what the user must supply.\n'
        "- If an objection is genuinely not applicable to the draft, say so "
        'with status "not_applicable" and a one-sentence reason.\n'
        "- Keep the revised draft compact: same 4-7 section structure, each "
        "body 2-5 sentences. Keep each fix_applied note to 1-2 sentences.\n\n"
        "Return STRICT JSON with EXACTLY these keys and no others:\n"
        '"revised_draft" (object with "title" string, "sections" array of '
        '{"heading","body"} objects, "full_text" string), '
        '"hardening_report" (array with one entry per objection, each with '
        '"objection_summary" string, "severity" string, "fix_applied" '
        'string, "status" one of "fixed"|"needs_user_input"|"not_applicable"). '
        "No markdown, no code fences, no commentary outside the JSON object."
    )


def _validate_hardened(payload: dict) -> dict:
    revised = payload["revised_draft"]
    if not isinstance(revised, dict):
        raise AIServiceError("Hardening response has no revised_draft object.")
    sections = revised.get("sections")
    if not isinstance(sections, list) or not sections:
        raise AIServiceError("Revised draft contained no sections.")
    for i, s in enumerate(sections):
        if not isinstance(s, dict) or not isinstance(
            s.get("heading"), str
        ) or not isinstance(s.get("body"), str):
            raise AIServiceError(f"Revised draft section {i} is malformed.")
    if not isinstance(revised.get("full_text"), str) or not revised[
        "full_text"
    ].strip():
        raise AIServiceError("Revised draft has an empty full_text.")
    if not isinstance(revised.get("title"), str) or not revised["title"].strip():
        raise AIServiceError("Revised draft has an empty title.")

    report = payload["hardening_report"]
    if not isinstance(report, list):
        raise AIServiceError("Hardening report is not a list.")
    cleaned = []
    for i, r in enumerate(report):
        if not isinstance(r, dict):
            raise AIServiceError(f"Hardening report entry {i} is not an object.")
        if r.get("status") not in STATUSES:
            raise AIServiceError(
                f"Hardening report entry {i} has invalid status '{r.get('status')}'."
            )
        for key in ("objection_summary", "severity", "fix_applied"):
            if key not in r or not isinstance(r[key], str):
                raise AIServiceError(
                    f"Hardening report entry {i} is missing a string '{key}'."
                )
        cleaned.append(
            {
                "objection_summary": r["objection_summary"],
                "severity": r["severity"],
                "fix_applied": r["fix_applied"],
                "status": r["status"],
            }
        )
    return {
        "revised_draft": {
            "title": revised["title"],
            "sections": [
                {"heading": s["heading"], "body": s["body"]} for s in sections
            ],
            "full_text": revised["full_text"],
        },
        "hardening_report": cleaned,
    }


def harden_draft(
    draft_text: str, attack: dict, model: str | None = None
) -> dict:
    """Revise a draft against an attack; returns revised_draft + hardening_report."""
    prompt = build_harden_prompt(draft_text, attack)
    raw = run_prompt(prompt, model=resolve_model(model))
    return _validate_hardened(parse_json_response(raw, REVISED_KEYS, "hardening"))

# ============================================================
# module: main.py (flattened)
# ============================================================
"""ProSe Pro API server.

Exposes:
  GET  /health           - liveness probe
  POST /analyze          - upload a legal document (.pdf/.docx/.txt), get AI analysis
  POST /draft            - draft a motion/response from an analysis + objective
  POST /adversary/round  - adversary attacks the draft (round N+1, cap 10)
  POST /adversary/harden - revise the draft against the last attack + hardening report

Round history is persisted per session under server/data/ (see app/sessions.py).
All AI output is general legal information only — never legal advice.
"""
from fastapi import FastAPI, File, HTTPException, Request, UploadFile


MAX_CHARS = 60_000

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

DISCLAIMER = (
    "General legal information only — not legal advice. Consult a licensed attorney."
)

app = FastAPI(
    title="ProSe Pro API",
    description="Legal document analysis, drafting, and adversary red-teaming API (general information only).",
    version="0.2.0",
)


async def _json_body(request: Request) -> dict:
    """Parse the request body as a JSON object; 400 on anything else."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Request body must be valid JSON."
        )
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object."
        )
    return body


def _require_str(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=400,
            detail=f"'{key}' is required and must be a non-empty string.",
        )
    return value.strip()


def _optional_str(body: dict, key: str, default: str | None = None) -> str | None:
    value = body.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(
            status_code=400, detail=f"'{key}' must be a string when provided."
        )
    return value


def _require_draft_text(body: dict) -> str:
    value = body.get("draft_text")
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=422,
            detail="'draft_text' must be a non-empty string.",
        )
    return value


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze an uploaded legal document.

    Form field name must be "file". Accepts .pdf, .docx, .txt.
    """
    filename = (file.filename or "").strip()
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{filename or '(missing)'}'. "
                "Supported extensions: .pdf, .docx, .txt"
            ),
        )

    try:
        raw = file.file.read()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Could not read the uploaded file."
        )

    try:
        text = extract_text(ext, raw)
    except ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]

    try:
        result = analyze_document_text(text)
    except AnalysisError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "filename": filename,
        "document_type": result["document_type"],
        "parties": result["parties"],
        "key_claims": result["key_claims"],
        "deadlines": result["deadlines"],
        "weaknesses": result["weaknesses"],
        "summary": result["summary"],
        "disclaimer": DISCLAIMER,
        "truncated": truncated,
    }


_ANALYSIS_KEYS = (
    "document_type",
    "parties",
    "key_claims",
    "deadlines",
    "weaknesses",
    "summary",
)


def _require_analysis(body: dict) -> dict:
    analysis = body.get("analysis")
    if not isinstance(analysis, dict):
        raise HTTPException(
            status_code=400,
            detail="'analysis' is required and must be an object with keys: "
            + ", ".join(_ANALYSIS_KEYS),
        )
    missing = [k for k in _ANALYSIS_KEYS if k not in analysis]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"'analysis' is missing keys: {', '.join(missing)}.",
        )
    return {k: analysis[k] for k in _ANALYSIS_KEYS}


@app.post("/draft")
async def create_draft(request: Request) -> dict:
    """Draft a motion/response from a document analysis + objective.

    Body: {session_id?, analysis, objective, jurisdiction_state, court,
           facts?, language?, model?}
    A new session (round 0) is created when no session_id is given.
    """
    body = await _json_body(request)
    analysis = _require_analysis(body)
    objective = _require_str(body, "objective")
    jurisdiction_state = _require_str(body, "jurisdiction_state")
    court = _require_str(body, "court")
    facts = _optional_str(body, "facts")
    language = _optional_str(body, "language", default="en") or "en"
    model = _optional_str(body, "model")

    raw_session_id = body.get("session_id")
    if raw_session_id is not None and (
        not isinstance(raw_session_id, str) or not raw_session_id.strip()
    ):
        raise HTTPException(
            status_code=400, detail="'session_id' must be a non-empty string."
        )

    try:
        draft = draft_document(
            analysis=analysis,
            objective=objective,
            jurisdiction_state=jurisdiction_state,
            court=court,
            facts=facts,
            language=language,
            model=model,
        )
    except DraftError as e:
        raise HTTPException(status_code=502, detail=str(e))

    session = sessions.get_or_create(raw_session_id)
    session["objective"] = objective
    session["jurisdiction_state"] = jurisdiction_state
    session["court"] = court
    session["language"] = language
    session["model"] = model
    sessions.add_event(
        session,
        "draft",
        0,
        {
            "objective": objective,
            "draft": draft,
            "facts": facts,
        },
    )

    return {
        "session_id": session["session_id"],
        "draft": draft,
        "disclaimer": DISCLAIMER,
        "round": 0,
    }


@app.post("/adversary/round")
async def adversary_round(request: Request) -> dict:
    """Run one adversary attack round against a draft.

    Body: {session_id, draft_text, adversary_role, jurisdiction_state,
           court, model?}
    Rounds increment from the session's current round; exceeding
    MAX_ADVERSARY_ROUNDS (10) returns 400.
    """
    body = await _json_body(request)
    session_id = _require_str(body, "session_id")
    draft_text = _require_draft_text(body)
    adversary_role = _require_str(body, "adversary_role")
    jurisdiction_state = _require_str(body, "jurisdiction_state")
    court = _require_str(body, "court")
    model = _optional_str(body, "model")

    session = sessions.get_or_create(session_id)
    next_round = session["round"] + 1
    if next_round > MAX_ADVERSARY_ROUNDS:
        raise HTTPException(
            status_code=400, detail="maximum adversary rounds reached"
        )

    try:
        attack = attack_draft(
            draft_text=draft_text,
            adversary_role=adversary_role,
            jurisdiction_state=jurisdiction_state,
            court=court,
            model=model,
        )
    except AdversaryError as e:
        raise HTTPException(status_code=502, detail=str(e))

    session["jurisdiction_state"] = jurisdiction_state
    session["court"] = court
    session["model"] = model
    sessions.add_event(
        session,
        "attack",
        next_round,
        {"adversary_role": adversary_role, "attack": attack},
    )

    return {
        "session_id": session["session_id"],
        "round": next_round,
        "attack": attack,
        "disclaimer": DISCLAIMER,
    }


@app.post("/adversary/harden")
async def adversary_harden(request: Request) -> dict:
    """Revise a draft against the previous round's attack.

    Body: {session_id, draft_text, attack, model?}
    `attack` must be the attack object from an /adversary/round response
    (a dict with an "objections" list). Returns the revised draft plus a
    per-objection hardening report, at the session's current round number.
    """
    body = await _json_body(request)
    session_id = _require_str(body, "session_id")
    draft_text = _require_draft_text(body)
    model = _optional_str(body, "model")

    attack = body.get("attack")
    if not isinstance(attack, dict) or not isinstance(
        attack.get("objections"), list
    ):
        raise HTTPException(
            status_code=400,
            detail="'attack' must be an object with an 'objections' list "
            "(use the 'attack' object from an /adversary/round response).",
        )

    session = sessions.get_or_create(session_id)
    if session["round"] < 1:
        raise HTTPException(
            status_code=400,
            detail="No adversary round has been run for this session yet; "
            "call /adversary/round first.",
        )

    try:
        result = harden_draft(draft_text=draft_text, attack=attack, model=model)
    except AdversaryError as e:
        raise HTTPException(status_code=502, detail=str(e))

    sessions.add_event(
        session,
        "harden",
        session["round"],
        {
            "revised_draft": result["revised_draft"],
            "hardening_report": result["hardening_report"],
        },
    )

    return {
        "session_id": session["session_id"],
        "round": session["round"],
        "revised_draft": result["revised_draft"],
        "hardening_report": result["hardening_report"],
        "disclaimer": DISCLAIMER,
    }
