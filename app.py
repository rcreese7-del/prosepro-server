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
DEFAULT_MODEL = "claude-sonnet-5-5"
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
# module: lang.py (flattened)
# ============================================================
"""Output-language support for the ProSe Pro API.

The client sends a language code with every AI call; the prompts then
instruct the model to respond fully in that language. The launch set is
fixed (the owner's Nov 24, 2026 language list). Unknown or missing codes
fall back to English so older clients keep working unchanged.

The not-a-lawyer disclaimer is part of every API response; the strings
below are careful human-grade translations of the canonical English
line, one per launch language — never machine-generated at runtime.
"""

# code -> (English name used in prompts, prompt-facing instruction name)
LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "tl": "Tagalog",
    "zh": "Mandarin Chinese (Simplified Chinese)",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "fr": "French",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
}

# Canonical disclaimer, carefully translated per launch language.
DISCLAIMERS: dict[str, str] = {
    "en": "General legal information only — not legal advice. Consult a licensed attorney.",
    "es": "Solo información legal general — no es asesoramiento legal. Consulte a un abogado con licencia.",
    "tl": "Pangkalahatang impormasyong legal lamang — hindi ito payong legal. Kumonsulta sa isang lisensyadong abogado.",
    "zh": "仅供一般法律信息参考 — 不构成法律建议。请咨询持牌律师。",
    "vi": "Chỉ là thông tin pháp lý chung — không phải tư vấn pháp lý. Hãy tham khảo ý kiến của luật sư có giấy phép hành nghề.",
    "ar": "معلومات قانونية عامة فقط — وليست استشارة قانونية. استشر محاميًا مرخصًا.",
    "fr": "Informations juridiques générales uniquement — ceci ne constitue pas un avis juridique. Consultez un avocat agréé.",
    "ko": "일반적인 법률 정보일 뿐 법률 자문이 아닙니다. 자격을 갖춘 변호사와 상담하십시오.",
    "ru": "Только общая юридическая информация — не является юридической консультацией. Обратитесь к лицензированному адвокату.",
    "pt": "Apenas informações jurídicas gerais — não constitui aconselhamento jurídico. Consulte um advogado licenciado.",
}


def normalize_language(raw: str | None) -> str:
    """Map any incoming value to a supported code; default English."""
    if not raw or not isinstance(raw, str):
        return "en"
    code = raw.strip().lower().replace("_", "-")
    if code in LANGUAGES:
        return code
    # Tolerate regional variants like "es-MX" / "pt-BR" / "zh-CN".
    base = code.split("-", 1)[0]
    return base if base in LANGUAGES else "en"


def language_name(code: str) -> str:
    return LANGUAGES.get(normalize_language(code), "English")


def disclaimer_for(code: str | None) -> str:
    return DISCLAIMERS[normalize_language(code)]


def language_instruction(code: str, *, json_note: bool = True) -> str:
    """Prompt block instructing the model to answer fully in the language.

    Empty for English (the prompts' native language). Enum/structural
    values (JSON keys, severity, category, status codes) always stay in
    English — only human-readable content is localized.
    """
    code = normalize_language(code)
    if code == "en":
        return ""
    name = LANGUAGES[code]
    base = (
        f"LANGUAGE: Write ALL human-readable content entirely in {name} — "
        f"every sentence a person will read must be in {name}, including "
        "any disclaimer line you write out. Keep proper nouns, party "
        "names, case captions, case numbers, and legal citations exactly "
        "as they appear in the source material. "
    )
    if json_note:
        base += (
            "JSON keys stay in English, and any fixed enum values named in "
            "the output contract (severity, category, status, change_type) "
            "stay in English exactly as specified."
        )
    return base

# ============================================================
# module: research.py (flattened)
# ============================================================
"""Live legal research layer for ProSe Pro.

Retrieves real sources for a query from keyless/public services:
  - CourtListener search API   -> case law (opinions) with citations.
    The v4 API requires a (free) account token; when the
    COURTLISTENER_API_TOKEN env var is set it is sent as an auth header,
    otherwise this source simply contributes nothing.
  - Cornell LII Wex            -> legal definitions/encyclopedia entries
    (the free online stand-in for a law dictionary), fetched by term.
  - eCFR search API            -> Code of Federal Regulations full text.
  - Federal Register API       -> federal rules, notices, and actions.
  - General web fallback       -> DuckDuckGo HTML, then Bing HTML
    (state legislature sites, court rules, ethics codes, treatises —
    the broad "on record" tier).

Every fetcher is best-effort: failures return whatever was retrieved
(possibly nothing) and NEVER raise, so callers fall back to model
knowledge gracefully. Results carry source, title, citation, URL, and
a short snippet — sources are cited by name/link; snippets are brief
search excerpts only, never long reproductions.
"""

import base64
import html as _html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_TIMEOUT = 12
_MAX_SNIPPET = 400


class ResearchError(Exception):
    """Raised only for caller errors (e.g. empty query)."""


def _fetch(url: str, timeout: int = _TIMEOUT) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _fetch_json(url: str, headers: dict | None = None) -> dict | None:
    hdrs = {"User-Agent": _UA}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _result(source: str, title: str, citation: str, url: str, snippet: str) -> dict:
    return {
        "source": source,
        "title": _clean(title)[:200],
        "citation": _clean(citation)[:200],
        "url": url,
        "snippet": _clean(snippet)[:_MAX_SNIPPET],
    }


# ---- Case law: CourtListener API -------------------------------------------

def search_courtlistener(query: str, jurisdiction: str | None = None, limit: int = 5) -> list[dict]:
    q = query if not jurisdiction else f"{query} {jurisdiction}"
    url = (
        "https://www.courtlistener.com/api/rest/v4/search/?"
        + urllib.parse.urlencode({"q": q, "type": "o"})
    )
    headers = {}
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    if token:
        headers["Authorization"] = f"Token {token}"
    payload = _fetch_json(url, headers=headers)
    if not payload:
        return []
    out: list[dict] = []
    for item in (payload.get("results") or [])[:limit]:
        case_name = item.get("caseName") or item.get("case_name") or "Untitled case"
        cites = item.get("citation") or []
        citation = ", ".join(cites) if isinstance(cites, list) else str(cites)
        if not citation:
            court = item.get("court") or ""
            date = (item.get("dateFiled") or "")[:4]
            citation = " ".join(p for p in (court, date) if p)
        path = item.get("absolute_url") or ""
        link = (
            "https://www.courtlistener.com" + path
            if path.startswith("/")
            else (path or "https://www.courtlistener.com")
        )
        out.append(_result("CourtListener", case_name, citation, link, item.get("snippet") or ""))
    return out


# ---- Definitions: Cornell LII Wex ------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "what",
    "is", "are", "how", "does", "do", "under", "law", "legal", "illinois",
    "indiana", "federal", "state", "court", "rule", "rules",
}


def search_lii(query: str, limit: int = 3) -> list[dict]:
    """Look up Wex entries on Cornell LII by term slug.

    Wex pages live at law.cornell.edu/wex/<term>; try the full query as a
    slug first (for short queries), then individual significant words.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS]
    candidates: list[str] = []
    if 1 <= len(words) <= 3:
        candidates.append("_".join(words))
    candidates += words[:3]
    out: list[dict] = []
    seen: set[str] = set()
    for slug in candidates:
        if slug in seen or len(slug) < 3:
            continue
        seen.add(slug)
        url = f"https://www.law.cornell.edu/wex/{slug}"
        page = _fetch(url)
        if not page:
            continue
        m = re.search(r"<title>(.*?)</title>", page, re.DOTALL)
        title = _clean(m.group(1)).split("|")[0].strip() if m else slug.replace("_", " ")
        # First substantive paragraph as the definition excerpt.
        snippet = ""
        for p in re.finditer(r"<p[^>]*>(.*?)</p>", page, re.DOTALL):
            text = _clean(p.group(1))
            if len(text) >= 60:
                snippet = text
                break
        out.append(
            _result(
                "Cornell LII — Wex",
                f"{title} (Wex definition)",
                "Wex, Cornell Legal Information Institute",
                url,
                snippet,
            )
        )
        if len(out) >= limit:
            break
    return out


# ---- Regulations: eCFR ------------------------------------------------------

def search_ecfr(query: str, limit: int = 4) -> list[dict]:
    url = (
        "https://www.ecfr.gov/api/search/v1/results?"
        + urllib.parse.urlencode({"query": query, "per_page": limit, "page": 1})
    )
    payload = _fetch_json(url)
    if not payload:
        return []
    out: list[dict] = []
    for item in (payload.get("results") or [])[:limit]:
        hier = item.get("hierarchy") or {}
        heads = item.get("headings") or {}
        title_no = hier.get("title")
        section = hier.get("section")
        part = hier.get("part")
        if section:
            citation = f"{title_no} CFR § {section}"
            link = f"https://www.ecfr.gov/current/title-{title_no}/section-{section}"
        elif part:
            citation = f"{title_no} CFR part {part}"
            link = f"https://www.ecfr.gov/current/title-{title_no}/part-{part}"
        else:
            citation = f"Title {title_no} CFR"
            link = "https://www.ecfr.gov"
        name = heads.get("section") or heads.get("part") or citation
        heading = heads.get("title") or ""
        title = f"{name} — {heading}".strip(" —") if heading else str(name)
        out.append(_result("eCFR", title, citation, link, item.get("full_text_excerpt") or ""))
    return out


# ---- Federal rules/actions: Federal Register --------------------------------

def search_federal_register(query: str, limit: int = 3) -> list[dict]:
    params = {
        "conditions[term]": query,
        "per_page": limit,
        "fields[]": ["title", "citation", "html_url", "abstract", "document_number"],
    }
    url = "https://www.federalregister.gov/api/v1/documents.json?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url)
    if not payload:
        return []
    out: list[dict] = []
    for item in (payload.get("results") or [])[:limit]:
        out.append(
            _result(
                "Federal Register",
                item.get("title") or "Federal Register document",
                item.get("citation") or item.get("document_number") or "",
                item.get("html_url") or "https://www.federalregister.gov",
                item.get("abstract") or "",
            )
        )
    return out


# ---- General web fallback: DuckDuckGo HTML, then Bing HTML ------------------

def _search_ddg(query: str, limit: int) -> list[dict]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    page = _fetch(url)
    if not page:
        return []
    out: list[dict] = []
    pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for m in pattern.finditer(page):
        href, title, snippet = _html.unescape(m.group(1)), _clean(m.group(2)), m.group(3)
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if "uddg" in qs:
            href = qs["uddg"][0]
        if not title:
            continue
        host = urllib.parse.urlparse(href).netloc
        out.append(_result(f"Web — {host}" if host else "Web", title, host, href, snippet))
        if len(out) >= limit:
            break
    return out


def _unwrap_bing(href: str) -> str:
    href = _html.unescape(href)
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    u = (qs.get("u") or [""])[0]
    if u.startswith("a1"):
        s = u[2:]
        s += "=" * (-len(s) % 4)
        try:
            return base64.urlsafe_b64decode(s).decode("utf-8", "replace")
        except Exception:
            return href
    return href


def _search_bing(query: str, limit: int) -> list[dict]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "count": 10})
    page = _fetch(url)
    if not page:
        return []
    out: list[dict] = []
    for chunk in page.split('<li class="b_algo"')[1:]:
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', chunk, re.DOTALL)
        if not m:
            continue
        href = _unwrap_bing(m.group(1))
        title = _clean(m.group(2))
        host = urllib.parse.urlparse(href).netloc
        if not title or "bing.com" in host:
            continue
        p = re.search(r"<p[^>]*>(.*?)</p>", chunk, re.DOTALL)
        out.append(
            _result(
                f"Web — {host}" if host else "Web",
                title,
                host,
                href,
                p.group(1) if p else "",
            )
        )
        if len(out) >= limit:
            break
    return out


def search_web(query: str, jurisdiction: str | None = None, limit: int = 5) -> list[dict]:
    q = query if not jurisdiction else f"{query} {jurisdiction}"
    return _search_ddg(q, limit) or _search_bing(q, limit)


# ---- Combined research -------------------------------------------------------

def research(query: str, jurisdiction: str | None = None) -> dict:
    """Run all sources and return a combined, de-duplicated result set."""
    query = (query or "").strip()
    if not query:
        raise ResearchError("A non-empty 'query' is required.")
    jurisdiction = (jurisdiction or "").strip() or None

    results: list[dict] = []
    results += search_courtlistener(query, jurisdiction)
    results += search_lii(query)
    results += search_ecfr(query)
    results += search_federal_register(query)
    results += search_web(query, jurisdiction)

    seen: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)

    return {
        "query": query,
        "jurisdiction": jurisdiction,
        "results": deduped,
        "sources_ok": bool(deduped),
    }


def research_context(query: str, jurisdiction: str | None = None, max_items: int = 6) -> str:
    """Compact retrieved-context block for injection into AI prompts.

    Returns an empty string when retrieval finds nothing, so prompts fall
    back to model knowledge unchanged.
    """
    try:
        data = research(query, jurisdiction)
    except Exception:
        return ""
    items = data["results"][:max_items]
    if not items:
        return ""
    lines = [
        "RETRIEVED RESEARCH (live sources — prefer these over memory when "
        "they are on point; cite by name/citation only, never quote long "
        "passages):"
    ]
    for r in items:
        cite = f" [{r['citation']}]" if r["citation"] else ""
        lines.append(f"- ({r['source']}) {r['title']}{cite} — {r['url']}")
        if r["snippet"]:
            lines.append(f"  {r['snippet']}")
    return "\n".join(lines)

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


def build_prompt(
    document_text: str, research_context: str = "", language: str = "en"
) -> str:
    research_block = f"\n\n{research_context}\n" if research_context else ""
    lang_block = language_instruction(language)
    lang_prefix = f"{lang_block}\n\n" if lang_block else ""
    return (
        f"{lang_prefix}"
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
        f"{research_block}"
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


def analyze_document_text(
    document_text: str,
    model: str | None = None,
    research_context: str = "",
    language: str = "en",
) -> dict:
    """Run the analysis and return the validated result dict.

    `model` is optional; falls back to the PROSEPRO_MODEL env var, then the
    default model. Omitting it keeps /analyze behavior identical to before.
    `research_context` (from app.research) is appended to the prompt when
    the research layer retrieved live sources; empty keeps the old prompt.
    `language` is a launch-set code (app.lang); non-English makes every
    JSON string value come back in that language.
    """
    prompt = build_prompt(document_text, research_context, language)
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

import re


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
    research_context: str = "",
) -> str:
    facts_block = facts.strip() if facts and facts.strip() else "(none provided)"
    research_block = (
        f"\n{research_context}\n"
        "- Authorities listed in the RETRIEVED RESEARCH above MAY be cited "
        "by the name and citation exactly as given there. Do not extend, "
        "re-number, or embellish them.\n"
        if research_context
        else ""
    )

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
        f"Language: write the entire draft in {language_name(language)} "
        f"(language code '{normalize_language(language)}'). Write every "
        "human-readable string in that language — title, headings, bodies, "
        "full_text, and needs_user_input — and keep JSON keys in English. "
        "Party names, captions, case numbers, and citations stay exactly "
        "as given in the inputs.\n\n"
        "ANALYSIS OF THE SOURCE DOCUMENT:\n"
        f"Document type: {analysis.get('document_type', '')}\n"
        f"Parties: {_lines(analysis.get('parties', []))}\n"
        f"Key claims: {_lines(analysis.get('key_claims', []))}\n"
        f"Deadlines: {_lines(analysis.get('deadlines', []))}\n"
        f"Weaknesses: {_lines(analysis.get('weaknesses', []))}\n"
        f"Summary: {analysis.get('summary', '')}\n\n"
        f"USER-SUPPLIED FACTS:\n{facts_block}\n"
        f"{research_block}\n"
        "RULES:\n"
        "- Never reproduce copyrighted text verbatim. When referring to "
        "published legal sources, cite them by name and edition only "
        "(e.g. 'Black's Law Dictionary, 11th ed.') and link-free; never "
        "quote or reprint their content.\n"
        "- Do not invent case citations, docket numbers, statutes with "
        "specific section numbers, or quotations from authorities. If a "
        "point needs authority the user has not supplied, mark it "
        "needs_user_input instead of inventing one.\n"
        "- CARRY KNOWN FACTS VERBATIM: any case or cause number, caption "
        "(the court name and the party names as styled in the source), "
        "court, and party names that appear in the ANALYSIS above or in "
        "the USER-SUPPLIED FACTS must be carried into the draft exactly "
        "as given. Never ask the user for a fact that already appears in "
        "the analysis or the user-supplied facts, and never list such a "
        "fact in needs_user_input.\n"
        "- Open full_text with a caption block: the court, the party "
        "names as styled in the source, and the case/cause number "
        "verbatim when one appears in the inputs. Only if no case/cause "
        "number appears anywhere in the inputs, put a marker like "
        "[USER INPUT NEEDED: case/cause number] in the caption and list "
        "it ONCE in needs_user_input.\n"
        "- ADDRESS THE FIRST FLAGGED WEAKNESS: the draft must deal with "
        "the first item in the analysis Weaknesses list — cure it, "
        "concede and excuse it, or argue around it in the body. If it "
        "cannot be addressed with the facts supplied, add ONE entry to "
        "needs_user_input explaining exactly what is needed and why; "
        "never silently ignore it.\n"
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


_CASE_NUMBER_RE = re.compile(
    r"\b(?:case|cause)\s*(?:number|no\.?|#)\b", re.IGNORECASE
)
_CAPTION_SUFFIX_RE = re.compile(
    r"\s+(?:for|in)\s+(?:the\s+)?caption$", re.IGNORECASE
)


def _need_key(text: str) -> str:
    """Canonical key for de-duplicating needs_user_input entries (D3).

    Normalizes case, whitespace and trailing punctuation, and collapses
    the case-number synonym family ("case number", "cause number",
    "case no.", "cause no", ...) into one canonical item — so the user
    is never asked twice for the same fact.
    """
    key = re.sub(r"\s+", " ", text.strip().lower()).strip(" .")
    key = _CASE_NUMBER_RE.sub("case/cause number", key)
    key = _CAPTION_SUFFIX_RE.sub("", key)
    return key.strip(" .")


def _dedupe_needs(needs: list[str]) -> list[str]:
    """Drop duplicate needs_user_input entries, keeping first-seen order
    and the first entry's original wording."""
    seen: set[str] = set()
    out: list[str] = []
    for item in needs:
        key = _need_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
        "needs_user_input": _dedupe_needs(needs),
    }


def draft_document(
    analysis: dict,
    objective: str,
    jurisdiction_state: str,
    court: str,
    facts: str | None = None,
    language: str = "en",
    model: str | None = None,
    research_context: str = "",
) -> dict:
    """Generate a draft document; returns the validated draft dict."""
    prompt = build_draft_prompt(
        analysis, objective, jurisdiction_state, court, facts, language,
        research_context,
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

import re


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

# The ONLY kinds of change the harden step is allowed to make. There is
# deliberately no "new fact" category: a change is a restructure of existing
# material, a citation/argument-framing change, or a placeholder for a fact
# the user must supply.
CHANGE_TYPES = {"RESTRUCTURE", "CITATION", "PLACEHOLDER"}

DISCLAIMER_LINE = (
    "General legal information only — not legal advice. Consult a licensed attorney."
)

VERIFY_PLACEHOLDER = "[VERIFY: ___]"
REMOVED_CLAIM = (
    "[VERIFY: a statement was removed here because it asserted a fact that "
    "does not appear in your supplied draft. Add it back only if you can "
    "confirm it yourself.]"
)

GENERAL_FRAME = (
    "You are assisting with a legal drafting exercise. Everything you "
    "produce is GENERAL LEGAL INFORMATION ONLY — you are not a lawyer, "
    "this is not legal advice, no attorney-client relationship is created, "
    "and a licensed attorney should review any resulting document before "
    "it is used. This is a DRAFT exercise; nothing is filed or sent."
)


def build_attack_prompt(
    draft_text: str,
    adversary_role: str,
    jurisdiction_state: str,
    court: str,
    language: str = "en",
) -> str:
    lang_block = language_instruction(language)
    lang_prefix = f"{lang_block}\n\n" if lang_block else ""
    return (
        f"{lang_prefix}"
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
    language: str = "en",
) -> dict:
    """Run one adversary round; returns the validated attack dict."""
    prompt = build_attack_prompt(
        draft_text, adversary_role, jurisdiction_state, court, language
    )
    raw = run_prompt(prompt, model=resolve_model(model))
    return _validate_attack(parse_json_response(raw, ATTACK_KEYS, "adversary attack"))


def build_harden_prompt(draft_text: str, attack: dict, language: str = "en") -> str:
    lines = []
    for i, o in enumerate(attack.get("objections", []), start=1):
        lines.append(
            f"{i}. [{o.get('severity', '?')}/{o.get('category', '?')}] "
            f"{o.get('objection', '')} — {o.get('argument', '')}"
        )
    objections_block = "\n".join(lines) if lines else "(none)"

    lang_block = language_instruction(language)
    lang_prefix = f"{lang_block}\n\n" if lang_block else ""
    return (
        f"{lang_prefix}"
        f"{GENERAL_FRAME}\n\n"
        "You are now the drafter again. Revise the draft below to address "
        "EACH of the adversary's objections, one by one. Then report exactly "
        "what you changed.\n\n"
        "DRAFT TO REVISE:\n"
        f"{draft_text}\n\n"
        "ADVERSARY OBJECTIONS:\n"
        f"{objections_block}\n\n"
        "HARD RULES — NO NEW FACTS. Breaking any of these is a failure:\n"
        "- You may ONLY do three kinds of change: (a) RESTRUCTURE — "
        "reorder, reword, or reframe material that is already in the draft "
        "or the objections; (b) CITATION — add or correct legal citations "
        "and argument framing; (c) PLACEHOLDER — insert a clearly marked "
        "placeholder like [VERIFY: what the user must confirm] where an "
        "objection needs a fact that is NOT in the draft or objections.\n"
        "- NEVER introduce a factual assertion that does not appear in the "
        "draft or the objections: no new dates or deadlines, no page, "
        "document, video, or other counts, no new events, no completed "
        "actions (nothing 'has been reviewed', 'was served', 'was filed', "
        "'was mailed', or 'was received' unless the draft already says "
        "so), no agreements, and no waivers or concessions of any kind.\n"
        "- NEVER complete a certificate of service: no service method and "
        "no service date unless it appears verbatim in the draft. If it "
        "does not, leave a [VERIFY: ...] placeholder and mark the entry "
        "needs_user_input.\n"
        "- NEVER add a waiver (for example of speedy-trial time) or any "
        "other substantive concession, even if an objection suggests one. "
        "If an objection raises it, use a placeholder and needs_user_input "
        "so the user decides.\n"
        "- If an objection can only be fixed with a fact or authority the "
        "user has not supplied, insert a [VERIFY: ...] placeholder at that "
        'point in the draft, mark the status "needs_user_input", and say '
        "exactly what the user must supply.\n"
        "- Never reproduce copyrighted text verbatim; cite published "
        "sources by name and edition only. Do not invent case citations, "
        "docket numbers, statutes with specific section numbers, or "
        "quotations.\n"
        "- If an objection is genuinely not applicable to the draft, say "
        'so with status "not_applicable" and a one-sentence reason; make '
        "no change for it.\n"
        "- Keep the revised draft compact: same 4-7 section structure, "
        "each body 2-5 sentences. Keep each fix_applied note to 1-2 "
        "sentences.\n\n"
        "Return STRICT JSON with EXACTLY these keys and no others:\n"
        '"revised_draft" (object with "title" string, "sections" array '
        'of {"heading","body"} objects, "full_text" string), '
        '"hardening_report" (array with one entry per objection, each '
        'with "objection_summary" string, "severity" string, '
        '"fix_applied" string, "status" one of '
        '"fixed"|"needs_user_input"|"not_applicable", and "change_type" '
        'one of "RESTRUCTURE"|"CITATION"|"PLACEHOLDER" — PLACEHOLDER '
        "whenever a [VERIFY: ...] placeholder was inserted, CITATION when "
        "the change is a citation or legal-framing change, RESTRUCTURE "
        "otherwise). No markdown, no code fences, no commentary outside "
        "the JSON object."
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
        change_type = r.get("change_type")
        if change_type not in CHANGE_TYPES:
            # Tolerate older-shaped responses: derive the only sensible tag.
            change_type = (
                "PLACEHOLDER"
                if r["status"] == "needs_user_input"
                else "RESTRUCTURE"
            )
        cleaned.append(
            {
                "objection_summary": r["objection_summary"],
                "severity": r["severity"],
                "fix_applied": r["fix_applied"],
                "status": r["status"],
                "change_type": change_type,
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


# ---------------------------------------------------------------------------
# Deterministic no-new-facts guard (hardening only).
#
# The harden prompt forbids new facts, but a prompt is a request, not an
# enforcement. This guard runs on every hardened output and removes any
# concrete specific that does not appear in the inputs:
#   * dates and quantities (pages, days, counts, money, percents) must
#     appear in the original draft OR the adversary objections;
#   * completed-action claims ("has reviewed", "was served"), waiver
#     claims, and certificate-of-service methods must appear in the
#     original draft itself — an objection merely raising a topic (e.g.
#     asking whether the user will waive speedy-trial time) never licenses
#     writing that claim into the document.
# Anything unsupported is replaced with a [VERIFY: ...] placeholder and
# listed in the returned "unverified_additions" so the user can see
# exactly what was stripped and decide for themselves.
# ---------------------------------------------------------------------------

_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
_MONTH_ALT = "|".join(_MONTH_NUM)

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_UNIT_ALT = (
    r"(?:pages?|videos?|recordings?|documents?|exhibits?|files?|copies|"
    r"days?|weeks?|months?|years?|hours?|minutes?)"
)
_QTY_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?|" + "|".join(_WORD_NUM) + r")\s*-?\s*("
    + _UNIT_ALT + r")\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_PCT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")

# Verbs that assert a completed real-world action when used in a
# perfect/passive construction ("has reviewed", "was served").
_CLAIM_VERBS = {
    "reviewed", "read", "examined", "inspected", "served", "filed",
    "mailed", "emailed", "delivered", "sent", "received", "provided",
    "submitted", "obtained", "prepared", "signed", "executed",
}
_PASSIVE_CLAIM_VERBS = {
    "served", "mailed", "emailed", "delivered", "filed", "sent",
    "submitted",
}
_ACTIVE_CLAIM_RE = re.compile(
    r"\b(?:has|have|had)\s+(?:not\s+)?([a-z]+)\b", re.IGNORECASE
)
_PASSIVE_CLAIM_RE = re.compile(
    r"\b(?:was|were|been|be)\s+([a-z]+)\b", re.IGNORECASE
)
_WAIVER_RE = re.compile(r"\bwaiv\w*\b", re.IGNORECASE)
_CERT_RE = re.compile(
    r"(certificate of service|hereby certify|i certify)", re.IGNORECASE
)
_CERT_METHODS = (
    "electronic mail", "e-mail", "email", "mailed", "mail", "hand-deliver",
    "hand deliver", "personal delivery", "e-filed", "efiled", "facsimile",
    "overnight courier",
)


def _bracket_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*\]", text)]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _iter_dates(text: str):
    """Yield (start, end, md, my, triple) for each date phrase in text.

    md = (month, day); my = (month, year); triple = (month, day, year).
    Unused components are None.
    """
    found = []
    for m in re.finditer(
        rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?",
        text, re.IGNORECASE,
    ):
        mon = _MONTH_NUM[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None
        found.append((m.start(), m.end(), (mon, day), None,
                      (mon, day, year) if year else None))
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text):
        mon, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        found.append((m.start(), m.end(), (mon, day), None, (mon, day, year)))
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        year, mon, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        found.append((m.start(), m.end(), (mon, day), None, (mon, day, year)))
    spans = [(s, e) for s, e, *_ in found]
    for m in re.finditer(rf"\b({_MONTH_ALT})\s+(\d{{4}})\b", text, re.IGNORECASE):
        if _in_spans(m.start(), spans):
            continue
        mon = _MONTH_NUM[m.group(1).lower()]
        found.append((m.start(), m.end(), None, (mon, int(m.group(2))), None))
    return sorted(found)


def _qty_key(num_raw: str, unit_raw: str) -> tuple:
    unit = unit_raw.lower()
    if unit.endswith("ies"):
        unit = unit[:-3] + "y"
    elif unit.endswith("s"):
        unit = unit[:-1]
    num_raw = num_raw.lower().replace(",", "")
    if num_raw in _WORD_NUM:
        return (_WORD_NUM[num_raw], unit)
    try:
        value = float(num_raw)
    except ValueError:
        return (num_raw, unit)
    return (int(value) if value == int(value) else value, unit)


def _support_sets(support_text: str, draft_text: str) -> dict:
    dates = _iter_dates(support_text)
    return {
        "date_md": {d[2] for d in dates if d[2]},
        "date_my": {d[3] for d in dates if d[3]},
        "date_triple": {d[4] for d in dates if d[4]},
        "qty": {
            _qty_key(m.group(1), m.group(2))
            for m in _QTY_RE.finditer(support_text)
        },
        "money": {
            m.group(0).replace(" ", "").replace(",", "")
            for m in _MONEY_RE.finditer(support_text)
        },
        "pct": {
            m.group(0).replace(" ", "")
            for m in _PCT_RE.finditer(support_text)
        },
        "draft_claim_verbs": {
            m.group(1).lower()
            for m in _ACTIVE_CLAIM_RE.finditer(draft_text)
            if m.group(1).lower() in _CLAIM_VERBS
        } | {
            m.group(1).lower()
            for m in _PASSIVE_CLAIM_RE.finditer(draft_text)
            if m.group(1).lower() in _PASSIVE_CLAIM_VERBS
        },
        "draft_has_waiver": bool(_WAIVER_RE.search(draft_text)),
        "draft_lower": draft_text.lower(),
    }


def _guard_phrases(text: str, sup: dict, flags: list[str]) -> str:
    """Replace unsupported dates/quantities/money/percents with placeholders."""
    spans = _bracket_spans(text)
    edits = []  # (start, end, kind, original)

    for start, end, md, my, triple in _iter_dates(text):
        if _in_spans(start, spans):
            continue
        if triple is not None:
            ok = triple in sup["date_triple"]
        elif md is not None:
            ok = md in sup["date_md"]
        else:
            ok = my in sup["date_my"]
        if not ok:
            edits.append((start, end, "date", text[start:end]))

    for m in _QTY_RE.finditer(text):
        if _in_spans(m.start(), spans):
            continue
        if _qty_key(m.group(1), m.group(2)) not in sup["qty"]:
            edits.append((m.start(), m.end(), "quantity", m.group(0)))

    for m in _MONEY_RE.finditer(text):
        if _in_spans(m.start(), spans):
            continue
        if m.group(0).replace(" ", "").replace(",", "") not in sup["money"]:
            edits.append((m.start(), m.end(), "amount", m.group(0)))

    for m in _PCT_RE.finditer(text):
        if _in_spans(m.start(), spans):
            continue
        if m.group(0).replace(" ", "") not in sup["pct"]:
            edits.append((m.start(), m.end(), "percentage", m.group(0)))

    edits.sort()
    out, last = [], 0
    for start, end, kind, original in edits:
        if start < last:
            continue
        out.append(text[last:start])
        out.append(VERIFY_PLACEHOLDER)
        flags.append(f"Unsupported {kind} removed: '{original}'")
        last = end
    out.append(text[last:])
    return "".join(out)


def _sentence_is_unsupported(sentence: str, sup: dict) -> bool:
    """True when a sentence asserts a completion/waiver/service claim that
    the original draft does not contain."""
    bare = re.sub(r"\[[^\]]*\]", " ", sentence)
    if _WAIVER_RE.search(bare) and not sup["draft_has_waiver"]:
        return True
    for m in _ACTIVE_CLAIM_RE.finditer(bare):
        verb = m.group(1).lower()
        if verb in _CLAIM_VERBS and verb not in sup["draft_claim_verbs"]:
            return True
    for m in _PASSIVE_CLAIM_RE.finditer(bare):
        verb = m.group(1).lower()
        if verb in _PASSIVE_CLAIM_VERBS and verb not in sup["draft_claim_verbs"]:
            return True
    if _CERT_RE.search(bare):
        low = bare.lower()
        for method in _CERT_METHODS:
            if method in low and method not in sup["draft_lower"]:
                return True
    return False


def _guard_sentences(text: str, sup: dict, flags: list[str]) -> str:
    lines = []
    for line in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = []
        for part in parts:
            if part.strip() and _sentence_is_unsupported(part, sup):
                flags.append(
                    "Unsupported claim removed: '"
                    + (part.strip()[:140] + ("..." if len(part.strip()) > 140 else ""))
                    + "'"
                )
                kept.append(REMOVED_CLAIM)
            else:
                kept.append(part)
        lines.append(" ".join(kept))
    return "\n".join(lines)


def guard_unverified_specifics(
    revised_draft: dict,
    hardening_report: list,
    draft_text: str,
    attack: dict,
    disclaimer_line: str = DISCLAIMER_LINE,
) -> tuple[dict, list, list[str]]:
    """Strip every concrete specific in the hardened output that the inputs
    do not support. Returns (revised_draft, hardening_report, flags)."""
    objections_text = "\n".join(
        f"{o.get('objection', '')} {o.get('argument', '')}"
        for o in attack.get("objections", [])
        if isinstance(o, dict)
    )
    sup = _support_sets(draft_text + "\n" + objections_text, draft_text)
    flags: list[str] = []

    def clean(text: str) -> str:
        return _guard_sentences(_guard_phrases(text, sup, flags), sup, flags)

    revised = {
        "title": revised_draft.get("title", ""),
        "sections": [
            {"heading": s.get("heading", ""), "body": clean(s.get("body", ""))}
            for s in revised_draft.get("sections", [])
        ],
        "full_text": clean(revised_draft.get("full_text", "")),
    }
    if disclaimer_line not in revised["full_text"]:
        revised["full_text"] = (
            revised["full_text"].rstrip() + "\n\n" + disclaimer_line
        )

    report = []
    for entry in hardening_report:
        entry = dict(entry)
        if entry.get("fix_applied"):
            entry["fix_applied"] = clean(entry["fix_applied"])
        report.append(entry)

    # De-duplicate flags, preserving order.
    seen, unique = set(), []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            unique.append(flag)
    return revised, report, unique


def harden_draft(
    draft_text: str,
    attack: dict,
    model: str | None = None,
    language: str = "en",
) -> dict:
    """Revise a draft against an attack; returns revised_draft +
    hardening_report + unverified_additions (specifics the guard stripped
    because the inputs did not support them)."""
    language = normalize_language(language)
    prompt = build_harden_prompt(draft_text, attack, language)
    raw = run_prompt(prompt, model=resolve_model(model))
    result = _validate_hardened(parse_json_response(raw, REVISED_KEYS, "hardening"))
    revised, report, flags = guard_unverified_specifics(
        result["revised_draft"],
        result["hardening_report"],
        draft_text,
        attack,
        disclaimer_line=disclaimer_for(language),
    )
    return {
        "revised_draft": revised,
        "hardening_report": report,
        "unverified_additions": flags,
    }

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
  POST /research         - live legal research (CourtListener, Cornell LII, web)

Round history is persisted per session under server/data/ (see app/sessions.py).
All AI output is general legal information only — never legal advice.
"""
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile


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


@app.post("/research")
async def do_research(request: Request) -> dict:
    """Live legal research across keyless public sources.

    Body: {query, jurisdiction?, language?}
    Returns retrieved sources (CourtListener case law, Cornell LII
    statutes/definitions, and general web results) with titles,
    citations, URLs, and snippets. Retrieval is best-effort: when every
    source is unreachable the result list is simply empty. Retrieved
    sources are returned in their original language; `language` selects
    the response disclaimer language and is echoed back for the client.
    """
    body = await _json_body(request)
    query = _require_str(body, "query")
    jurisdiction = _optional_str(body, "jurisdiction")
    language = normalize_language(_optional_str(body, "language", default="en"))
    try:
        data = research(query, jurisdiction)
    except ResearchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {**data, "language": language, "disclaimer": disclaimer_for(language)}


@app.post("/analyze")
def analyze(file: UploadFile = File(...), language: str = Form("en")) -> dict:
    """Analyze an uploaded legal document.

    Form field name must be "file". Accepts .pdf, .docx, .txt.
    Optional form field "language" (launch-set code, default "en") makes
    the analysis come back in that language.
    """
    language = normalize_language(language)
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

    # Research layer: seed the analysis prompt with live sources keyed to
    # the file name when it carries topical words; retrieval failure just
    # leaves the prompt on model knowledge (research_context returns "").
    seed = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    context = research_context(seed) if len(seed) >= 4 and seed.lower() != "document" else ""

    try:
        result = analyze_document_text(
            text, research_context=context, language=language
        )
    except AnalysisError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Also return retrieved sources on the document's topic so the client
    # can show them; best-effort and additive (older clients ignore it).
    topic = result["document_type"]
    if result["key_claims"]:
        topic = f"{topic} {result['key_claims'][0]}"
    try:
        found = research(topic)["results"]
    except Exception:
        found = []

    return {
        "filename": filename,
        "document_type": result["document_type"],
        "parties": result["parties"],
        "key_claims": result["key_claims"],
        "deadlines": result["deadlines"],
        "weaknesses": result["weaknesses"],
        "summary": result["summary"],
        "research": found,
        "language": language,
        "disclaimer": disclaimer_for(language),
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
    language = normalize_language(_optional_str(body, "language", default="en"))
    model = _optional_str(body, "model")

    raw_session_id = body.get("session_id")
    if raw_session_id is not None and (
        not isinstance(raw_session_id, str) or not raw_session_id.strip()
    ):
        raise HTTPException(
            status_code=400, detail="'session_id' must be a non-empty string."
        )

    # Research layer: pull live sources on the objective for this
    # jurisdiction and hand them to the drafting prompt; empty string on
    # any retrieval failure (draft then relies on model knowledge).
    context = research_context(objective, jurisdiction_state)

    try:
        draft = draft_document(
            analysis=analysis,
            objective=objective,
            jurisdiction_state=jurisdiction_state,
            court=court,
            facts=facts,
            language=language,
            model=model,
            research_context=context,
        )
    except DraftError as e:
        raise HTTPException(status_code=502, detail=str(e))

    session = get_or_create(raw_session_id)
    session["objective"] = objective
    session["jurisdiction_state"] = jurisdiction_state
    session["court"] = court
    session["language"] = language
    session["model"] = model
    add_event(
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
        "disclaimer": disclaimer_for(language),
        "round": 0,
    }


@app.post("/adversary/round")
async def adversary_round(request: Request) -> dict:
    """Run one adversary attack round against a draft.

    Body: {session_id, draft_text, adversary_role, jurisdiction_state,
           court, model?, request_id?}
    Rounds increment from the session's current round; exceeding
    MAX_ADVERSARY_ROUNDS (10) returns 400.

    Idempotency (defect D2): when the client supplies a request_id, it is
    persisted with the attack event. A retry carrying the same request_id
    returns the stored response verbatim — no new round is consumed and
    the model is not called again — so a response lost in transit no
    longer silently burns a round (or a credit). No request_id keeps the
    legacy behavior.
    CLIENT NOTE: the Android client should generate one stable request_id
    (e.g. a UUID) per user tap on "run adversary round" and reuse that
    same id on any retry of that tap. (Client change tracked separately;
    this server side is backward compatible.)
    """
    body = await _json_body(request)
    session_id = _require_str(body, "session_id")
    draft_text = _require_draft_text(body)
    adversary_role = _require_str(body, "adversary_role")
    jurisdiction_state = _require_str(body, "jurisdiction_state")
    court = _require_str(body, "court")
    model = _optional_str(body, "model")
    request_id = _optional_str(body, "request_id")
    if request_id is not None:
        request_id = request_id.strip() or None

    session = get_or_create(session_id)
    language = normalize_language(
        _optional_str(body, "language") or session.get("language") or "en"
    )

    if request_id:
        for event in session.get("events", []):
            if event.get("type") != "attack":
                continue
            if (event.get("data") or {}).get("request_id") == request_id:
                return {
                    "session_id": session["session_id"],
                    "round": event["round"],
                    "attack": event["data"]["attack"],
                    "disclaimer": disclaimer_for(language),
                }

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
            language=language,
        )
    except AdversaryError as e:
        raise HTTPException(status_code=502, detail=str(e))

    session["jurisdiction_state"] = jurisdiction_state
    session["court"] = court
    session["language"] = language
    session["model"] = model
    add_event(
        session,
        "attack",
        next_round,
        {
            "adversary_role": adversary_role,
            "attack": attack,
            "request_id": request_id,
        },
    )

    return {
        "session_id": session["session_id"],
        "round": next_round,
        "attack": attack,
        "disclaimer": disclaimer_for(language),
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

    session = get_or_create(session_id)
    if session["round"] < 1:
        raise HTTPException(
            status_code=400,
            detail="No adversary round has been run for this session yet; "
            "call /adversary/round first.",
        )
    language = normalize_language(
        _optional_str(body, "language") or session.get("language") or "en"
    )

    try:
        result = harden_draft(
            draft_text=draft_text, attack=attack, model=model, language=language
        )
    except AdversaryError as e:
        raise HTTPException(status_code=502, detail=str(e))

    unverified = result.get("unverified_additions", [])
    add_event(
        session,
        "harden",
        session["round"],
        {
            "revised_draft": result["revised_draft"],
            "hardening_report": result["hardening_report"],
            "unverified_additions": unverified,
        },
    )

    return {
        "session_id": session["session_id"],
        "round": session["round"],
        "revised_draft": result["revised_draft"],
        "hardening_report": result["hardening_report"],
        "unverified_additions": unverified,
        "disclaimer": disclaimer_for(language),
    }
