"""
FastAPI backend for the Resume-Matcher tool.

New pipeline (keyword-coverage strategy)
-----------------------------------------
1. extract_jd_keywords()          → 25-50 ATS keyword phrases
2. embed_resume()                  → bullet embeddings
3. embed_keywords()                → keyword embeddings
4. compute_keyword_coverage(before)
5. keyword_driven_rewrite()        → vector-search assign → GPT inject verbatim
6. compute_keyword_coverage(after)
7. xml_patch_docx()                → formatting-preserving DOCX

Endpoints
---------
POST /api/generate   – run full pipeline, return coverage stats + changes
GET  /api/download/{filename} – download generated DOCX
GET  /api/health     – liveness check
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel

from pipeline.embedder import embed_keywords, embed_resume
from pipeline.jd_extractor import extract_company_role, extract_jd_keywords
from pipeline.parser import parse_resume
from pipeline.rewriter import keyword_driven_rewrite
from pipeline.scorer import compute_keyword_coverage
from pipeline.xml_builder import xml_patch_docx

# ── constants ──────────────────────────────────────────────────────────────

# Vercel's filesystem is read-only everywhere except /tmp.
# When running on Vercel (VERCEL env var is set to "1"), write generated
# DOCX files to /tmp/outputs so the download endpoint can serve them.
# Locally, the existing "outputs/" directory is used as before.
_ON_VERCEL = bool(os.environ.get("VERCEL"))
ORIGINAL_RESUME = Path("data/Gnyani_Resume_Final__2_.docx")
OUTPUTS_DIR = Path("/tmp/outputs" if _ON_VERCEL else "outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── app ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Resume Matcher API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],   # "null" covers file:// origin in browsers
    allow_credentials=False,       # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40]


def _html_to_text(html: str) -> str:
    """Convert raw HTML into readable text."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _fetch_url_text_static(url: str) -> str:
    """Fetch raw text from a URL with a plain HTTP request."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    return _html_to_text(html)


def _fetch_url_text_browser(url: str) -> str:
    """Fetch rendered page text from a JavaScript-heavy URL."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except PlaywrightTimeoutError:
                pass
            text = page.locator("body").inner_text(timeout=10000).strip()
            text = re.sub(r"\s{3,}", "\n\n", text)
            return text
        finally:
            context.close()
            browser.close()


def _fetch_url_text(url: str) -> str:
    """Fetch job-description text from a URL, with JS-rendered fallback."""
    JS_SPA_HINTS = [
        "explore.jobs.netflix", "jobs.lever.co", "boards.greenhouse.io",
        "jobs.ashbyhq.com", "careers.smartrecruiters.com", "workday.com",
        "icims.com", "taleo.net", "successfactors.com",
    ]
    is_known_spa = any(h in url for h in JS_SPA_HINTS)

    try:
        text = _fetch_url_text_static(url)
    except Exception as exc:
        text = ""
        static_error = exc
    else:
        static_error = None

    if len(text.split()) >= 150 and not is_known_spa:
        return text

    try:
        rendered_text = _fetch_url_text_browser(url)
        if len(rendered_text.split()) >= 150:
            return rendered_text
    except Exception as exc:
        browser_error = exc
    else:
        browser_error = None

    if static_error:
        detail = f"Could not fetch URL: {static_error}"
    elif browser_error:
        detail = (
            "Could not extract meaningful text from the URL, even after "
            f"rendering it in a browser: {browser_error}"
        )
    else:
        detail = (
            "Could not extract enough job-description text from the URL after "
            "trying both static and browser-rendered fetches."
        )

    raise HTTPException(status_code=422, detail=detail)
    return text


def _enrich_rewrites(
    rewrites: list[dict[str, Any]],
    resume_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add 'section' label to each rewrite by cross-referencing the parsed resume."""
    enriched: list[dict[str, Any]] = []
    for rw in rewrites:
        section = ""
        norm = rw["original"].strip().lower()[:80]
        for exp in resume_json.get("experience", []):
            for bullet in exp.get("bullets", []):
                text = bullet if isinstance(bullet, str) else bullet.get("text", "")
                if text.strip().lower()[:80] == norm:
                    company = exp.get("company", "")
                    role = exp.get("role", "")
                    section = f"{company} — {role}".strip(" —")
                    break
            if section:
                break
        enriched.append({**rw, "section": section})
    return enriched


# ── request / response ─────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    jd_text: str = ""
    jd_url: str = ""


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Serve the browser UI.

    FileResponse reads frontend.html from the project root, which is bundled
    into the Vercel deployment package (the file is committed to git and not
    gitignored).  Falls back to a JSON message if the file is somehow absent.
    """
    html_path = Path("frontend.html")
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"message": "Resume Matcher API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    """Run the full keyword-coverage pipeline and return results."""

    # 1. Resolve JD text
    jd_text = req.jd_text.strip()
    if not jd_text and req.jd_url.strip():
        jd_text = _fetch_url_text(req.jd_url.strip())
    if not jd_text:
        raise HTTPException(status_code=400, detail="Provide jd_text or jd_url.")

    if not ORIGINAL_RESUME.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Original resume not found at {ORIGINAL_RESUME}",
        )

    # 2. Extract JD keywords (25-50 ATS terms)
    keywords = extract_jd_keywords(jd_text)

    # 3. Parse resume
    resume_json = parse_resume(ORIGINAL_RESUME)

    # 4. Embed bullets + keywords in parallel (sequential calls, batched)
    resume_with_emb = embed_resume(resume_json)
    keywords_with_emb = embed_keywords(keywords)

    # 5. Keyword coverage BEFORE rewriting
    before_cov = compute_keyword_coverage(resume_json, keywords)

    # 6. Vector-search assign + GPT inject keywords verbatim
    updated_resume, rewrites, assignment = keyword_driven_rewrite(
        resume_json, resume_with_emb, keywords_with_emb
    )

    # 7. Keyword coverage AFTER rewriting
    after_cov = compute_keyword_coverage(updated_resume, keywords)

    # 8. Build output DOCX (XML-patch — preserves formatting)
    identity = extract_company_role(jd_text)
    company = _sanitize(identity.get("company", "Company"))
    role = _sanitize(identity.get("role", "Role"))
    filename = f"Gnyani_{company}_{role}.docx"
    output_path = OUTPUTS_DIR / filename
    xml_patch_docx(ORIGINAL_RESUME, rewrites, output_path)

    # 9. Enrich rewrites with section labels
    enriched = _enrich_rewrites(rewrites, resume_json)

    # 10. Build per-keyword before/after diff
    before_map = {r["keyword"]: r["matched"] for r in before_cov["keywords"]}
    after_map  = {r["keyword"]: r["matched"] for r in after_cov["keywords"]}
    keyword_diff = [
        {
            "keyword": kw,
            "before": before_map.get(kw, False),
            "after":  after_map.get(kw, False),
            "gained": (not before_map.get(kw, False)) and after_map.get(kw, False),
        }
        for kw in keywords
    ]

    return {
        "filename": filename,
        # Primary metric: keyword coverage
        "keywords_total":   len(keywords),
        "keywords_before":  before_cov["matched"],
        "keywords_after":   after_cov["matched"],
        "coverage_before":  before_cov["pct"],
        "coverage_after":   after_cov["pct"],
        "coverage_gain":    round(after_cov["pct"] - before_cov["pct"], 1),
        # Per-keyword breakdown
        "keyword_diff": keyword_diff,
        # Changes (bullets rewritten)
        "rewrites_count": len(rewrites),
        "changes": enriched,
    }


@app.get("/api/download/{filename}")
def download(filename: str):
    """Download the generated DOCX."""
    safe_name = Path(filename).name
    path = OUTPUTS_DIR / safe_name
    if not path.exists() or path.suffix != ".docx":
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(path),
        filename=safe_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
