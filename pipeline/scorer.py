"""
ATS scoring module.

Two scoring modes:

1. compute_keyword_coverage(resume, keywords)  ← PRIMARY
   Measures what fraction of the extracted JD keyword list appears in the
   resume text (case-insensitive substring).  This mirrors how real ATS
   systems actually rank candidates — they count keyword occurrences.
   Target: ≥ 80 % after rewriting.

2. score_resume_against_jd(embedded_resume, embedded_jd)  ← SECONDARY
   Hybrid: 60 % keyword-in-requirement coverage + 40 % semantic similarity.
   Used for per-requirement breakdown display.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from config import SIMILARITY_THRESHOLD


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "do", "for",
    "from", "have", "i", "in", "is", "it", "its", "not", "of", "on", "or",
    "our", "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "we", "were", "will", "with", "you",
    # resume/JD filler
    "experience", "ability", "familiarity", "comfortable", "strong", "years",
    "similar", "like", "using", "such", "including", "working", "across",
}

_PHRASE_PAIRS = [
    ("infrastructure", "code"),
    ("data", "quality"),
    ("data", "pipelines"),
    ("best", "practices"),
    ("junior", "engineers"),
    ("warehouse", "models"),
    ("pipeline", "performance"),
    ("business", "intelligence"),
    ("product", "analytics"),
    ("validation", "checks"),
    ("distributed", "processing"),
]

# Synonym map: if a JD keyword isn't found, check these alternatives
_SYNONYMS: dict[str, list[str]] = {
    "spark":  ["spark", "distributed processing", "large-scale processing", "flink", "kafka"],
    "flink":  ["flink", "spark", "distributed processing", "kafka"],
    "dbt":    ["dbt", "data build tool", "data transformation", "sql transformation"],
    "terraform": ["terraform", "infrastructure-as-code", "iac", "infrastructure as code"],
    "infrastructure-as-code": ["infrastructure-as-code", "terraform", "iac", "cloudformation"],
    "snowflake": ["snowflake", "redshift", "bigquery", "data warehouse", "cloud warehouse"],
    "redshift": ["redshift", "snowflake", "bigquery", "data warehouse"],
    "airflow": ["airflow", "orchestration", "workflow orchestration", "pipeline orchestration"],
    "mentoring": ["mentor", "mentored", "mentoring", "coaching", "junior engineers", "best practices"],
    "validation": ["validation", "data quality", "testing", "tdd", "quality framework"],
}


def _extract_keywords(requirement: str) -> list[str]:
    """Extract meaningful keywords and key phrases from a single requirement."""
    text = requirement.lower()
    tokens = re.findall(r"\b[a-z][a-z0-9.+#/-]*\b", text)

    keywords: list[str] = []

    # Multi-word phrases first (exact substring match later)
    for w1, w2 in _PHRASE_PAIRS:
        if w1 in tokens and w2 in tokens:
            phrase = f"{w1} {w2}"
            if phrase in text:
                keywords.append(phrase)

    # Single meaningful tokens (proper nouns / tech terms get priority)
    for token in tokens:
        if token in _STOPWORDS or len(token) < 3:
            continue
        # Keep specific tool/tech names and meaningful nouns
        if (
            token not in {k.split()[0] for k in keywords}  # not already covered by phrase
            and token not in {"proficiency", "hands", "designing", "driving",
                              "building", "supporting", "improving", "mentoring"}
        ):
            keywords.append(token)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for kw in keywords:
        if kw not in seen:
            deduped.append(kw)
            seen.add(kw)
    return deduped


def _keyword_score(keywords: list[str], resume_full_text: str) -> float:
    """Fraction of keywords that appear anywhere in the full resume text.
    Uses synonym matching so that e.g. 'Airflow' matches 'orchestration',
    'Spark' matches 'distributed processing', 'Terraform' matches 'infrastructure-as-code'.
    """
    if not keywords:
        return 0.0
    lower = resume_full_text.lower()

    def _match(kw: str) -> bool:
        kl = kw.lower()
        if kl in lower:
            return True
        # Check synonyms
        for syn in _SYNONYMS.get(kl, []):
            if syn.lower() in lower:
                return True
        return False

    hits = sum(1 for kw in keywords if _match(kw))
    return hits / len(keywords)


def _full_resume_text(resume_json: dict[str, Any]) -> str:
    """Flatten the entire resume into a single string for keyword scanning."""
    parts: list[str] = []
    for exp in resume_json.get("experience", []):
        parts.append(str(exp.get("role", "")))
        parts.append(str(exp.get("company", "")))
        for bullet in exp.get("bullets", []):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            parts.append(text)
    for cat, values in resume_json.get("skills", {}).items():
        parts.append(cat)
        if isinstance(values, dict):
            parts.extend(values.get("items", []))
        elif isinstance(values, list):
            parts.extend(values)
    return " ".join(parts)


def compute_keyword_coverage(
    resume: dict[str, Any] | str,
    keywords: list[str],
) -> dict[str, Any]:
    """
    Primary ATS metric: what % of JD keywords appear in the resume?

    Parameters
    ----------
    resume   : either a parsed resume dict or a plain string of resume text
    keywords : list of keyword strings extracted from the JD

    Returns
    -------
    {
      "matched":  int,          # number of keywords found
      "total":    int,          # total keywords
      "pct":      float,        # percentage 0–100
      "keywords": [{            # per-keyword result
          "keyword": str,
          "matched": bool,
      }],
    }
    """
    if isinstance(resume, dict):
        text = _full_resume_text(resume)
    else:
        text = resume

    lower = text.lower()

    results: list[dict[str, Any]] = []
    for kw in keywords:
        # Try exact substring first
        matched = kw.lower() in lower
        # Also try each word of a multi-word keyword (partial match counts)
        if not matched and " " in kw:
            words = [w for w in kw.lower().split() if len(w) > 3]
            matched = all(w in lower for w in words) if words else False
        results.append({"keyword": kw, "matched": matched})

    matched_count = sum(1 for r in results if r["matched"])
    total = len(keywords)
    return {
        "matched": matched_count,
        "total": total,
        "pct": round(matched_count / total * 100, 1) if total else 0.0,
        "keywords": results,
    }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    return float(np.dot(a_np, b_np) / denom) if denom != 0 else 0.0


# --------------------------------------------------------------------------- #
# main scorer
# --------------------------------------------------------------------------- #

def score_resume_against_jd(
    resume_with_embeddings: dict[str, Any],
    jd_with_embeddings: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Score resume against JD using a hybrid keyword + semantic metric.
    Returns overall_score (0–1), per-requirement breakdown, and weak_bullets list.
    """
    # Full resume text for keyword scanning
    resume_text = _full_resume_text(resume_with_embeddings)

    # Flat list of embedded experience bullets
    bullet_entries: list[dict[str, Any]] = []
    for exp in resume_with_embeddings.get("experience", []):
        for bullet in exp.get("bullets", []):
            if isinstance(bullet, dict) and bullet.get("embedding"):
                bullet_entries.append({
                    "section": f'Experience - {exp.get("role", "Role")}',
                    "bullet": bullet.get("text", ""),
                    "embedding": bullet["embedding"],
                })

    requirement_coverage: list[dict[str, Any]] = []
    weak_bullets: list[dict[str, Any]] = []

    for requirement in jd_with_embeddings:
        req_text = requirement["text"]
        keywords = _extract_keywords(req_text)

        # Keyword coverage score (real ATS behaviour)
        kw_score = _keyword_score(keywords, resume_text)

        # Semantic similarity — best bullet match
        best_sem_score = 0.0
        best_match_bullet = ""
        for bullet in bullet_entries:
            s = cosine_similarity(bullet["embedding"], requirement["embedding"])
            if s > best_sem_score:
                best_sem_score = s
                best_match_bullet = bullet["bullet"]

        # Combined score: keyword-weighted (mirrors real ATS more closely)
        combined = 0.6 * kw_score + 0.4 * best_sem_score

        requirement_coverage.append({
            "requirement": req_text,
            "best_match_bullet": best_match_bullet,
            "score": combined,
            "keyword_score": kw_score,
            "semantic_score": best_sem_score,
            "keywords_checked": keywords,
        })

    # Identify weak bullets (low best semantic score — still drives rewrites)
    for exp_index, exp in enumerate(resume_with_embeddings.get("experience", [])):
        for bullet_index, bullet in enumerate(exp.get("bullets", [])):
            if not isinstance(bullet, dict) or not bullet.get("embedding"):
                continue
            best_score = max(
                (cosine_similarity(bullet["embedding"], req["embedding"])
                 for req in jd_with_embeddings),
                default=0.0,
            )
            if best_score < SIMILARITY_THRESHOLD:
                weak_bullets.append({
                    "section": f'Experience - {exp.get("role", "Role")}',
                    "experience_index": exp_index,
                    "index": bullet_index,
                    "bullet": bullet["text"],
                    "best_score": best_score,
                })

    overall = float(np.mean([r["score"] for r in requirement_coverage])) if requirement_coverage else 0.0

    return {
        "overall_score": overall,
        "requirement_coverage": requirement_coverage,
        "weak_bullets": weak_bullets,
    }
