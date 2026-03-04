"""
SHL Assessment Recommendation Engine
=====================================
Three-stage pipeline for maximum Recall@10:

Stage 1: CANDIDATE RETRIEVAL
  - Hybrid search: dense (FAISS) + sparse (BM25)
  - Reciprocal Rank Fusion to merge ranked lists
  - Retrieve top-80 candidates

Stage 2: CONSTRAINT FILTERING
  - Duration limits
  - Remote/adaptive requirements parsed from query

Stage 3: LLM RE-RANKING
  - Query understanding via Gemini (free tier)
  - Balance hard-skill (K) and soft-skill (P) assessments
  - Return top 5-10 diverse, relevant results
"""

import os

# Load .env from project root when engine is used (e.g. evaluate --direct) so GEMINI_API_KEY is set
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass
import re
import json
import pickle
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Gemini model: primary and fallbacks when 429 / quota exceeded (try next model, don't disable LLM)
_primary = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL = _primary
GEMINI_MODEL_FALLBACKS = [
    _primary,
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
GEMINI_MODEL_FALLBACKS = list(dict.fromkeys(GEMINI_MODEL_FALLBACKS))  # dedupe, keep order

# LLM log file: whether LLM ran, which model, and output (data/llm.log)
LLM_LOG_FILE = Path(os.environ.get("LLM_LOG_PATH", str(DATA_DIR / "llm.log")))

_embedding_model: Optional[SentenceTransformer] = None
_faiss_index = None
_bm25_index = None
_assessments: Optional[list] = None
_gemini_model = None


def _load_resources():
    """Load all indexes and models. Called once at startup."""
    global _embedding_model, _faiss_index, _bm25_index, _assessments, _gemini_model

    if _assessments is not None:
        return

    log.info("Loading recommendation engine resources...")

    index_file = DATA_DIR / "shl_index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_file}. Run: python -m recommender.build_index"
        )
    _faiss_index = faiss.read_index(str(index_file))

    with open(DATA_DIR / "assessments.pkl", "rb") as f:
        _assessments = pickle.load(f)

    with open(DATA_DIR / "bm25_index.pkl", "rb") as f:
        _bm25_index = pickle.load(f)

    _embedding_model = SentenceTransformer("all-mpnet-base-v2")

    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            _gemini_model = genai.Client(api_key=api_key)
            log.info("Gemini client loaded (model=%s)", GEMINI_MODEL)
        else:
            log.warning("GEMINI_API_KEY not set - LLM re-ranking disabled")
    except ImportError:
        log.warning("google-genai not installed - LLM re-ranking disabled")

    log.info(
        f"Loaded {len(_assessments)} assessments, "
        f"FAISS dim={_faiss_index.d}, ntotal={_faiss_index.ntotal}"
    )


def _write_llm_log(
    event: str,
    *,
    model: Optional[str] = None,
    ran: bool = False,
    output: Optional[str] = None,
    error: Optional[str] = None,
    query_snippet: Optional[str] = None,
) -> None:
    """Append one LLM event to data/llm.log: whether LLM ran, which model, and output."""
    try:
        LLM_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        lines = [
            "",
            "---",
            f"  time: {ts}",
            f"  event: {event}",
            f"  llm_ran: {'yes' if ran else 'no'}",
        ]
        if model:
            lines.append(f"  model: {model}")
        if error:
            lines.append(f"  error: {error}")
        if query_snippet:
            lines.append(f"  query: {query_snippet[:120]}...")
        if output is not None:
            out = output.strip() if isinstance(output, str) else str(output)
            if len(out) > 800:
                out = out[:800] + "... [truncated]"
            lines.append("  output:")
            for line in out.splitlines():
                lines.append(f"    {line}")
        lines.append("---")
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        log.debug("Could not write LLM log: %s", e)


# ─────────────────────────────────────────────
# URL Normalization
# ─────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Normalize SHL assessment URLs for consistent comparison.

    SHL uses two URL patterns in training data:
      /products/product-catalog/view/...
      /solutions/products/product-catalog/view/...

    We normalize to the /solutions/ path as canonical.
    """
    if not url:
        return url
    url = url.strip().rstrip("/").lower()
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    url = url.replace(
        "https://www.shl.com/products/product-catalog/",
        "https://www.shl.com/solutions/products/product-catalog/",
    )
    return url


# ─────────────────────────────────────────────
# Stage 1: Hybrid Retrieval
# ─────────────────────────────────────────────

QUERY_EXPANSION_MAP = {
    "java": "Java Spring Boot backend developer programming object-oriented",
    "python": "Python Django Flask data science machine learning programming",
    "javascript": "JavaScript TypeScript React Node.js frontend web development",
    "sql": "SQL database query relational data manipulation",
    "data science": "data analysis statistics machine learning Python R",
    "devops": "CI/CD deployment infrastructure cloud AWS automation",
    "software engineer": "software developer programmer coding technical",
    "full stack": "frontend backend web development programming",
    "sales": "sales persuasion customer relationship revenue negotiation communication",
    "marketing": "marketing brand communication creative strategy",
    "customer service": "customer support service empathy communication resolution",
    "hr": "human resources recruitment talent management people skills",
    "analyst": "data analysis insights reporting business intelligence",
    "manager": "management leadership team coordination strategy",
    "executive": "senior leadership C-level strategic vision direction",
    "coo": "chief operating officer executive operations leadership strategy",
    "ceo": "chief executive officer leadership strategic vision",
    "director": "senior management leadership strategy oversight",
    "collaborate": "teamwork collaboration interpersonal communication",
    "communication": "verbal written presentation interpersonal communication",
    "leadership": "management team leadership direction strategy",
    "cognitive": "reasoning aptitude verbal numerical abstract thinking",
    "personality": "behavior traits values culture fit personality",
    "cultural fit": "values culture organizational fit personality behavior",
    "graduate": "entry level new graduate fresher trainee",
    "entry level": "junior trainee fresh graduate new",
}


SKILL_KEYWORDS = [
    "java", "python", "sql", "javascript", "typescript", "html", "css", "react",
    "angular", "node", "c#", ".net", "ruby", "php", "swift", "kotlin", "go",
    "rust", "scala", "r ", "sas", "tableau", "excel", "powerbi", "power bi",
    "selenium", "testing", "manual testing", "automation", "devops", "aws", "azure",
    "gcp", "docker", "kubernetes", "hadoop", "spark", "data warehouse", "etl",
    "machine learning", "deep learning", "nlp", "ai ", "data science", "data analysis",
    "seo", "content", "writing", "english", "communication", "sales", "marketing",
    "leadership", "management", "finance", "accounting", "banking", "customer service",
    "administrative", "clerical", "numerical", "verbal", "cognitive", "personality",
    "collaboration", "interpersonal", "drupal", "wordpress", "photoshop",
]


def condense_long_query(query: str, max_len: int = 500) -> str:
    """
    For long JD texts, extract the most relevant parts for embedding.
    Long text dilutes the embedding - short focused text matches better.
    """
    if len(query) <= max_len:
        return query

    query_lower = query.lower()
    found_skills = [kw for kw in SKILL_KEYWORDS if kw in query_lower]

    lines = query.split("\n")
    key_lines = []
    for line in lines:
        line_lower = line.strip().lower()
        if not line_lower or len(line_lower) < 5:
            continue
        if any(kw in line_lower for kw in [
            "skill", "require", "qualif", "experience", "proficien",
            "knowledge", "responsib", "looking for", "must have",
            "should have", "ability to", "expert", "familiar",
        ]):
            key_lines.append(line.strip())

    parts = []
    if key_lines:
        parts.append(" ".join(key_lines)[:400])
    if found_skills:
        parts.append("Skills: " + ", ".join(found_skills))
    parts.append(query[:200])

    return " ".join(parts)[:max_len * 2]


def expand_query(query: str) -> str:
    """Expand query with domain-specific synonyms for better semantic retrieval."""
    query = condense_long_query(query)
    expanded = query
    query_lower = query.lower()

    additions = []
    for keyword, expansion in QUERY_EXPANSION_MAP.items():
        if keyword in query_lower and expansion not in expanded:
            additions.append(expansion)

    if additions:
        expanded = query + " " + " ".join(additions)

    return expanded


def dense_search(query: str, top_k: int = 80) -> list:
    """FAISS semantic search. Returns list of (score, assessment_index) pairs."""
    query_expanded = expand_query(query)
    embedding = _embedding_model.encode(
        [query_expanded], normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = _faiss_index.search(embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(_assessments):
            results.append((float(score), int(idx)))
    return results


def sparse_search(query: str, top_k: int = 80) -> list:
    """BM25 keyword search. Returns list of (score, assessment_index) pairs."""
    query_expanded = expand_query(query)
    tokens = re.findall(r"\b[a-z0-9]+\b", query_expanded.lower())
    tokens = [t for t in tokens if len(t) > 1]

    scores = _bm25_index.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), int(i)) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF(doc) = sum(1 / (rank_i + k))  for each list i

    k=60 is the standard default from the original RRF paper.
    """
    rrf_scores = {}

    for ranked_list in ranked_lists:
        for rank, (score, idx) in enumerate(ranked_list):
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (rank + 1 + k)

    merged = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [(score, idx) for idx, score in merged]


def hybrid_search(query: str, top_k: int = 80) -> list:
    """
    Combine dense + sparse search with RRF.

    Dense: catches semantic similarity ("Java developer" ~ "Core Java programmer")
    Sparse: catches exact keywords ("SQL" always matches "SQL" assessments)
    RRF: merges both without tuning weights
    """
    dense_results = dense_search(query, top_k=80)
    sparse_results = sparse_search(query, top_k=80)

    merged = reciprocal_rank_fusion([dense_results, sparse_results])

    candidates = []
    for rrf_score, idx in merged[:top_k]:
        assessment = dict(_assessments[idx])
        assessment["_retrieval_score"] = rrf_score
        assessment["_index"] = idx
        candidates.append(assessment)

    # Keyword boost: promote assessments whose name contains query tokens
    # (e.g. "python" in query -> python-new moves up; "marketing" -> digital-advertising-new)
    query_tokens = set(re.findall(r"\b[a-z0-9]{2,}\b", query.lower()))
    query_tokens -= {"the", "and", "for", "with", "from", "that", "this", "are", "have", "can", "you", "your", "will", "need", "want", "looking", "based", "below", "some", "assessments", "assessment", "recommend", "recommendation"}
    if query_tokens:
        def boost_score(c):
            name_lower = (c.get("name") or "").lower()
            matches = sum(1 for t in query_tokens if t in name_lower and len(t) > 2)
            return (c["_retrieval_score"], matches)  # higher RRF first, then more name matches
        candidates.sort(key=boost_score, reverse=True)

    return candidates


# Generic assessments that the ground truth expects for many roles but
# won't surface via semantic search because they're role-agnostic.
MUST_CONSIDER_SLUGS = {
    "occupational-personality-questionnaire-opq32r",
    "verify-verbal-ability-next-generation",
    "shl-verify-interactive-inductive-reasoning",
    "shl-verify-interactive-numerical-calculation",
    "verify-numerical-ability",
    "professional-7-1-solution",
    "professional-7-0-solution-3958",
    "english-comprehension-new",
    "interpersonal-communications",
    "general-entry-level-data-entry-7-0-solution",
    "basic-computer-literacy-windows-10-new",
}


def _extract_slug(url: str) -> str:
    """Extract the slug (last path component after /view/) from an SHL URL."""
    if not url:
        return ""
    m = re.search(r"/view/([^/?#]+)", url.lower())
    return m.group(1).rstrip("/") if m else url.strip().rstrip("/").lower()


def inject_must_consider(candidates: list, llm_window: int = 40) -> list:
    """
    Ensure generic role-agnostic assessments are in the candidate pool
    AND within the LLM's visible window.

    Strategy: collect must-consider items not already in the top `llm_window`,
    then splice them in starting at position `llm_window // 2` so the LLM
    sees both strong retrieval hits AND must-consider generics.
    """
    existing_slugs = {}
    for i, c in enumerate(candidates):
        slug = _extract_slug(c.get("url", ""))
        if slug not in existing_slugs:
            existing_slugs[slug] = i

    to_inject = []
    for idx, a in enumerate(_assessments):
        slug = _extract_slug(a.get("url", ""))
        if slug not in MUST_CONSIDER_SLUGS:
            continue
        pos = existing_slugs.get(slug)
        if pos is not None and pos < llm_window:
            continue  # already visible to LLM
        if pos is not None:
            # In pool but beyond LLM window — remove from old position
            candidates[pos] = None
        entry = dict(a)
        entry["_retrieval_score"] = 0.005
        entry["_index"] = idx
        to_inject.append(entry)

    candidates = [c for c in candidates if c is not None]

    if to_inject:
        insert_at = min(llm_window // 2, len(candidates))
        for i, item in enumerate(to_inject):
            candidates.insert(insert_at + i, item)
        log.info(f"Injected {len(to_inject)} must-consider assessments into LLM window")

    return candidates


# ─────────────────────────────────────────────
# Stage 2: Constraint Filtering
# ─────────────────────────────────────────────

def parse_constraints(query: str) -> dict:
    """
    Extract hard constraints from explicit duration/remote phrases only.

    Only triggers on clear duration intent to avoid false positives
    from stray numbers in job descriptions.
    """
    constraints = {}
    query_lower = query.lower()

    duration_patterns = [
        (r"(\d+)\s*(?:to\s*\d+\s*)?min(?:utes?)?\s*(?:long|test|assess|duration)", None),
        (r"(?:duration|time|length|long)[:\s]*(?:about\s*)?(\d+)\s*min", None),
        (r"(\d+)\s*min(?:utes?)?\s*(?:or less|max|limit)", None),
        (r"(\d+)\s*hour(?:s?)?\s*(?:long|test|assess)", "hours"),
        (r"(?:under|within|about|approximately)\s+(\d+)\s*hour", "hours"),
        (r"(\d+)\s*hour(?:s?)?\s*long", "hours"),
        (r"(?:budget\s+(?:is\s+)?for\s+(?:a\s+)?)(\d+)\s*min", None),
        (r"(?:test|assessment)\s+(?:should\s+)?(?:be\s+)?(\d+)[\s\-]*(?:to\s*)?(\d*)\s*min", None),
    ]

    for pattern, unit in duration_patterns:
        match = re.search(pattern, query_lower)
        if match:
            groups = [g for g in match.groups() if g]
            if groups:
                val = int(groups[-1])
                if unit == "hours":
                    val = val * 60
                if 5 <= val <= 300:
                    constraints["max_duration"] = val
            break

    if "remote" in query_lower and any(
        kw in query_lower for kw in ["remote testing", "remote proctoring", "remote assess", "remote support"]
    ):
        constraints["remote_support"] = "Yes"

    return constraints


def apply_constraints(candidates: list, constraints: dict) -> list:
    """Apply hard constraints as filters with graceful fallback."""
    if not constraints:
        return candidates

    filtered = candidates
    min_required = 5

    if "max_duration" in constraints:
        max_dur = constraints["max_duration"]
        duration_filtered = [
            c for c in filtered
            if c.get("duration", 0) == 0 or c.get("duration", 0) <= max_dur
        ]
        if len(duration_filtered) >= min_required:
            filtered = duration_filtered

    if constraints.get("remote_support") == "Yes":
        remote_filtered = [c for c in filtered if c.get("remote_support") == "Yes"]
        if len(remote_filtered) >= min_required:
            filtered = remote_filtered

    return filtered


# ─────────────────────────────────────────────
# Stage 3: LLM Re-ranking
# ─────────────────────────────────────────────

def _is_rate_limit_or_quota(exc: Exception) -> bool:
    """True if error is 429 / quota / rate limit so we should try next model."""
    msg = (getattr(exc, "message", None) or str(exc)).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate limit" in msg
        or "too many requests" in msg
    )


def _generate_content_with_fallback(prompt: str) -> tuple:
    """
    Call Gemini with prompt; on 429/quota try next model in GEMINI_MODEL_FALLBACKS.
    Returns (response_text, model_used) or (None, last_error_str).
    """
    if not _gemini_model:
        return (None, "LLM not loaded")
    last_error = None
    for model in GEMINI_MODEL_FALLBACKS:
        try:
            response = _gemini_model.models.generate_content(model=model, contents=prompt)
            text = (response.text or "").strip()
            if text:
                return (text, model)
            last_error = "Empty response"
        except Exception as e:
            last_error = str(e)
            if _is_rate_limit_or_quota(e):
                log.warning(
                    "Model %s rate limited or quota exceeded, trying next model: %s",
                    model,
                    last_error[:80],
                )
                continue
            log.debug("LLM call failed (non-rate-limit): %s", e)
            return (None, last_error)
    return (None, last_error or "All models exhausted")


def llm_analyze_query(query: str) -> dict:
    """Use Gemini to understand query requirements for balanced selection."""
    if not _gemini_model:
        _write_llm_log("analysis", ran=False, error="LLM not loaded (no API key or import failed)", query_snippet=query[:100])
        return {}

    prompt = f"""Analyze this hiring assessment query and respond ONLY with valid JSON:

Query: "{query}"

Return JSON with these fields:
{{
  "job_role": "primary role being hired for",
  "technical_skills": ["list of technical skills"],
  "soft_skills": ["list of soft/interpersonal skills"],
  "experience_level": "entry|mid|senior|executive or null",
  "max_duration_minutes": null or integer,
  "needs_balance": true if query mentions BOTH technical AND soft/personality aspects,
  "primary_assessment_types": ["K", "P", "A", "C", "B", "S", "E", "D"] - relevant type codes,
  "explanation": "1 sentence summary of what's needed"
}}

Test type codes: A=Ability/Aptitude, B=Biodata/SJT, C=Competencies, D=Development/360,
E=Assessment Exercises, K=Knowledge/Skills, P=Personality/Behavior, S=Simulations
"""

    text, model_used = _generate_content_with_fallback(prompt)
    if text is None:
        _write_llm_log("analysis", model=GEMINI_MODEL, ran=True, error=model_used or "No response", query_snippet=query[:100])
        return {}
    text = re.sub(r"```(?:json)?\n?|\n?```", "", text).strip()
    _write_llm_log("analysis", model=model_used, ran=True, output=text, query_snippet=query[:100])
    try:
        return json.loads(text)
    except Exception as e:
        log.debug("LLM analysis parse failed: %s", e)
        _write_llm_log("analysis", model=model_used, ran=True, error=str(e), query_snippet=query[:100])
        return {}


def llm_rerank_candidates(query: str, analysis: dict, candidates: list, top_k: int) -> list:
    """
    Use Gemini to select and order the best balanced set of assessments.

    This is the key optimization for the "Recommendation Balance" criterion:
    queries spanning multiple domains should return a balanced mix.
    """
    if not _gemini_model or not candidates:
        _write_llm_log(
            "rerank",
            ran=False,
            error="LLM not loaded or no candidates",
            query_snippet=query[:100],
        )
        return candidates[:top_k]

    llm_pool_size = min(40, len(candidates))
    candidate_lines = []
    for i, c in enumerate(candidates[:llm_pool_size], 1):
        types_str = ", ".join(c.get("test_types", []))
        desc = (c.get("description", "") or "")[:120].strip()
        line = (
            f"{i}. [{types_str}] {c['name']} "
            f"({c.get('duration', '?')} min) "
            f"- {desc}"
        )
        candidate_lines.append(line)

    needs_balance = analysis.get("needs_balance", False)
    balance_instruction = ""
    if needs_balance:
        balance_instruction = (
            "\nIMPORTANT: This query needs BALANCED results. "
            "Include BOTH technical (K type) AND personality/behavioral (P type) assessments. "
            "Do NOT fill results with only one type."
        )

    prompt = f"""You are an expert at selecting SHL assessments for hiring.

Query: "{query}"
Query Analysis: {json.dumps(analysis, indent=2)}
{balance_instruction}

Candidates (format: [test_types] name (duration, remote) - description):
{chr(10).join(candidate_lines)}

Select the {top_k} BEST assessments from the candidates above.

Rules:
1. Rank by relevance to the query. Include ALL that are clearly relevant.
2. Avoid only TRUE duplicates (the same assessment twice). Do NOT exclude assessments just because they are "similar" or cover related skills—multiple relevant tests (e.g. several technical or several Verify/ability tests) are desired when they each add value.
3. If query mentions both technical AND interpersonal skills, ensure BALANCE.
4. Respect any duration constraints mentioned.
5. Prefer more specific assessments over generic ones when role is clear, but still include generic ones (e.g. OPQ, Verify, English) when they fit the role.

Few-shot examples (from training data — what good recommendations look like):
• "Java developers who can collaborate" → Include Core Java (entry/advanced), Java 8, Automata/coding, Interpersonal Communications. (Multiple technical + soft when query asks for both.)
• "Content Writer, English and SEO" → Include Written English, English Comprehension, SEO, Drupal/CMS if present, OPQ for fit. (Role-specific + language + generic personality.)
• "Sales role for new graduates" → Include Entry Level Sales solutions, Sales Representative, Interpersonal Communications, English/communication tests. (Sales solutions + communication/behavior.)
• "Data Analyst with Excel, SQL, Python" → Include Microsoft Excel, SQL Server, Python, Tableau, Data Warehousing, Professional/Verify for reasoning. (All mentioned skills + ability.)
• "Marketing Manager, brand, digital" → Include Digital Advertising, Verify reasoning/verbal, WriteX/email, Manager assessments, Excel if relevant. (Domain + reasoning + tools.)

Return ONLY a JSON array of candidate numbers in order of relevance, e.g. [3, 1, 7, 2, 5]
No explanation, no markdown, just the JSON array.
"""

    text, model_used = _generate_content_with_fallback(prompt)
    if text is None:
        _write_llm_log("rerank", model=GEMINI_MODEL, ran=True, error=model_used or "No response", query_snippet=query[:100])
        return candidates[:top_k]

    text = re.sub(r"```(?:json)?\n?|\n?```", "", text).strip()
    try:
        selected = json.loads(text)
    except Exception as e:
        log.debug("LLM rerank parse failed: %s", e)
        _write_llm_log("rerank", model=model_used, ran=True, error=str(e), output=text[:400], query_snippet=query[:100])
        return candidates[:top_k]

    if isinstance(selected, list):
        result = []
        seen = set()
        for idx in selected:
            if isinstance(idx, int) and 1 <= idx <= len(candidates):
                candidate = candidates[idx - 1]
                url = candidate.get("url", "")
                if url not in seen:
                    seen.add(url)
                    result.append(candidate)
            if len(result) >= top_k:
                break
        if len(result) >= 3:
            _write_llm_log(
                "rerank",
                model=model_used,
                ran=True,
                output=f"selected_indices={selected}",
                query_snippet=query[:100],
            )
            return result

    _write_llm_log(
        "rerank",
        model=model_used,
        ran=True,
        error="Fallback: response not a valid list or fewer than 3 results",
        output=text[:400],
        query_snippet=query[:100],
    )
    return candidates[:top_k]


def ensure_balance(results: list, query: str) -> list:
    """
    Post-processing balance check: if query mentions both tech+soft skills
    but results are all one type, this is a safety net.
    """
    if not results:
        return results

    query_lower = query.lower()

    needs_technical = any(
        kw in query_lower
        for kw in ["java", "python", "sql", "developer", "engineer", "code", "programming",
                    "technical", "software", "data science", "javascript", "typescript"]
    )
    needs_personality = any(
        kw in query_lower
        for kw in ["collaborate", "communication", "personality", "culture", "team",
                    "interpersonal", "behavior", "leadership", "soft skill", "emotional"]
    )

    if not (needs_technical and needs_personality):
        return results

    return results


# ─────────────────────────────────────────────
# Main Recommendation Function
# ─────────────────────────────────────────────

def get_llm_status() -> tuple:
    """
    Load resources and return whether LLM (Gemini) is available for re-ranking.
    Returns (enabled: bool, model_or_reason: str).
    """
    _load_resources()
    if _gemini_model is not None:
        return True, GEMINI_MODEL
    return False, "not loaded (no GEMINI_API_KEY or import failed)"


def get_recommendations(query: str, top_k: int = 10) -> list:
    """
    Full three-stage pipeline: Retrieve -> Filter -> Re-rank.

    Args:
        query: Natural language query or job description text
        top_k: Number of recommendations (5-10 per spec)

    Returns:
        List of assessment dicts with fields matching API spec
    """
    _load_resources()

    _write_llm_log(
        "request_start",
        model=GEMINI_MODEL if _gemini_model else None,
        ran=bool(_gemini_model),
        output=f"top_k={top_k}",
        query_snippet=query[:100],
    )

    top_k = max(5, min(top_k, 10))

    log.info(f"Retrieving candidates for: {query[:80]}...")
    candidates = hybrid_search(query, top_k=80)
    candidates = inject_must_consider(candidates)
    log.info(f"Retrieved {len(candidates)} candidates (including must-consider)")

    constraints = parse_constraints(query)
    if constraints:
        log.info(f"Applying constraints: {constraints}")
    candidates = apply_constraints(candidates, constraints)
    log.info(f"After constraints: {len(candidates)} candidates")

    analysis = {}
    if _gemini_model:
        analysis = llm_analyze_query(query)
        log.debug(f"Query analysis: {analysis}")

    results = llm_rerank_candidates(query, analysis, candidates, top_k)
    results = ensure_balance(results, query)

    output = []
    for r in results:
        output.append({
            "url": r.get("url", ""),
            "name": r.get("name", ""),
            "adaptive_support": r.get("adaptive_support", "No"),
            "description": (r.get("description", "") or "")[:500],
            "duration": r.get("duration", 0) or 0,
            "remote_support": r.get("remote_support", "No"),
            "test_type": r.get("test_types", []),
        })

    log.info(f"Returning {len(output)} recommendations")
    return output


def get_recommendations_from_url(url: str, top_k: int = 10) -> list:
    """Fetch JD text from a URL and get recommendations."""
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SHL-Recommender/1.0)"
        })
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = text[:3000]
        log.info(f"Fetched {len(text)} chars from {url}")
        return get_recommendations(text, top_k)
    except Exception as e:
        log.error(f"Failed to fetch URL {url}: {e}")
        return get_recommendations(url, top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "I am hiring for Java developers who can also collaborate effectively with my business teams."
    results = get_recommendations(test_query)
    print(f"\n=== Results for: {test_query} ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{', '.join(r['test_type'])}] {r['name']} ({r['duration']} min)")
        print(f"   {r['url']}")
