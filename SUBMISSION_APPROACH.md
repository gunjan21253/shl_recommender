# SHL Assessment Recommendation System — Approach & Optimization (2-page submission)

## Problem & Approach

**Goal:** Given a natural language query or job description, return 5–10 relevant SHL Individual Test Solutions (name + URL). The system is evaluated on **Mean Recall@10** and must **balance** recommendations when queries span both technical and behavioral skills.

**Solution:** A three-stage pipeline: (1) **Hybrid retrieval** (FAISS + BM25, RRF merge) → (2) **Constraint filtering** (duration, remote) → (3) **LLM re-ranking** (Gemini) for relevance and balance.

---

## Pipeline Details

- **Stage 1 — Hybrid retrieval:** Sentence-Transformers (`all-mpnet-base-v2`) + FAISS for dense search; BM25 for sparse keyword search. RRF merges rankings. Retrieves top-80 candidates (widened from 50 to improve recall).
- **Stage 2 — Constraints:** Regex-based parsing for max duration and remote support; filters applied with fallback if too few candidates remain.
- **Stage 3 — LLM re-ranking:** Gemini (with fallback models on 429) does (a) query analysis (role, skills, needs_balance) and (b) selection of best 5–10 from the candidate pool. Must-consider injection ensures generic assessments (e.g. OPQ, Verify, English) appear in the pool when relevant.

---

## Performance Optimization Efforts

**Initial baseline (semantic-only, no LLM):** Low recall on keyword-heavy queries; no balance between K (knowledge) and P (personality) types.

**Optimizations applied:**

1. **Hybrid search (FAISS + BM25 + RRF)** — Large recall gain on exact skill names (e.g. SQL, Java).
2. **Stronger embeddings** — Switched to `all-mpnet-base-v2`; better semantic match on multi-skill queries.
3. **Query expansion & rich text** — Expanded query terms and embedded assessment text with test types and metadata.
4. **LLM re-ranking (Gemini)** — Query analysis + re-rank for relevance and balance (K + P when needed).
5. **URL normalization** — Slug-based comparison in evaluation to match training labels (multiple URL formats).
6. **Larger retrieval & must-consider** — Top-80 retrieval, 40 candidates to LLM, injection of generic assessments (OPQ, Verify, English, etc.) so they are not missed.
7. **Prompt and model tuning** — Clarified “avoid only true duplicates”; added few-shot examples; model fallback on 429 (e.g. gemini-2.0-flash).

**Evaluation:** Labeled train set (10 queries, sheet `Train-Set`). Metrics: Mean Recall@10 (primary), Precision@10, NDCG@10. Run:  
`python -m evaluation.evaluate --data "Gen_AI Dataset.xlsx" --sheet "Train-Set" --direct`

---

## Data & Tech Stack

- **Data:** Scraper (requests + BeautifulSoup) crawls SHL product catalog; 377+ Individual Test Solutions; no Pre-packaged Job Solutions. Index build: `python -m recommender.build_index`.
- **Stack:** FastAPI, FAISS, BM25, Sentence-Transformers, Google Gemini API. Frontend: vanilla HTML/JS served by API.

---

## Submission Checklist

- **API:** `GET /health`, `POST /recommend` (JSON; 1–10 assessments with `name`, `url`, etc.).
- **CSV:** `Query`, `Assessment_url` on the 9 test queries (sheet `Test-Set`); generated with  
  `python -m evaluation.generate_predictions --test "Gen_AI Dataset.xlsx" --sheet "Test-Set" --output submission.csv --direct`.
