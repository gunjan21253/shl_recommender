# SHL Recommendation System — Detailed Explanation

This document describes **every file**, **every function**, and **every library/model** used in the SHL Assessment Recommendation project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Root & Configuration Files](#2-root--configuration-files)
3. [API Module (`api/`)](#3-api-module-api)
4. [Recommender Module (`recommender/`)](#4-recommender-module-recommender)
5. [Scraper Module (`scraper/`)](#5-scraper-module-scraper)
6. [Evaluation Module (`evaluation/`)](#6-evaluation-module-evaluation)
7. [Frontend (`frontend/`)](#7-frontend-frontend)
8. [Data Directory (`data/`)](#8-data-directory-data)
9. [Libraries & Models Reference](#9-libraries--models-reference)

---

## 1. Project Overview

The system recommends SHL Individual Test Solutions (assessments) from natural language queries or job descriptions. It uses a **three-stage pipeline**:

1. **Hybrid retrieval** — FAISS (dense) + BM25 (sparse), merged with Reciprocal Rank Fusion.
2. **Constraint filtering** — Duration and remote/adaptive requirements parsed from the query.
3. **LLM re-ranking** — Google Gemini 1.5 Flash selects and orders the best 5–10 assessments.

Primary evaluation metric: **Mean Recall@10**.

---

## 2. Root & Configuration Files

### 2.1 `requirements.txt`

**Purpose:** Python dependency list for pip.

| Package | Version | Role |
|--------|---------|------|
| **fastapi** | ≥0.100.0 | Web API framework |
| **uvicorn[standard]** | ≥0.20.0 | ASGI server to run FastAPI |
| **pydantic** | ≥2.0.0 | Request/response validation and serialization |
| **python-dotenv** | ≥1.0.0 | Load `.env` (e.g. `GEMINI_API_KEY`) |
| **faiss-cpu** | ≥1.7.0 | Dense vector similarity search (CPU) |
| **sentence-transformers** | ≥2.2.0 | Text → embedding models (e.g. all-mpnet-base-v2) |
| **rank-bm25** | ≥0.2.2 | BM25 sparse retrieval |
| **numpy** | ≥1.24.0 | Arrays and numerical ops for FAISS/embeddings |
| **playwright** | ≥1.40.0 | Browser automation for scraping JS-rendered pages |
| **beautifulsoup4** | ≥4.12.0 | HTML parsing (e.g. for URL content) |
| **requests** | ≥2.31.0 | HTTP client (fetching job description URLs) |
| **google-generativeai** | ≥0.3.0 | Google Gemini API client |
| **pandas** | ≥2.0.0 | CSV handling in evaluation and prediction scripts |

---

### 2.2 `.env.example`

**Purpose:** Template for environment variables. Copy to `.env` and fill in.

- **GEMINI_API_KEY** — Google Gemini API key (required for LLM re-ranking). Get at https://ai.google.dev/gemini-api/docs  
- **PORT** — Optional; server port (default 8000).

---

### 2.3 `.gitignore`

**Purpose:** Exclude from git: `__pycache__`, `*.pyc`, `venv`, `.env`, IDE/config files, swap files, `.DS_Store`, `Thumbs.db`, `*.log`.

---

### 2.4 `Procfile`

**Purpose:** Declares the web process for Heroku/Render-style platforms.

- **Single command:** `web: uvicorn api.main:app --host 0.0.0.0 --port $PORT`  
- Starts the FastAPI app with Uvicorn; `$PORT` is set by the platform.

---

### 2.5 `render.yaml`

**Purpose:** Render.com service definition.

- **Type:** `web`  
- **Name:** `shl-recommender`  
- **Runtime:** python  
- **buildCommand:** `pip install -r requirements.txt`  
- **startCommand:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`  
- **envVars:** `GEMINI_API_KEY` (sync: false — user must set), `PYTHON_VERSION`: `"3.11.0"`

---

## 3. API Module (`api/`)

### 3.1 `api/__init__.py`

**Purpose:** Empty package marker; makes `api` a Python package.

---

### 3.2 `api/main.py`

**Purpose:** FastAPI app: health check, recommend endpoint, frontend serving, CORS, logging middleware, and global exception handler.

#### Imports and setup

- **os** — Environment (e.g. `PORT`).  
- **sys**, **pathlib.Path** — Add project root to `sys.path` so `recommender` can be imported.  
- **logging** — Application logging.  
- **time** — Request timing in middleware.  
- **contextlib.asynccontextmanager** — Lifespan context for startup/shutdown.  
- **typing.List** — Type hints.  
- **dotenv.load_dotenv** — Load `.env`.  
- **fastapi** — `FastAPI`, `HTTPException`, `Request`.  
- **fastapi.middleware.cors.CORSMiddleware** — CORS.  
- **fastapi.responses** — `JSONResponse`, `FileResponse`.  
- **pydantic** — `BaseModel`, `field_validator` for request/response models.

#### Pydantic models

- **QueryRequest**  
  - **Field:** `query: str`.  
  - **Validator `query_not_empty`:** Strips whitespace; raises if empty.  
  - Used as body for `POST /recommend`.

- **AssessmentResult**  
  - **Fields:** `url`, `name`, `adaptive_support`, `description`, `duration` (int), `remote_support`, `test_type` (List[str]).  
  - One recommended assessment in the response.

- **RecommendResponse**  
  - **Field:** `recommended_assessments: List[AssessmentResult]`.  
  - Response model for `POST /recommend`.

#### Functions

- **lifespan(app)**  
  - Async context manager used as FastAPI `lifespan`.  
  - On startup: calls `recommender.engine._load_resources()` to load FAISS, assessments, BM25, embedding model, and optionally Gemini.  
  - On failure: logs error and continues (engine can still load on first request).  
  - On shutdown: logs.

- **log_requests(request, call_next)**  
  - HTTP middleware. Records method, path, status code, and duration for every request, then returns the response.

- **serve_frontend()**  
  - **Route:** `GET /` (excluded from OpenAPI).  
  - If `frontend/index.html` exists, returns it as `FileResponse` (HTML).  
  - Otherwise returns JSON with message and docs link.

- **health_check()**  
  - **Route:** `GET /health`.  
  - Returns `{"status": "healthy"}`.

- **recommend(request: QueryRequest)**  
  - **Route:** `POST /recommend`.  
  - Trims `request.query`.  
  - If query starts with `http://` or `https://`: calls `get_recommendations_from_url(query, top_k=10)` (fetches page text, then recommends).  
  - Else: calls `get_recommendations(query, top_k=10)`.  
  - Maps engine results to `AssessmentResult` (description truncated to 500 chars, duration as int).  
  - Returns `RecommendResponse(recommended_assessments=cleaned)`.  
  - On empty results: raises `HTTPException(500)`.

- **generic_exception_handler(request, exc)**  
  - Catches unhandled `Exception`. Logs with traceback, returns `JSONResponse` 500 with `detail` and `error` message.

#### Entry point

- **`if __name__ == "__main__"`**  
  - Runs `uvicorn.run("api.main:app", host="0.0.0.0", port from `PORT` or 8000, reload=True, log_level="info")`.

---

## 4. Recommender Module (`recommender/`)

### 4.1 `recommender/__init__.py`

**Purpose:** Empty package marker.

---

### 4.2 `recommender/engine.py`

**Purpose:** Three-stage recommendation engine: hybrid retrieval → constraint filtering → LLM re-ranking. Uses FAISS, BM25, sentence-transformers, and optionally Gemini.

#### Constants and globals

- **DATA_DIR** — `Path(__file__).parent.parent / "data"`.  
- **QUERY_EXPANSION_MAP** — Dict mapping keywords (e.g. `"java"`, `"python"`) to expanded phrases for better retrieval.  
- Module-level globals: `_embedding_model`, `_faiss_index`, `_bm25_index`, `_assessments`, `_gemini_model`.

#### Functions

- **\_load_resources()**  
  - Loads all engine resources once (idempotent if already loaded).  
  - Reads `data/shl_index.faiss` with `faiss.read_index`.  
  - Loads `data/assessments.pkl` and `data/bm25_index.pkl` with `pickle`.  
  - Instantiates `SentenceTransformer("all-mpnet-base-v2")`.  
  - If `GEMINI_API_KEY` is set, configures `google.generativeai` and sets `_gemini_model = genai.GenerativeModel("gemini-1.5-flash")`.  
  - Raises `FileNotFoundError` if FAISS index is missing (hint: run `python -m recommender.build_index`).

- **normalize_url(url)**  
  - Normalizes SHL URLs: strip, lower, ensure `https://`, and map `/products/product-catalog/` to `/solutions/products/product-catalog/` for consistent comparison.

- **expand_query(query)**  
  - For each key in `QUERY_EXPANSION_MAP` that appears in lowercased query, appends the corresponding expansion (if not already in the string).  
  - Returns the expanded query string for better semantic/keyword recall.

- **dense_search(query, top_k=40)**  
  - Expands query, encodes it with `_embedding_model.encode(..., normalize_embeddings=True)`, runs `_faiss_index.search(embedding, top_k)`.  
  - Returns list of `(score, assessment_index)` for valid indices.

- **sparse_search(query, top_k=40)**  
  - Tokenizes expanded query (lowercase, words/digits, length > 1), gets scores from `_bm25_index.get_scores(tokens)`, sorts by score descending.  
  - Returns list of `(score, index)` for indices with score > 0.

- **reciprocal_rank_fusion(ranked_lists, k=60)**  
  - Implements RRF: for each list, for each document at rank r (0-based), adds `1 / (r + 1 + k)` to that document’s total score.  
  - Returns merged list of `(rrf_score, idx)` sorted by score descending.

- **hybrid_search(query, top_k=30)**  
  - Runs `dense_search(query, 40)` and `sparse_search(query, 40)`, merges with `reciprocal_rank_fusion`, takes top `top_k`.  
  - Returns list of assessment dicts with `_retrieval_score` and `_index` added.

- **parse_constraints(query)**  
  - Extracts constraints from query text using regex:  
    - **max_duration** — e.g. "X minutes", "X hour(s)", "about an hour", "within an hour", "completed in ... X".  
    - **remote_support** — if "remote" in query, sets `"Yes"`.  
  - Returns dict of constraints.

- **apply_constraints(candidates, constraints)**  
  - Filters candidates by `max_duration` and/or `remote_support` if present in constraints.  
  - Keeps at least 5 candidates: only applies a filter if the filtered list has ≥ 5 items.

- **llm_analyze_query(query)**  
  - If `_gemini_model` is None, returns `{}`.  
  - Sends a prompt asking for JSON: job_role, technical_skills, soft_skills, experience_level, max_duration_minutes, needs_balance, primary_assessment_types, explanation.  
  - Strips markdown code fences from response and parses JSON.  
  - On failure returns `{}`.

- **llm_rerank_candidates(query, analysis, candidates, top_k)**  
  - If no Gemini or no candidates, returns first `top_k` candidates.  
  - Builds a numbered list of up to 20 candidates (test types, name, duration, remote, short description).  
  - If `analysis.get("needs_balance")` is true, adds instruction to balance technical (K) and personality (P) types.  
  - Asks Gemini to return a JSON array of candidate numbers in relevance order.  
  - Maps numbers back to candidates, deduplicates by URL, returns up to `top_k`.  
  - On parse/API failure, falls back to `candidates[:top_k]`.

- **ensure_balance(results, query)**  
  - Placeholder: checks if query needs both technical and personality terms; currently returns `results` unchanged (balance is enforced in LLM re-ranking).

- **get_recommendations(query, top_k=10)**  
  - Main entry. Calls `_load_resources()`, clamps `top_k` to 5–10.  
  - Runs `hybrid_search(query, 30)` → `parse_constraints` → `apply_constraints` → `llm_analyze_query` (if Gemini) → `llm_rerank_candidates` → `ensure_balance`.  
  - Maps each result to the API shape: url, name, adaptive_support, description (truncated 500), duration, remote_support, test_type (from `test_types`).  
  - Returns list of dicts.

- **get_recommendations_from_url(url, top_k=10)**  
  - Uses `requests.get` and `BeautifulSoup` to fetch URL, strip script/style/nav/header/footer, get text (first 3000 chars).  
  - Calls `get_recommendations(text, top_k)`.  
  - On fetch error, falls back to `get_recommendations(url, top_k)`.

#### Script block

- **`if __name__ == "__main__"`** — Sets logging, runs `get_recommendations` on a sample query and prints results.

---

### 4.3 `recommender/build_index.py`

**Purpose:** Builds FAISS dense index and BM25 sparse index from the scraped catalog and saves them plus assessments and texts to `data/`.

#### Constants

- **DATA_DIR**, **CATALOG_FILE** (`data/shl_catalog.json`), **INDEX_FILE** (`data/shl_index.faiss`), **ASSESSMENTS_FILE** (`data/assessments.pkl`), **BM25_FILE** (`data/bm25_index.pkl`), **TEXTS_FILE** (`data/texts.pkl`).  
- **EMBEDDING_MODEL** — `"all-mpnet-base-v2"`.  
- **TEST_TYPE_MAP** — Letters (A, B, C, …) to short descriptions (e.g. "Ability and Aptitude cognitive reasoning").  
- **ROLE_KEYWORDS** — Role words (e.g. "developer") to skill/role expansion text.

#### Functions

- **clean_text(text)**  
  - Collapses whitespace and strips; returns cleaned string.

- **create_embedding_text(assessment)**  
  - Builds one concatenated text per assessment: name, test type descriptions from TEST_TYPE_MAP, description, duration, remote/adaptive, languages, and inferred role keywords from ROLE_KEYWORDS.  
  - Joins with `" | "`, then `clean_text`. Used for both embedding and (via tokenization) BM25.

- **create_bm25_tokens(text)**  
  - Lowercases, extracts words/digits with regex `\b[a-z0-9]+\b`, filters to length > 1.  
  - Returns list of tokens for BM25.

- **build_indexes(assessments)**  
  - Builds texts with `create_embedding_text`, loads `SentenceTransformer(EMBEDDING_MODEL)`, encodes all texts (batch 32, normalized).  
  - Builds `faiss.IndexFlatIP(dimension)` and adds embeddings.  
  - Tokenizes texts with `create_bm25_tokens` and builds `BM25Okapi(tokenized_texts)`.  
  - Writes FAISS index, pickles assessments, BM25 index, and texts to DATA_DIR.  
  - Returns `(faiss_index, bm25_index, assessments)`.

- **main()**  
  - Requires `CATALOG_FILE`. Loads JSON catalog, optionally warns if &lt; 100 assessments, calls `build_indexes(assessments)`.

#### Script block

- **`if __name__ == "__main__"`** — Calls `main()`.

---

## 5. Scraper Module (`scraper/`)

### 5.1 `scraper/__init__.py`

**Purpose:** Empty package marker.

---

### 5.2 `scraper/catalog_scraper.py`

**Purpose:** Scrape SHL Individual Test Solutions from the product catalog using Playwright (JS-rendered pages), then optionally visit each detail page for description and duration.

#### Constants

- **CATALOG_URL** — SHL product catalog URL.  
- **OUTPUT_FILE** — `data/shl_catalog.json`.  
- **TEST_TYPE_MAP** — Letter codes (A–S) to human-readable type names.

#### Functions

- **normalize_url(url)**  
  - Strips, rstrip "/", forces `https://`.  
  - Same idea as in engine but without the path rewrite (scraper outputs raw URLs).

- **parse_duration(text)**  
  - Extracts hours (e.g. "X hour") and minutes (e.g. "X min") with regex; converts to total minutes.  
  - If no hour/min match, uses first integer in text.  
  - Returns 0 if no number found.

- **scrape_detail_page(page, url)**  
  - Navigates to assessment detail URL with Playwright, waits.  
  - Tries multiple selectors for description (e.g. `.product-catalogue-training-catalogue__description`, `.product-hero__description`, etc.).  
  - Tries selectors for duration/timing, then falls back to body-text regex for "X min".  
  - Collects languages from a `[class*='language']` element.  
  - Returns `{"description", "duration", "languages"}`.  
  - On timeout or exception returns empty/defaults.

- **scrape_catalog_page(page)**  
  - Waits for catalog table/row selectors.  
  - Gets all table rows (or product-catalogue rows).  
  - For each row: finds link (name, href), normalizes URL, collects test-type badges from cells, infers remote_support and adaptive_support from cell content (e.g. "circle--yes", "check").  
  - Returns list of assessment dicts (name, url, test_types, test_types_expanded, remote_support, adaptive_support, description="", duration=0, languages=[]).

- **click_individual_test_filter(page)**  
  - Tries several selectors (e.g. "text=Individual Test", data-filter, etc.) to click the "Individual Test Solutions" filter.  
  - Returns True if clicked, False otherwise.

- **scrape_all(skip_details=False, max_pages=50)**  
  - Uses `sync_playwright()`, launches Chromium headless, opens catalog page, applies Individual Test filter.  
  - Paginates: on each page calls `scrape_catalog_page`, deduplicates by URL, collects assessments.  
  - Stops when no rows, no "next" button, or next disabled, or after `max_pages`.  
  - If not `skip_details`, visits each assessment URL with a second page and updates with `scrape_detail_page` result (with small sleep).  
  - Returns list of assessment dicts.

- **main()**  
  - Ensures `OUTPUT_FILE.parent` exists.  
  - Calls `scrape_all(skip_details=False)`, writes JSON to OUTPUT_FILE, logs type distribution.

#### Script block

- **`if __name__ == "__main__"`** — Calls `main()`.

---

## 6. Evaluation Module (`evaluation/`)

### 6.1 `evaluation/__init__.py`

**Purpose:** Empty package marker.

---

### 6.2 `evaluation/evaluate.py`

**Purpose:** Compute Mean Recall@K, Precision@K, and NDCG@K over a labeled CSV (query → list of relevant URLs) by calling the recommender (direct or via API).

#### URL normalization

- **normalize_url(url)**  
  - Strips, lowercases, strips trailing "/".  
  - Extracts slug with regex `r"/view/([^/?#]+)"` for consistent comparison with training labels.  
  - Returns slug or original if no match.

#### Metrics

- **recall_at_k(predicted_urls, relevant_urls, k=10)**  
  - Recall@K = |predicted_top_k ∩ relevant| / |relevant|, using normalized URLs (slugs).

- **precision_at_k(predicted_urls, relevant_urls, k=10)**  
  - Precision@K = |predicted_top_k ∩ relevant| / min(k, len(predicted_urls)), slug-based.

- **ndcg_at_k(predicted_urls, relevant_urls, k=10)**  
  - DCG = sum over top-k of (1 / log2(rank+2)) for relevant docs; IDCG = same for ideal ordering.  
  - Returns DCG/IDCG (0 if IDCG=0).

#### Data and prediction

- **load_ground_truth(csv_path)**  
  - Reads CSV with pandas, detects query and URL columns (by name), builds dict `{query: [url1, url2, ...]}`.  
  - Returns that dict.

- **predict_via_api(query, api_url)**  
  - POSTs `{"query": query}` to `{api_url}/recommend`, returns list of assessment URLs from response.

- **predict_direct(query)**  
  - Calls `recommender.engine.get_recommendations(query, top_k=10)`, returns list of URLs.

#### Main evaluation

- **evaluate(csv_path, api_url=None, direct=False, k=10, verbose=True)**  
  - Loads ground truth, then for each query: gets predictions (direct or via API), computes recall@k, precision@k, ndcg@k.  
  - Aggregates mean recall, mean precision, mean ndcg; optionally logs per-query hits/misses.  
  - Returns dict: mean_recall, mean_precision, mean_ndcg, per_query list (query, recall, precision, ndcg, n_relevant, n_predicted, predicted_urls).

- **main()**  
  - Parses args: `--csv`, `--api`, `--direct`, `--k`, `--output`.  
  - Calls `evaluate()`, optionally saves results JSON to `--output`.

---

### 6.3 `evaluation/generate_predictions.py`

**Purpose:** Run test-set queries through the recommender (direct or API) and write a submission CSV: `query, Assessment_url` (one row per query–URL pair).

#### Functions

- **load_test_queries(csv_path)**  
  - Reads CSV, finds query column (by name), drops NaNs, unique, strips.  
  - Returns list of query strings.

- **get_predictions_via_api(query, api_url)**  
  - POSTs to `{api_url}/recommend`, returns list of assessment URLs.

- **get_predictions_direct(query)**  
  - Calls `recommender.engine.get_recommendations(query, top_k=10)`, returns list of URLs.

- **generate_predictions(test_csv, output_csv, api_url=None, direct=False, delay=1.0)**  
  - Loads queries from test CSV.  
  - For each query, gets URLs (direct or API; optional `time.sleep(delay)` between API calls).  
  - Builds rows `{"query": query, "Assessment_url": url}` for each URL.  
  - Writes DataFrame to CSV, logs and prints first 20 rows.  
  - Returns the DataFrame.

- **main()**  
  - Parses `--test`, `--output`, `--api`, `--direct`, `--delay`.  
  - Calls `generate_predictions()`.

---

## 7. Frontend (`frontend/`)

### 7.1 `frontend/index.html`

**Purpose:** Single-page UI: textarea for query/URL, “Get Recommendations” button, sample chips, results table, error/loading states. No build step; vanilla HTML/CSS/JS.

#### Structure

- **Head:** charset, viewport, title “SHL Assessment Recommender”, one large `<style>` block.  
- **Body:** header, container with card (textarea, button, timing span, sample chips), error div, loading div (spinner + text), results section (count + table), footer with API docs and health links.

#### CSS (summary)

- **:root** — Primary blue, surface, text, border, success, radius, shadow.  
- **Layout** — Reset, body font and background, header gradient, container max-width, card styling.  
- **Form** — Textarea focus border, button and disabled state, sample chips.  
- **Loading** — Spinner animation, hidden by default, `.active` to show.  
- **Results table** — Header row, badges per test type (K, P, A, … with distinct colors), support yes/no styling.  
- **Error** — Red border/background, shown with `.active`.  
- **Footer** — Muted text, links.  
- **Media** — Simpler layout and horizontal scroll for table on small screens.

#### JavaScript

- **API_BASE** — `window.location.origin`.  
- **TYPE_NAMES** — Map of type code (K, P, A, …) to display names.  
- **useSample(el)** — Sets textarea to chip text and calls `search()`.  
- **search()**  
  - Validates non-empty query, disables button, shows loading, hides results/error.  
  - POSTs `{ query }` to `API_BASE/recommend`, measures time with `performance.now()`.  
  - On success: shows timing, calls `renderResults(data.recommended_assessments)`.  
  - On error: `showError(message)`.  
  - Re-enables button and hides loading in `finally`.  
- **renderResults(assessments)**  
  - Clears tbody, sets count.  
  - For each assessment: badges from `test_type`, description (escaped), duration, remote/adaptive (Yes/No with class).  
  - Rows: #, assessment name (link) + description, test type badges, duration, remote, adaptive.  
  - Shows results container.  
- **showError(msg)** — Sets error div text and adds `.active`.  
- **escHtml(str)** — Creates a div, sets textContent, returns innerHTML to escape HTML.  
- **keydown** on textarea — Ctrl+Enter or Cmd+Enter triggers `search()`.

---

## 8. Data Directory (`data/`)

- **`.gitkeep`** — Keeps `data/` in git when empty.  
- **Generated (not in repo by default):**  
  - **shl_catalog.json** — Output of scraper; list of assessment objects.  
  - **shl_index.faiss** — FAISS index from `build_index`.  
  - **assessments.pkl** — List of assessment dicts used at runtime.  
  - **bm25_index.pkl** — BM25Okapi instance.  
  - **texts.pkl** — List of embedding texts (optional cache).  
- **train.csv / test.csv** — Used by evaluation and generate_predictions (paths passed as args).

---

## 9. Libraries & Models Reference

### 9.1 FastAPI

- **What:** Async web framework for APIs.  
- **Use here:** App definition, lifespan, routes (`/`, `/health`, `/recommend`), middleware, exception handler, Pydantic request/response models, OpenAPI docs at `/docs`.

### 9.2 Uvicorn

- **What:** ASGI server.  
- **Use here:** Run `api.main:app` with host/port/reload; referenced in Procfile and render.yaml.

### 9.3 Pydantic (v2)

- **What:** Data validation and serialization via type hints and validators.  
- **Use here:** `QueryRequest`, `AssessmentResult`, `RecommendResponse`; `field_validator` for non-empty query.

### 9.4 python-dotenv

- **What:** Loads `.env` into `os.environ`.  
- **Use here:** `load_dotenv()` in `api/main.py` so `GEMINI_API_KEY` and `PORT` are available.

### 9.5 FAISS (faiss-cpu)

- **What:** Facebook AI Similarity Search — library for similarity search over vectors (here, CPU only).  
- **Use here:** `IndexFlatIP` (inner product on normalized vectors ≈ cosine similarity).  
- **Index:** Built in `build_index` from 768-dim embeddings; read in `engine._load_resources()` and queried in `dense_search`.

### 9.6 Sentence-Transformers

- **What:** Library to encode text into dense vectors using pre-trained models.  
- **Model used:** **all-mpnet-base-v2**  
  - 768-dimensional embeddings, good quality for semantic similarity.  
  - Used in `build_index` to encode assessment texts and in `engine` to encode the (expanded) query.  
  - Encodings are L2-normalized for use with inner-product index in FAISS.

### 9.7 rank_bm25 (BM25Okapi)

- **What:** BM25 ranking for sparse, keyword-style retrieval.  
- **Use here:** Corpus = tokenized assessment texts (from `create_embedding_text`); query = tokenized expanded query.  
- **get_scores(tokens):** Returns per-document scores for the query tokens.  
- Used in `build_index` to build the index and in `engine.sparse_search` for retrieval.

### 9.8 NumPy

- **What:** Array and numerical operations.  
- **Use here:** Embedding arrays (float32) for FAISS; `np.argsort` in sparse_search; general array handling in index building and search.

### 9.9 Playwright

- **What:** Browser automation (Chromium/ Firefox/ WebKit).  
- **Use here:** `sync_playwright()` and Chromium to open SHL catalog (JS-rendered), paginate, click filter, and open detail pages; `page.goto`, `query_selector`, `inner_text`, etc.

### 9.10 Beautiful Soup (bs4)

- **What:** HTML/XML parsing.  
- **Use here:** In `engine.get_recommendations_from_url`: parse fetched HTML, remove script/style/nav/header/footer, extract text for the job description.

### 9.11 Requests

- **What:** HTTP client.  
- **Use here:** In `engine.get_recommendations_from_url` to fetch JD URL; in `evaluate.py` and `generate_predictions.py` to call the recommend API when using `--api`.

### 9.12 Google Generative AI (Gemini)

- **What:** Client for Google’s Gemini API.  
- **Model used:** **gemini-1.5-flash**  
  - Fast, instruction-following model (free tier).  
- **Use here:**  
  - **llm_analyze_query:** One prompt to get structured JSON (job role, skills, needs_balance, etc.).  
  - **llm_rerank_candidates:** Second prompt to select and order top 5–10 from candidate list, with optional balance instruction.  
- **Config:** `genai.configure(api_key=os.environ["GEMINI_API_KEY"])`; if key missing, re-ranking is skipped.

### 9.13 Pandas

- **What:** DataFrames and CSV I/O.  
- **Use here:**  
  - **evaluate.py:** Read ground-truth CSV, infer columns.  
  - **generate_predictions.py:** Read test CSV for queries, write predictions CSV.

### 9.14 Standard library

- **json** — Load catalog, parse LLM JSON.  
- **pickle** — Save/load assessments, BM25 index, texts.  
- **re** — URL normalization, constraint parsing, duration parsing, tokenization.  
- **logging** — All modules.  
- **argparse** — evaluate and generate_predictions CLIs.  
- **pathlib** — Paths for data dir and files.  
- **collections.defaultdict** — evaluate ground truth aggregation.

---

## Summary Table: File → Main Responsibility

| File | Main responsibility |
|------|----------------------|
| `requirements.txt` | Declare Python dependencies |
| `.env.example` | Env var template (GEMINI_API_KEY, PORT) |
| `.gitignore` | Ignore venv, cache, .env, IDE, logs |
| `Procfile` | Web process for PaaS |
| `render.yaml` | Render.com service config |
| `api/__init__.py` | Package marker |
| `api/main.py` | FastAPI app, /health, /recommend, frontend, CORS, lifespan |
| `recommender/__init__.py` | Package marker |
| `recommender/engine.py` | Load resources; hybrid search; constraints; Gemini analyze/rerank; get_recommendations |
| `recommender/build_index.py` | Build FAISS + BM25 from catalog; save to data/ |
| `scraper/__init__.py` | Package marker |
| `scraper/catalog_scraper.py` | Playwright scrape catalog + detail pages → JSON |
| `evaluation/__init__.py` | Package marker |
| `evaluation/evaluate.py` | Recall/Precision/NDCG@K vs labeled CSV (direct or API) |
| `evaluation/generate_predictions.py` | Run test queries → submission CSV |
| `frontend/index.html` | Single-page UI: query input, samples, results table |
| `data/.gitkeep` | Keep data/ in git |

This completes the detailed explanation of every file, function, and library/model in the project.
