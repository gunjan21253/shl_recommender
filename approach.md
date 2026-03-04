# Solution Approach: SHL Assessment Recommendation System

## Problem Understanding

The goal is to build a system that takes a natural language query or job description and returns the most relevant SHL Individual Test Solutions (5-10 results). The system must handle diverse queries spanning technical skills ("Java developer"), soft skills ("collaboration"), and mixed requirements ("Java developer who collaborates"). The primary evaluation metric is **Mean Recall@10**.

## Solution Architecture

I designed a **three-stage pipeline** that progressively narrows candidates while improving precision:

### Stage 1: Hybrid Retrieval (Dense + Sparse Search)

Pure semantic search misses exact keyword matches (e.g., "SQL" not matching "SQL" in assessment names), while pure keyword search misses synonyms ("programmer" vs "developer"). I combined both:

- **FAISS Dense Search:** Sentence-Transformers (`all-mpnet-base-v2`, 768-dim) encodes assessments and queries into embeddings. FAISS IndexFlatIP provides fast cosine similarity search.
- **BM25 Sparse Search:** Classic keyword matching using `rank_bm25`. Catches exact skill names that semantic search may rank lower.
- **Reciprocal Rank Fusion (RRF):** Merges both ranked lists using the formula `RRF(doc) = sum(1/(rank + k))`, avoiding the need to tune combination weights. This is robust and parameter-free (k=60 per the original paper).

The hybrid approach retrieves the **top-30 candidates** for downstream processing.

### Stage 2: Constraint Filtering

Queries sometimes contain hard constraints: "assessments under 40 minutes" or "remote testing required." I parse these using regex patterns and apply them as hard filters, with a graceful fallback that relaxes constraints if too few candidates remain (minimum 5).

### Stage 3: LLM Re-ranking (Gemini 1.5 Flash)

The top candidates are re-ranked by Google Gemini (free tier) in two steps:

1. **Query Analysis:** The LLM identifies the job role, technical skills, soft skills, experience level, and whether the query needs balanced results (both K-type and P-type assessments).
2. **Re-ranking:** Given the candidates and analysis, the LLM selects and orders the best 5-10 assessments, ensuring diversity and balance between assessment types.

## Key Optimizations and Iteration

### Initial Baseline
- Semantic-only search with `all-MiniLM-L6-v2`
- No query expansion, no BM25, no LLM re-ranking
- **Result:** Low recall on keyword-heavy queries (e.g., "SQL" returning irrelevant results)

### Optimization 1: Hybrid Search (+BM25)
- Added BM25 sparse search alongside FAISS dense search
- Used RRF to merge results
- **Impact:** Significant recall improvement on queries with specific skill names

### Optimization 2: Better Embeddings
- Switched from `all-MiniLM-L6-v2` (384-dim) to `all-mpnet-base-v2` (768-dim)
- Higher quality embeddings improved semantic matching
- **Impact:** Better recall on complex, multi-skill queries

### Optimization 3: Query Expansion
- Added domain-specific synonym expansion (e.g., "Java" expands to "Java Spring Boot backend developer programming")
- Bridges vocabulary gap between queries and assessment descriptions
- **Impact:** Improved recall for role-based queries

### Optimization 4: Rich Embedding Text
- Instead of embedding just assessment names, I create information-dense text including expanded test type descriptions, duration, remote support, and inferred role keywords
- **Impact:** Better semantic matching across diverse query types

### Optimization 5: LLM Re-ranking for Balance
- Added Gemini-based re-ranking specifically to handle the "balance" requirement
- Queries like "Java developer who collaborates" now return both K (technical) and P (personality) assessments
- **Impact:** Directly addresses the evaluation criterion on recommendation balance

### Optimization 6: URL Normalization
- Training labels use inconsistent URL formats (`/products/...` vs `/solutions/products/...`)
- Normalized all URLs to slug-based comparison, eliminating false negatives in evaluation
- **Impact:** Significant improvement in measured recall (many "misses" were actually URL format mismatches)

## Data Pipeline

1. **Scraping:** Playwright-based scraper navigates the SHL product catalog, paginates through all results, and visits each detail page for descriptions and durations. Scraped 377+ Individual Test Solutions.
2. **Indexing:** Each assessment is encoded into a rich text representation, then embedded using `all-mpnet-base-v2`. Both FAISS and BM25 indexes are built and persisted to disk.
3. **Serving:** FastAPI loads indexes at startup. Each query goes through the three-stage pipeline in ~2-5 seconds.

## Evaluation

I evaluated using the provided labeled training set (10 queries) with three metrics:
- **Mean Recall@10** (primary): Measures how many relevant assessments appear in the top 10
- **Mean Precision@10**: Measures what fraction of the top 10 are relevant
- **Mean NDCG@10**: Rewards correct answers ranked higher

The evaluation script supports both direct engine evaluation and API-based evaluation, with detailed per-query breakdowns showing hits and misses.

## Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| Embedding Model | `all-mpnet-base-v2` | Best quality free model from sentence-transformers |
| Vector Store | FAISS (IndexFlatIP) | Fast, reliable, no external dependencies |
| Keyword Search | BM25Okapi | Standard, effective sparse retrieval |
| LLM | Gemini 1.5 Flash | Free tier, fast, good instruction following |
| API Framework | FastAPI | Modern, async, auto-generates OpenAPI docs |
| Scraper | Playwright | Handles JS-rendered pages (SHL catalog uses dynamic loading) |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, deploys anywhere, serves from FastAPI |
