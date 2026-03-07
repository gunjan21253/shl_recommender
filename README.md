# SHL Assessment Recommendation System

An intelligent recommendation engine that finds relevant SHL Individual Test Solutions from natural language queries or job descriptions. Built with **Hybrid Search (FAISS + BM25) + LLM Re-ranking** for high Recall@10.

**Live Demo:** https://shl-recommender-tkj0.onrender.com/
**API Endpoint:** https://shl-recommender-tkj0.onrender.com/recommend

---

## Implementation and evaluation (in this repo)

This repository contains both the **implementation** and **evaluation** as required:

| Part | Location | What it does |
|------|----------|--------------|
| **Implementation** | `api/`, `recommender/`, `scraper/`, `frontend/` | Full pipeline: scraping, indexing, hybrid search, LLM re-ranking, FastAPI + UI. |
| **Evaluation** | `evaluation/` | Scripts to measure and iterate on the system. |

**Evaluation scripts:**

- **`evaluation/evaluate.py`** — Computes Mean Recall@10, Precision@10, NDCG@10 on the labeled train set (10 queries, sheet `Train-Set`). Run:  
  `python -m evaluation.evaluate --data "Gen_AI Dataset.xlsx" --sheet "Train-Set" --direct`
- **`evaluation/generate_predictions.py`** — Generates the submission CSV for the 9 test queries (sheet `Test-Set`). Run:  
  `python -m evaluation.generate_predictions --test "Gen_AI Dataset.xlsx" --sheet "Test-Set" --output submission.csv --direct`
- **`evaluation/diagnose.py`** — Diagnoses retrieval vs ranking misses (optional). Run:  
  `python -m evaluation.diagnose --data "Gen_AI Dataset.xlsx" --sheet "Train-Set"`

Train/test data: `Gen_AI Dataset.xlsx` (sheets `Train-Set` and `Test-Set`). Place it in the project root to run evaluation.

---

## Architecture

```
User Query / Job Description / URL
               |
  +============v=============+
  | STAGE 1: HYBRID RETRIEVAL |
  |                           |
  |  FAISS Dense    BM25      |
  |  (semantic)    (keyword)  |
  |       |            |      |
  |       +-- RRF Merge --+   |
  |            |              |
  |       Top 30 candidates   |
  +============|=============+
               |
  +============v==============+
  | STAGE 2: CONSTRAINT FILTER |
  | Duration, Remote prefs     |
  +============|==============+
               |
  +============v==============+
  | STAGE 3: LLM RE-RANKING   |
  | Gemini 1.5 Flash           |
  | Balance K + P types         |
  | Return top 5-10             |
  +============================+
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get a free key: https://ai.google.dev/gemini-api/docs
```

### 3. Scrape SHL catalog

```bash
python -m scraper.catalog_scraper
# Output: data/shl_catalog.json (377+ assessments)
# Takes ~10-20 minutes
```

### 4. Build search indexes

```bash
python -m recommender.build_index
# Output: data/shl_index.faiss, data/assessments.pkl, data/bm25_index.pkl
# Takes ~5 minutes (downloads embedding model + encodes)
```

### 5. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Frontend: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 6. Evaluate (Train-Set, 10 labeled queries)

```bash
# Direct (no API)
python -m evaluation.evaluate --data "Gen_AI Dataset.xlsx" --sheet "Train-Set" --direct

# Via API
python -m evaluation.evaluate --data "Gen_AI Dataset.xlsx" --sheet "Train-Set" --api http://localhost:8000
```

### 7. Generate submission CSV (Test-Set, 9 queries)

```bash
python -m evaluation.generate_predictions --test "Gen_AI Dataset.xlsx" --sheet "Test-Set" --output submission.csv --direct
```

---

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{"status": "healthy"}
```

### Recommend Assessments

```
POST /recommend
Content-Type: application/json
```

Request body:
```json
{"query": "I am hiring Java developers who can collaborate with teams"}
```

Response:
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "name": "Core Java (Entry Level) (New)",
      "adaptive_support": "No",
      "description": "Multi-choice test measuring Java knowledge...",
      "duration": 12,
      "remote_support": "Yes",
      "test_type": ["K"]
    }
  ]
}
```

---

## Project Structure

```
shl_recommendation/
├── api/
│   ├── __init__.py
│   └── main.py                   # FastAPI endpoints + frontend serving
├── recommender/
│   ├── __init__.py
│   ├── engine.py                 # Three-stage recommendation pipeline
│   └── build_index.py            # FAISS + BM25 index builder
├── scraper/
│   ├── __init__.py
│   └── catalog_scraper.py        # SHL catalog scraper (requests + BeautifulSoup)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py               # Recall@10, Precision@10, NDCG@10
│   └── generate_predictions.py   # Test set prediction CSV generator
├── frontend/
│   └── index.html                # Single-page web UI
├── data/                         # Scraped data + indexes
│   ├── shl_catalog.json
│   ├── shl_index.faiss
│   ├── assessments.pkl
│   ├── bm25_index.pkl
│   └── texts.pkl
├── requirements.txt
├── .env.example
├── .gitignore
├── Procfile                      # Render / Heroku deployment
├── render.yaml                   # Render deployment config
├── approach.md                   # Detailed solution approach
├── SUBMISSION_APPROACH.md        # 2-page submission document (export to PDF)
├── SUBMISSION_CHECKLIST.md       # Step-by-step submission checklist (all 5 items)
├── scripts/
│   └── export_approach_to_html.py  # Export approach doc to HTML for Print to PDF
└── README.md
```

---

## Optimization Techniques

| Technique | Impact |
|---|---|
| **Hybrid Search (Dense + BM25)** | Dense catches semantic meaning; BM25 catches exact keywords like "Java", "SQL" |
| **Reciprocal Rank Fusion** | Parameter-free merging of ranked lists from both search methods |
| **Query Expansion** | "Java developer" adds "Spring Boot backend programming" for better recall |
| **LLM Re-ranking (Gemini)** | Selects most relevant and balanced set from top candidates |
| **Balance Enforcement** | Ensures K+P type mix when query spans technical and soft skills |
| **URL Normalization** | Handles inconsistent URL formats across training labels |
| **Constraint Extraction** | Parses duration limits ("40 minutes") and remote requirements from query |

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Search:** FAISS (dense vectors), BM25 (sparse keywords), Sentence-Transformers (all-mpnet-base-v2)
- **LLM:** Google Gemini (e.g. gemini-2.0-flash; fallbacks on 429)
- **Scraping:** requests + BeautifulSoup (SHL catalog)
- **Frontend:** Vanilla HTML/CSS/JS (single-page, no build step)
- **Evaluation:** Custom Recall@K, Precision@K, NDCG@K implementation
