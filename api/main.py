"""
SHL Recommender API
===================
FastAPI backend with required endpoints:
  GET  /health      -> {"status": "healthy"}
  POST /recommend   -> {"recommended_assessments": [...]}

Also serves the frontend at the root URL.

Run locally:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pydantic Models (exact field names per spec)
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


class AssessmentResult(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResult]


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Do not load engine at startup — keeps server light so / and /health always work.
    # Engine loads on first /recommend (may take 1–2 min; Render free tier has 512MB RAM).
    log.info("Starting up (engine loads on first /recommend request)...")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Recommends SHL assessments from natural language queries or job descriptions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Middleware: Request logging + timing
# ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    log.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({duration:.2f}s)"
    )
    return response


# ─────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────

FRONTEND_FILE = PROJECT_ROOT / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if FRONTEND_FILE.exists():
        return FileResponse(str(FRONTEND_FILE), media_type="text/html")
    return JSONResponse(
        {"message": "SHL Assessment Recommender API", "docs": "/docs"},
    )


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health_check():
    return {"status": "healthy"}


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get assessment recommendations",
)
async def recommend(request: QueryRequest):
    """
    Given a natural language query or job description text/URL,
    return 5-10 relevant SHL Individual Test Solutions.
    """
    from recommender.engine import get_recommendations, get_recommendations_from_url

    query = request.query.strip()

    if query.startswith("http://") or query.startswith("https://"):
        log.info(f"Input is URL, fetching content: {query}")
        results = get_recommendations_from_url(query, top_k=10)
    else:
        results = get_recommendations(query, top_k=10)

    if not results:
        raise HTTPException(
            status_code=500,
            detail="Could not generate recommendations. Check engine logs."
        )

    cleaned = []
    for r in results:
        cleaned.append(AssessmentResult(
            url=r.get("url", ""),
            name=r.get("name", ""),
            adaptive_support=r.get("adaptive_support", "No"),
            description=(r.get("description", "") or "")[:500],
            duration=int(r.get("duration", 0) or 0),
            remote_support=r.get("remote_support", "No"),
            test_type=r.get("test_type", []),
        ))

    return RecommendResponse(recommended_assessments=cleaned)


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ─────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        log_level="info",
    )
