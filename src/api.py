"""
FastAPI service for Amazon Product Intelligence.

Exposes the hybrid SQL + RAG system over HTTP.

Run:
    uvicorn src.api:app --reload --port 8000

Then visit:
    http://localhost:8000/docs       (interactive docs)
    http://localhost:8000/ask?q=...  (query endpoint)
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.router import Router


# ---------- Lifespan: load the router once at startup ----------
_router: Router | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once when the server starts."""
    global _router
    print("🚀 Starting up — loading router and models...")
    _router = Router()
    # Warm up the RAG model by doing a dummy search
    # (skip if the collection is empty)
    print("✅ Ready to serve requests.")
    yield
    print("👋 Shutting down.")


app = FastAPI(
    title="Amazon Product Intelligence",
    description="Hybrid SQL + RAG system for product review intelligence.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Response schemas ----------

class AskResponse(BaseModel):
    route: str
    reasoning: str
    question: str
    answer: Any


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------- Endpoints ----------

@app.get("/", tags=["meta"])
def root():
    """Root endpoint — describes the service."""
    return {
        "name": "Amazon Product Intelligence",
        "description": "Hybrid SQL + RAG system over 568,454 Amazon reviews.",
        "docs": "/docs",
        "ask_endpoint": "/ask?q=your+question+here",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness check — useful for deployment platforms."""
    return {"status": "ok", "version": app.version}


@app.get("/ask", response_model=AskResponse, tags=["query"])
def ask(
    q: str = Query(
        ...,
        min_length=3,
        max_length=500,
        description="Your natural-language question about the reviews.",
        examples=["How many 5-star reviews are there?"],
    ),
):
    """
    Ask a natural-language question. The router decides whether to use
    SQL, RAG, or hybrid, executes it, and returns the answer.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    try:
        result = _router.answer(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return AskResponse(**result)