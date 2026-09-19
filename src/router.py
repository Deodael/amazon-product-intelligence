"""
Router for Amazon Product Intelligence.

Classifies a natural-language question into one of three routes:
    - SQL:    aggregation / counting / ranking questions
    - RAG:    semantic / thematic questions about review content
    - HYBRID: questions that need SQL filtering + RAG search

This is a rule-based classifier (no LLM needed) — fast, deterministic, free.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from src.sql_engine import SQLEngine
from src.rag_engine import RAGEngine


class Route(str, Enum):
    SQL = "sql"
    RAG = "rag"
    HYBRID = "hybrid"


@dataclass
class RoutingResult:
    route: Route
    question: str
    reasoning: str


# Keyword patterns for each route
SQL_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\baverage\b",
    r"\bavg\b",
    r"\btotal\b",
    r"\bsum\b",
    r"\btop \d+\b",
    r"\bmost\b",
    r"\bleast\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bdistribution\b",
    r"\bnumber of\b",
    r"\bratio\b",
    r"\bpercent",
]

RAG_PATTERNS = [
    r"\bwhat do people (say|think|complain|mention|feel)\b",
    r"\bwhat are (the )?(complaints|issues|praises|reviews)\b",
    r"\bcomplain",
    r"\bpraise\b",
    r"\bmention",
    r"\bsimilar to\b",
    r"\babout\b",
    r"\bwhy do\b",
    r"\bwhat makes\b",
    r"\bopinions? on\b",
]

HYBRID_TRIGGERS = [
    r"\babout\b",
    r"\bmention",
    r"\bsay about\b",
    r"\bcomplain",
]


class Router:
    """Routes questions to SQL, RAG, or HYBRID."""

    def __init__(
        self,
        sql_engine: Optional[SQLEngine] = None,
        rag_engine: Optional[RAGEngine] = None,
    ):
        self.sql = sql_engine or SQLEngine()
        # Lazy-load RAG to avoid paying the model load cost if not needed
        self._rag = rag_engine

    @property
    def rag(self) -> RAGEngine:
        if self._rag is None:
            self._rag = RAGEngine()
        return self._rag

    # ---------- Classification ----------

    def classify(self, question: str) -> RoutingResult:
        """Classify a question into SQL / RAG / HYBRID."""
        q = question.lower().strip()

        has_sql = any(re.search(p, q) for p in SQL_PATTERNS)
        has_rag = any(re.search(p, q) for p in RAG_PATTERNS)
        has_hybrid_trigger = any(re.search(p, q) for p in HYBRID_TRIGGERS)

        if has_sql and has_hybrid_trigger:
            return RoutingResult(
                route=Route.HYBRID,
                question=question,
                reasoning="Has aggregation keywords AND semantic intent.",
            )
        if has_sql:
            return RoutingResult(
                route=Route.SQL,
                question=question,
                reasoning="Contains aggregation keywords.",
            )
        if has_rag:
            return RoutingResult(
                route=Route.RAG,
                question=question,
                reasoning="Contains semantic / thematic keywords.",
            )
        # Default: try RAG (most questions are about content)
        return RoutingResult(
            route=Route.RAG,
            question=question,
            reasoning="No SQL keywords detected — defaulting to RAG.",
        )

    # ---------- Execution ----------

    def answer(self, question: str) -> dict:
        """Classify + execute. Returns a dict with route, result, and metadata."""
        result = self.classify(question)

        if result.route == Route.SQL:
            return self._execute_sql(result)
        elif result.route == Route.RAG:
            return self._execute_rag(result)
        else:
            return self._execute_hybrid(result)

    def _execute_sql(self, result: RoutingResult) -> dict:
        """Execute SQL. Currently supports a small set of canned queries."""
        q = result.question.lower()

        if "5-star" in q or "5 star" in q or "five star" in q:
            df = self.sql.run_query(
                "SELECT COUNT(*) AS n FROM Reviews WHERE Score = 5"
            )
            answer = f"{int(df['n'].iloc[0]):,} five-star reviews."
        elif "distribution" in q or "score" in q:
            df = self.sql.score_distribution()
            answer = df.to_dict(orient="records")
        elif "average" in q or "avg" in q:
            avg = self.sql.average_score()
            answer = f"Average score: {avg:.2f}"
        elif "top" in q and "product" in q:
            df = self.sql.top_products(n=10)
            answer = df.to_dict(orient="records")
        else:
            # Fallback: total count
            n = self.sql.total_reviews()
            answer = f"{n:,} total reviews in the dataset."

        return {
            "route": result.route.value,
            "reasoning": result.reasoning,
            "question": result.question,
            "answer": answer,
        }

    def _execute_rag(self, result: RoutingResult) -> dict:
        """Execute RAG search."""
        df = self.rag.search_reviews(result.question, n_results=5)
        return {
            "route": result.route.value,
            "reasoning": result.reasoning,
            "question": result.question,
            "answer": df.to_dict(orient="records"),
        }

    def _execute_hybrid(self, result: RoutingResult) -> dict:
        """Execute hybrid: SQL filter + RAG search."""
        # Placeholder: for now, run RAG but note it's hybrid.
        # Real implementation would parse filters and pass to RAG metadata filter.
        df = self.rag.search_reviews(result.question, n_results=5)
        return {
            "route": result.route.value,
            "reasoning": result.reasoning,
            "question": result.question,
            "answer": df.to_dict(orient="records"),
            "note": "Hybrid route uses RAG for now; SQL pre-filtering is a future enhancement.",
        }


if __name__ == "__main__":
    router = Router()

    test_questions = [
        "How many 5-star reviews are there?",
        "What is the average score?",
        "What do people complain about in coffee reviews?",
        "What do people say about the packaging?",
        "How many reviews total?",
    ]

    print("=" * 70)
    print("ROUTER CLASSIFICATION TESTS")
    print("=" * 70)

    for q in test_questions:
        result = router.classify(q)
        print(f"\n📝 {q}")
        print(f"   → Route: {result.route.value.upper()}")
        print(f"   → Reason: {result.reasoning}")