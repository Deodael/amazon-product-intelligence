"""
RAG Engine for Amazon Product Intelligence.

Builds a semantic search layer over Amazon review text using
sentence-transformers embeddings and ChromaDB as the vector store.

On Streamlit Cloud, uses /tmp for the vector store since the repo
filesystem is read-only.
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.sql_engine import SQLEngine


# Disable ChromaDB telemetry (harmless but noisy)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "reviews"


def _pick_chroma_path() -> Path:
    """Choose a writable directory for the vector store."""
    candidates = [
        Path("/tmp/chroma_db"),                                    # Streamlit Cloud
        Path(__file__).resolve().parent.parent / "chroma_db",      # Local repo
        Path.cwd() / "chroma_db",                                  # Fallback
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Verify writable
            test_file = candidate / ".write_test"
            test_file.touch()
            test_file.unlink()
            return candidate
        except Exception:
            continue
    raise RuntimeError("Could not find a writable directory for ChromaDB.")


CHROMA_PATH = _pick_chroma_path()


def _is_cloud() -> bool:
    """Detect whether we're running on Streamlit Cloud."""
    return Path("/mount/src").exists()


class RAGEngine:
    """
    Semantic search engine over Amazon review text.

    Usage:
        engine = RAGEngine()
        engine.index_reviews(limit=500)
        results = engine.search_reviews("stale coffee", n_results=5)
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        chroma_path: Optional[str] = None,
    ):
        self.sql = SQLEngine(db_path)

        # Embedder
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ChromaDB persistent client
        path = Path(chroma_path) if chroma_path else CHROMA_PATH
        path.mkdir(parents=True, exist_ok=True)
        print(f"ChromaDB path: {path}")

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- Indexing ----------

    def index_reviews(self, limit: int = 500, batch_size: int = 128) -> int:
        """Pull reviews from SQLite, embed them, and store in ChromaDB."""
        existing = self.collection.count()
        if existing > 0:
            print(f"Collection already has {existing} reviews — skipping index.")
            return existing

        # Small limit on cloud to keep indexing fast
        if _is_cloud():
            limit = min(limit, 500)
            print(f"Cloud detected — capping index to {limit} reviews.")

        print(f"Fetching up to {limit} reviews from SQLite...")
        df = self.sql.run_query(f"""
            SELECT Id, ProductId, Score, Summary, Text
            FROM Reviews
            WHERE LENGTH(Text) > 50
            ORDER BY Id
            LIMIT {limit}
        """)
        print(f"Fetched {len(df)} reviews.")

        df["content"] = (
            df["Summary"].fillna("") + " " + df["Text"].fillna("")
        ).str.strip()

        print("Generating embeddings...")
        total = len(df)
        for start in range(0, total, batch_size):
            batch = df.iloc[start:start + batch_size]
            embeddings = self.embedder.encode(
                batch["content"].tolist(),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.collection.add(
                ids=[f"r{row.Id}" for row in batch.itertuples()],
                documents=batch["content"].tolist(),
                embeddings=embeddings.tolist(),
                metadatas=[
                    {
                        "product_id": str(row.ProductId),
                        "score": int(row.Score),
                        "review_id": int(row.Id),
                    }
                    for row in batch.itertuples()
                ],
            )
            print(f"  Indexed {min(start + batch_size, total)}/{total}")

        print(f"✅ Indexed {total} reviews.")
        return total

    # ---------- Search ----------

    def search_reviews(self, query: str, n_results: int = 5) -> pd.DataFrame:
        """Semantic search: find the most relevant reviews for a query."""
        if self.collection.count() == 0:
            raise RuntimeError(
                "No reviews indexed. Call index_reviews() first."
            )

        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        rows = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            rows.append(
                {
                    "content": doc,
                    "product_id": meta["product_id"],
                    "score": meta["score"],
                    "distance": round(dist, 4),
                }
            )

        return pd.DataFrame(rows)


if __name__ == "__main__":
    engine = RAGEngine()
    engine.index_reviews(limit=500)

    print("\n" + "=" * 60)
    print("TEST QUERIES")
    print("=" * 60)

    for query in [
        "stale coffee complaints",
        "packaging damaged on arrival",
        "my dog loves this treat",
    ]:
        print(f"\n🔍 Query: {query!r}")
        results = engine.search_reviews(query, n_results=3)
        for _, row in results.iterrows():
            snippet = row["content"][:120].replace("\n", " ")
            print(f"  [dist={row['distance']:.3f}] {snippet}...")
