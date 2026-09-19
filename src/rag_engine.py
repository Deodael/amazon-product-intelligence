"""
RAG Engine for Amazon Product Intelligence.

Builds a semantic search layer over Amazon review text using
sentence-transformers embeddings and ChromaDB as the vector store.
"""

from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.sql_engine import SQLEngine


DEFAULT_DB_PATH = Path.home() / "datasets" / "amazon_reviews" / "database.sqlite"
CHROMA_PATH = Path("chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "reviews"


class RAGEngine:
    """
    Semantic search engine over Amazon review text.

    Usage:
        engine = RAGEngine()
        engine.index_reviews(limit=5000)
        results = engine.search_reviews("stale coffee", n_results=5)
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        chroma_path: Optional[str] = None,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.chroma_path = Path(chroma_path) if chroma_path else CHROMA_PATH

        # SQL layer for pulling reviews
        self.sql = SQLEngine(str(self.db_path))

        # Embedder: turns text into 384-dim vectors
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ChromaDB persistent client — data lives on disk between runs
        self.chroma_path.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- Indexing ----------

    def index_reviews(self, limit: int = 5000, batch_size: int = 256) -> int:
        """
        Pull reviews from the SQLite DB, embed them, and store in ChromaDB.

        Args:
            limit: maximum number of reviews to index (start small).
            batch_size: how many to embed at once.

        Returns:
            Number of reviews indexed.
        """
        # Skip if already populated
        existing = self.collection.count()
        if existing > 0:
            print(f"Collection already has {existing} reviews — skipping index.")
            return existing

        print(f"Fetching up to {limit} reviews from SQLite...")
        df = self.sql.run_query(f"""
            SELECT Id, ProductId, Score, Summary, Text
            FROM Reviews
            WHERE LENGTH(Text) > 50
            ORDER BY Id
            LIMIT {limit}
        """)
        print(f"Fetched {len(df)} reviews.")

        # Combine summary + text into one searchable string
        df["content"] = (
            df["Summary"].fillna("") + " " + df["Text"].fillna("")
        ).str.strip()

        # Embed in batches
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
        """
        Semantic search: find the most relevant reviews for a query.

        Returns a DataFrame with columns: content, product_id, score, distance.
        """
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
    engine.index_reviews(limit=2000)

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