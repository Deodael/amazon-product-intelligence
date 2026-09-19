"""
SQL Engine for Amazon Product Intelligence.

Provides a clean interface for querying the Amazon Fine Food Reviews
SQLite database, returning results as Pandas DataFrames.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DEFAULT_DB_PATH = Path.home() / "datasets" / "amazon_reviews" / "database.sqlite"

class SQLEngine:
    """Wrapper around the Amazon reviews SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.engine: Engine = create_engine(f"sqlite:///{self.db_path}")

    def run_query(self, sql: str) -> pd.DataFrame:
        """Execute any SQL query and return a Pandas DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn)

    def total_reviews(self) -> int:
        df = self.run_query("SELECT COUNT(*) AS n FROM Reviews")
        return int(df["n"].iloc[0])

    def average_score(self) -> float:
        df = self.run_query("SELECT AVG(Score) AS avg_score FROM Reviews")
        return float(df["avg_score"].iloc[0])

    def score_distribution(self) -> pd.DataFrame:
        return self.run_query("""
            SELECT Score, COUNT(*) AS count
            FROM Reviews
            GROUP BY Score
            ORDER BY Score
        """)

    def top_products(self, n: int = 10) -> pd.DataFrame:
        return self.run_query(f"""
            SELECT ProductId, COUNT(*) AS review_count
            FROM Reviews
            GROUP BY ProductId
            ORDER BY review_count DESC
            LIMIT {n}
        """)

    def top_rated_products(self, min_reviews: int = 50, n: int = 10) -> pd.DataFrame:
        return self.run_query(f"""
            SELECT ProductId, AVG(Score) AS avg_score, COUNT(*) AS n
            FROM Reviews
            GROUP BY ProductId
            HAVING n >= {min_reviews}
            ORDER BY avg_score DESC
            LIMIT {n}
        """)

    def reviews_for_product(self, product_id: str, limit: int = 20) -> pd.DataFrame:
        safe_id = product_id.replace("'", "''")
        return self.run_query(f"""
            SELECT Id, Score, Summary, Text, Time
            FROM Reviews
            WHERE ProductId = '{safe_id}'
            ORDER BY Time DESC
            LIMIT {limit}
        """)


if __name__ == "__main__":
    engine = SQLEngine()
    print(f"Total reviews: {engine.total_reviews():,}")
    print(f"Average score: {engine.average_score():.2f}")
    print("\nScore distribution:")
    print(engine.score_distribution().to_string(index=False))
    print("\nTop 5 products by review count:")
    print(engine.top_products(n=5).to_string(index=False))