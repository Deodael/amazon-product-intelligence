```markdown
# Amazon Product Intelligence

> A hybrid SQL + RAG system for product review intelligence — FastAPI, LangChain, ChromaDB, and semantic search over 568,454 Amazon reviews.

## 🎯 What This Project Does

This system answers questions about the Amazon Fine Food Reviews dataset using **two complementary engines**:

| Question Type | Engine | Example |
|--------------|--------|---------|
| **Counting / ranking** | SQL | *"How many 5-star reviews are there?"* |
| **Thematic / semantic** | RAG | *"What do people complain about in coffee?"* |
| **Both** | Hybrid | *"Complaints about products with 500+ reviews?"* |

A **router** classifies each question and directs it to the right engine. The system is exposed via a FastAPI endpoint and a Streamlit UI.

---

## 🏗️ Architecture

```

```
User Question
## 🏗️ Architecture

```
                    User Question
                          │
                          ▼
                    ┌──────────┐
                    │  Router  │
                    └────┬─────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌────────┐      ┌────────┐      ┌────────┐
   │  SQL   │      │  RAG   │      │ HYBRID │
   └───┬────┘      └───┬────┘      └───┬────┘
       │               │               │
       ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │ SQLite  │    │ ChromaDB │    │ SQL+RAG  │
   └─────────┘    └──────────┘    └──────────┘
```


**Stack:**
- **Data:** SQLite database (`database.sqlite`, 372 MB, 568,454 rows)
- **SQL layer:** `sqlalchemy` + `pandas`
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim)
- **Vector store:** ChromaDB (persistent, cosine similarity)
- **Router:** custom rule-based classifier
- **API:** FastAPI *(coming)*
- **UI:** Streamlit *(coming)*

---

## 📊 Dataset Insights

Discovered during exploratory analysis:

| Metric | Value |
|--------|-------|
| Total reviews | **568,454** |
| Users | 256,059 |
| Products | 74,258 |
| Date range | Oct 1999 – Oct 2012 |
| Average score | **4.18 / 5** |

**Score distribution (key insight):**

| Score | Count | % |
|-------|-------|---|
| 5 ⭐ | 363,122 | **63.9%** |
| 4 ⭐ | 80,655 | 14.2% |
| 3 ⭐ | 42,640 | 7.5% |
| 2 ⭐ | 29,769 | 5.2% |
| 1 ⭐ | 52,268 | 9.2% |

**Observation:** The dataset exhibits **severe class imbalance** — 64% of reviews are 5-star, while 1–2 star reviews combined make up only 14.4%. This shapes both the modeling approach and which questions are worth asking.

---

## 🔍 Known Limitations & Failure Modes

Real systems break. Here's where this one does — and why documenting it matters.

### 1. RAG conflates sentiment with topic

**Observation:** Querying *"What do people complain about in coffee?"* returns reviews about coffee — but some are **positive**.

Example results:
```

[0.387] "Disgusting. This coffee is absolutely horrendous..."       ← correct
[0.479] "My aunt's favorite. I used to use this coffee after..."    ← NOT a complaint
[0.479] "Very Good Coffee. I love ordering whole bean coffee..."    ← opposite sentiment

```

**Root cause:** The embedding model (`all-MiniLM-L6-v2`) captures **topic similarity** (coffee) more strongly than **sentiment** (complaints). "Coffee complaints" and "coffee praise" land close together in embedding space.

**Fix (planned):** The **hybrid route** should pre-filter with SQL:
```sql
SELECT * FROM Reviews WHERE Score <= 2
```

Then run RAG semantic search on that subset only. This ensures retrieved reviews are already negative before semantic ranking is applied.

2. ChromaDB telemetry warning

```
Failed to send telemetry event: capture() takes 1 positional argument but 3 were given
```

Cause: A version mismatch between chromadb 0.5.0 and posthog (its telemetry client). Harmless — search works normally.

Workaround: Set ANONYMIZED_TELEMETRY=False in .env.

3. Router is rule-based, not semantic

The router uses regex keyword matching, not an LLM. This is fast and free, but brittle — a question like "What's the vibe on this coffee?" won't match any keyword and falls back to RAG by default.

Fix (planned): Optional LLM-based classification for ambiguous questions.

---

🚀 How to Run Locally

Prerequisites

· Python 3.12
· The database.sqlite file from Amazon Fine Food Reviews
· ~600 MB disk for the dataset + ~400 MB for embeddings

Setup

```bash
git clone https://github.com/Deodael/amazon-product-intelligence.git
cd amazon-product-intelligence

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the dataset

Download database.sqlite from Kaggle and place it at:

```
~/datasets/amazon_reviews/database.sqlite
```

(Or set a custom path when instantiating SQLEngine.)

Index reviews (first run only, ~2 minutes)

```bash
python -m src.rag_engine
```

This embeds 2,000 reviews and stores them in chroma_db/. Subsequent runs skip indexing.

Test the SQL layer

```bash
python -m src.sql_engine
```

Test the RAG layer

```bash
python -m src.rag_engine
```

Test the router

```bash
python -m src.router
```

---

📂 Project Structure

```
amazon-product-intelligence/
├── src/
│   ├── __init__.py
│   ├── sql_engine.py       # SQL query layer (Pandas + SQLAlchemy)
│   ├── rag_engine.py       # Semantic search (ChromaDB + sentence-transformers)
│   ├── router.py           # SQL/RAG/Hybrid classifier
│   ├── api.py              # FastAPI endpoints (planned)
│   └── ...
├── sql/
│   └── queries.sql         # Curated SQL query library
├── notebooks/              # Exploration notebooks
├── app/                    # Streamlit UI (planned)
├── data/                   # Local data (gitignored)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

🗺️ Roadmap

Layer Status
SQL Engine ✅ Complete
RAG Engine ✅ Complete
Router ✅ Complete
LLM Synthesis ⏳ Planned
FastAPI Service ⏳ Planned
Streamlit UI ⏳ Planned
Hybrid pre-filtering ⏳ Planned

---

📜 License

MIT — see LICENSE.

---

🙏 Acknowledgments

· Dataset: Amazon Fine Food Reviews (Stanford Network Analysis Project, CC0)
· Embeddings: sentence-transformers/all-MiniLM-L6-v2
· Vector store: ChromaDB