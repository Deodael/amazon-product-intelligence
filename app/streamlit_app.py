"""
Streamlit UI for Amazon Product Intelligence.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.router import Router


st.set_page_config(
    page_title="Amazon Product Intelligence",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def get_router() -> Router:
    """Load the router once and cache it across reruns."""
    router = Router()
    # Ensure the RAG collection is populated (auto-index on first run)
    try:
        if router.rag.collection.count() == 0:
            with st.spinner(
                "Indexing reviews for semantic search (first run only, ~60 sec)..."
            ):
                router.rag.index_reviews(limit=500)
    except Exception as e:
        st.warning(f"RAG indexing skipped: {e}")
    return router


st.title("🛒 Amazon Product Intelligence")
st.markdown(
    """
    Ask questions about Amazon food reviews.

    This system uses a **router** to decide whether your question needs:
    - **SQL** — for counting, ranking, aggregations
    - **RAG** — for thematic/semantic questions (LLM-synthesized answers)
    - **Hybrid** — both combined
    """
)
st.divider()


with st.sidebar:
    st.header("💡 Example Questions")
    st.markdown(
        """
        **SQL route:**
        - How many 5-star reviews are there?
        - What is the average score?
        - What are the top products by review count?

        **RAG route:**
        - What do people complain about in coffee?
        - What do people say about packaging?
        - Reviews about stale products?
        """
    )
    st.divider()
    st.caption("Built with FastAPI, ChromaDB, Groq, and sentence-transformers.")


question = st.text_input(
    "Your question:",
    placeholder="e.g., What do people complain about in coffee?",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

if ask_button and question.strip():
    router = get_router()

    with st.spinner("Thinking..."):
        try:
            # Classify first to determine if synthesis is needed
            route_result = router.classify(question)

            if route_result.route.value == "rag":
                # RAG route: use LLM synthesis for a written answer
                synth = router.rag.synthesize(question, n_results=5)
                result = {
                    "route": "rag",
                    "reasoning": route_result.reasoning,
                    "question": question,
                    "answer": synth["answer"],
                    "sources": synth["sources"],
                }
            else:
                # SQL and hybrid use existing logic
                result = router.answer(question)
        except Exception as e:
            err = str(e).lower()
            if "index" in err or "chroma" in err or "collection" in err:
                st.warning(
                    "⏳ RAG search is warming up. Wait ~60 seconds and try again."
                )
            else:
                st.error(f"❌ Query failed: {e}")
            st.stop()

    route = result["route"]
    route_colors = {"sql": "🔵", "rag": "🟢", "hybrid": "🟣"}
    route_label = {"sql": "SQL", "rag": "RAG", "hybrid": "HYBRID"}

    st.success(
        f"{route_colors.get(route, '⚪')} Routed to **{route_label.get(route, route.upper())}** engine"
    )
    st.caption(f"*Reasoning: {result['reasoning']}*")

    st.subheader("Answer")
    answer = result["answer"]

    if isinstance(answer, str):
        # LLM-synthesized answer (RAG route) — display as prose
        st.markdown(answer)

        # Show the source reviews the LLM used
        if result.get("sources"):
            with st.expander(f"📚 Show {len(result['sources'])} source reviews"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**Review {i}** (score {src['score']}/5):")
                    st.write(src["content"][:500])
                    st.divider()

    elif isinstance(answer, list) and answer and isinstance(answer[0], dict):
        first = answer[0]
        if "content" in first:
            # RAG results rendered as cards (fallback path)
            for i, item in enumerate(answer, 1):
                with st.expander(
                    f"**Match {i}** — distance {item['distance']}",
                    expanded=(i == 1),
                ):
                    st.write(item["content"][:1000])
                    st.caption(
                        f"Product: `{item['product_id']}` · Score: {item['score']}★"
                    )
        else:
            # SQL results rendered as a table
            import pandas as pd
            st.dataframe(pd.DataFrame(answer))
    else:
        st.write(answer)

else:
    if ask_button:
        st.warning("Please enter a question.")
