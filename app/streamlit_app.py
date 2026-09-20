"""
Streamlit UI for Amazon Product Intelligence.

Run:
    streamlit run app/streamlit_app.py

Then open:
    http://localhost:8501
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.router import Router


# ---------- Page config ----------
st.set_page_config(
    page_title="Amazon Product Intelligence",
    page_icon="🛒",
    layout="wide",
)


# ---------- Load the router once ----------
@st.cache_resource
def get_router() -> Router:
    """Load the router once and cache it across reruns."""
    return Router()


# ---------- Header ----------
st.title("🛒 Amazon Product Intelligence")
st.markdown(
    """
    Ask questions about **568,454 Amazon food reviews**.
    
    This system uses a **router** to decide whether your question needs:
    - **SQL** — for counting, ranking, aggregations
    - **RAG** — for thematic/semantic questions about review content
    - **Hybrid** — both combined
    """
)
st.divider()


# ---------- Sidebar ----------
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
    st.caption("Built with FastAPI, ChromaDB, and sentence-transformers.")


# ---------- Main interface ----------
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
            result = router.answer(question)
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            st.stop()

    # ---------- Route badge ----------
    route = result["route"]
    route_colors = {"sql": "🔵", "rag": "🟢", "hybrid": "🟣"}
    route_label = {"sql": "SQL", "rag": "RAG", "hybrid": "HYBRID"}

    st.success(
        f"{route_colors.get(route, '⚪')} Routed to **{route_label.get(route, route.upper())}** engine"
    )
    st.caption(f"*Reasoning: {result['reasoning']}*")

    # ---------- Answer ----------
    st.subheader("Answer")

    answer = result["answer"]

    # SQL route: answer is a string or list of dicts
    if isinstance(answer, str):
        st.markdown(f"### {answer}")

    elif isinstance(answer, list) and answer and isinstance(answer[0], dict):
        # Could be SQL results or RAG results
        first = answer[0]

        if "content" in first:
            # RAG results — show as cards
            for i, item in enumerate(answer, 1):
                with st.expander(f"**Match {i}** — distance {item['distance']}", expanded=(i == 1)):
                    st.write(item["content"][:1000])
                    st.caption(f"Product: `{item['product_id']}` · Score: {item['score']}★")
        else:
            # SQL results — show as table
            import pandas as pd
            st.dataframe(pd.DataFrame(answer))

    else:
        st.write(answer)

else:
    if ask_button:
        st.warning("Please enter a question.")