import streamlit as st
from rag_pipeline import run_rag_pipeline

# ---------------------------
# --- Streamlit интерфейс ---
# ---------------------------
st.set_page_config(page_title="Alzheimer RAG Assistant", layout="wide")
st.title("RAG Assistant for Alzheimer's Research")

# Примеры вопросов
example_queries = [
    "What are potential targets for Alzheimer's disease treatment?",
    "Are the targets druggable with small molecules, biologics, or other modalities?",
    "What additional studies are needed to advance these targets?"
]

query = st.text_area(
    "Enter your research question:",
    value=example_queries[0],
    height=120
)

k = st.slider("Number of retrieved chunks:", min_value=1, max_value=10, value=3)

if st.button("Generate Answer"):
    with st.spinner("Retrieving and generating answer..."):
        result = run_rag_pipeline(query, k=k)

    st.subheader("Generated Answer")
    st.markdown(result["answer"])

    st.subheader("Metrics")
    for metric, value in result["metrics"].items():
        st.write(f"{metric}: {value:.2f}")
