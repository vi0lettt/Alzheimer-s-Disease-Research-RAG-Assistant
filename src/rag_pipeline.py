import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List
from openai import OpenAI
import re

# ---------------------------
# --- Загрузка .env и API-ключа OpenRouter ---
# ---------------------------
load_dotenv()  # ищет .env в текущей папке
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Укажите OPENROUTER_API_KEY в окружении или в .env")

# ---------------------------
# --- Настройка DeepSeek/OpenRouter ---
# ---------------------------
DEFAULT_MODEL = "tngtech/deepseek-r1t2-chimera:free"

class OpenRouterChat:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = "https://openrouter.ai/api/v1"):
        self.model = model
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=base_url)

    def send(self, content: str, system_prompt: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content

def send_message(content: str, system_prompt: Optional[str] = None) -> str:
    return OpenRouterChat().send(content, system_prompt=system_prompt)


df = pd.read_csv("data/processed/text_chunks.csv")
texts = df["text"].tolist()

# Создаём embeddings локально через SentenceTransformer
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings for chunks...")
embeddings = embed_model.encode(texts, convert_to_numpy=True).astype(np.float32)

# FAISS индекс
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors")

# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query: str, k: int = 3):
    q_emb = embed_model.encode([query]).astype(np.float32)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        results.append({
            "text": df.iloc[idx]["text"],
            "title": df.iloc[idx]["title"],
            "source": df.iloc[idx]["source"],
            "source_url": df.iloc[idx]["source_url"]
        })
    return results


def generate_answer(query: str, k: int = 3):
    retrieved = retrieve(query, k)
    
    context = "\n\n".join([
        f"[{r['title']}]({r['source_url']}): {r['text']}"
        for r in retrieved
    ])


    system_prompt = (
        "You are a scientific assistant specialized in Alzheimer's research. "
        "When citing articles, use the title as a clickable link in Markdown."
    )

    prompt = f"""
Based on the following context from scientific articles, answer the question concisely and cite the sources.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = send_message(prompt, system_prompt=system_prompt)
    return answer, retrieved


# ---------------------------
# Метрики качества
# ---------------------------
def compute_relevance_metrics(retrieved: List[Dict], keyword: str = "alzheimer"):
    k = len(retrieved)
    relevant = [1 if keyword.lower() in r["text"].lower() else 0 for r in retrieved]
    precision_at_k = sum(relevant) / k if k > 0 else 0
    recall_at_k = sum(relevant) / k  # простая оценка, если все релевантные chunks известны
    return {"precision@k": precision_at_k, "recall@k": recall_at_k}


def extract_query_terms(query: str):
    query = query.lower()
    query = re.sub(r"[^a-z\s]", "", query)
    stopwords = {
        "what", "are", "the", "for", "of", "and", "to", "is", "in", "with"
    }
    return [w for w in query.split() if w not in stopwords and len(w) > 3]


def query_term_precision_at_k(retrieved: List[Dict], query: str):
    terms = extract_query_terms(query)

    if not terms:
        return 0.0

    relevant = []
    for r in retrieved:
        text = r["text"].lower()
        relevant.append(
            1 if any(term in text for term in terms) else 0
        )

    return sum(relevant) / len(retrieved)


# ---------------------------
# Удобная функция запуска RAG пайплайна
# ---------------------------
def run_rag_pipeline(query: str, k: int = 3):
    answer, retrieved = generate_answer(query, k=k)

    base_metrics = compute_relevance_metrics(retrieved)
    query_precision = query_term_precision_at_k(retrieved, query)

    metrics = {
        **base_metrics,
        "query_term_precision@k": query_precision
    }

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "metrics": metrics
    }


# ---------------------------
# Пример использования
# ---------------------------
if __name__ == "__main__":
    query = "What are potential targets for Alzheimer's disease treatment?"
    result = run_rag_pipeline(query, k=3)

    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Metrics ---")
    for metric, value in result["metrics"].items():
        print(f"{metric}: {value:.2f}")

    print("\nВывод:")
    print("RAG-пайплайн работает, показывая retrieved chunks, сгенерированный ответ и метрики качества.")
