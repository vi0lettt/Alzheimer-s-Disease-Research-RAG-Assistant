import pandas as pd

df = pd.read_csv("data/processed/articles_clean.csv")

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks

all_chunks = []

for _, row in df.iterrows():
    chunks = chunk_text(row["clean_text"])
    for i, chunk in enumerate(chunks):
        all_chunks.append({
        "chunk_id": f"{row['id']}_{i}",
        "text": chunk,
        "title": row["title"],
        "source": row["source"],
        "source_url": row["source_url"],
        "article_id": row["id"]
    })


chunks_df = pd.DataFrame(all_chunks)
chunks_df.to_csv("data/processed/text_chunks.csv", index=False)

print("Total chunks:", len(chunks_df))
