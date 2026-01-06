import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

df = pd.read_csv("data/raw/pubmed_biorxiv_raw.csv")

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def merge_text(row):
    return " ".join([
        safe_text(row["abstract"]),
        safe_text(row["introduction"]),
        safe_text(row["conclusion"])
    ])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["full_text"] = df.apply(merge_text, axis=1)
df["clean_text"] = df["full_text"].apply(clean_text)
df["text_length"] = df["clean_text"].apply(len)

# опционально: оставить только Alzheimer
df = df[df["clean_text"].str.contains("alzheimer")]

import os
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/articles_clean.csv", index=False)

print("Number of articles:", len(df))
print("Average length:", df["text_length"].mean())

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15
)

X = vectorizer.fit_transform(df["clean_text"])
tfidf_df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nTop TF-IDF terms:")
print(tfidf_df.mean().sort_values(ascending=False))
