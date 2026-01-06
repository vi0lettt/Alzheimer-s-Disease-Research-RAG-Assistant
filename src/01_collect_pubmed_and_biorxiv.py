from Bio import Entrez
import pandas as pd
import requests
from tqdm import tqdm



Entrez.email = "violetta_poleshchuck@mail.ru"

QUERIES = [
    "Alzheimer's disease targets",
    "Alzheimer therapeutic targets",
    "Alzheimer drug targets"
]

MAX_PAPERS_PER_QUERY = 10

def search_pubmed(query, max_results=10):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )
    results = Entrez.read(handle)
    return results["IdList"]

def fetch_pubmed_articles(id_list):
    articles = []

    for pmid in tqdm(id_list):
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)

        try:
            article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
            title = article.get("ArticleTitle", "")
            abstract = " ".join(article["Abstract"]["AbstractText"])
        except Exception:
            continue

        articles.append({
            "id": pmid,
            "title": title,
            "abstract": abstract,
            "introduction": "",
            "conclusion": "",
            "source": "PubMed",
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })

    return articles


def fetch_biorxiv(query, max_results=10):
    url = "https://api.biorxiv.org/details/biorxiv/2020-01-01/2024-12-31"
    response = requests.get(url).json()

    articles = []
    for item in response["collection"][:max_results]:
        doi = item.get("doi", "")
        articles.append({
            "id": doi,
            "title": item.get("title", ""),
            "abstract": item.get("abstract", ""),
            "introduction": "",
            "conclusion": "",
            "source": "bioRxiv",
            "source_url": f"https://www.biorxiv.org/content/{doi}"
        })

    return articles


all_articles = []

for q in QUERIES:
    pubmed_ids = search_pubmed(q, MAX_PAPERS_PER_QUERY)
    all_articles.extend(fetch_pubmed_articles(pubmed_ids))
    all_articles.extend(fetch_biorxiv(q, max_results=5))

df_raw = pd.DataFrame(all_articles)
df_raw.to_csv("data/raw/pubmed_biorxiv_raw.csv", index=False)

print(f"Collected articles: {len(df_raw)}")