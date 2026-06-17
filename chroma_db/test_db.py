import textwrap

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DB_PATH, PROCESSED_ARTICLES_PATH
from src.data_handler import load_from_json

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection("sp500_companies")
print(collection.count())
#df = pd.DataFrame(collection.peek()['metadatas'])

result = collection.get(ids=["sp500_GOOGL"], include=["documents"])

doc = result["documents"][0]
for field in doc.split(" | "):
    # wrap long fields (like Description) to 100 chars, indenting continuation lines
    print(textwrap.fill(field, width=100, subsequent_indent="    "))
    print()

# --- Experiment: does semantic search surface the right company for an
# article, even when its canonical name doesn't appear (or never appears)
# in the text? Compare a few article types. ---
# articles = load_from_json("2026Jun13_065911.json", PROCESSED_ARTICLES_PATH)
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# titles_to_test = [
#     "Does Google Stock Have More Upside?",            # "Google" only, never "Alphabet" in title
#     "Alphabet Deepens AI Bets With Waymo Testing Site And Anthropic Backing",  # already matched via alias
#     "SpaceX shares soar 19% in stock market debut",   # not an S&P 500 company at all
# ]

# for title in titles_to_test:
#     article = next(a for a in articles if a["title"] == title)
#     query_embedding = embedder.encode(article["text"]).tolist()
#     results = collection.query(query_embeddings=[query_embedding], n_results=5)

#     print(f"\n=== {title} ===")
#     for ticker, name, distance in zip(
#         [m["id"] for m in results["metadatas"][0]],
#         [m["Company name"] for m in results["metadatas"][0]],
#         results["distances"][0],
#     ):
#         print(f"{ticker:6} {name:30} distance={distance:.4f}")