"""Query the sp500_companies ChromaDB collection from the terminal.

Usage:
    python -m scripts.query_chroma "Waymo self-driving cars"
    python -m scripts.query_chroma "the iphone maker" --n 5

The collection stores pre-computed embeddings (no embedding function attached),
so we must embed the query with the SAME model used to build it (all-MiniLM-L6-v2)
and pass query_embeddings.
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DB_PATH


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.query_chroma "<query text>" [--n N]')
        return

    query = sys.argv[1]
    n = 5
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db.get_collection("sp500_companies")
    print(f"Collection has {collection.count()} records.\n")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode([query]).tolist()

    result = collection.query(query_embeddings=query_embedding, n_results=n)

    ids = result["ids"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]
    print(f'Top {n} matches for: "{query}"\n')
    for rank, (cid, dist, meta) in enumerate(zip(ids, distances, metadatas), 1):
        name = meta.get("Company name") or meta.get("name") or "?"
        print(f"  {rank}. {cid:18} dist={dist:.3f}  {name}")


if __name__ == "__main__":
    main()
