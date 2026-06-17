import chromadb
import re
from src.config import CHROMA_DB_PATH
from loguru import logger


def _normalize_company_names() -> dict:
    logger.info("Loading and normalizing company names from ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db.get_collection(name="sp500_companies")
    result = collection.get(include=["metadatas"])
    suffixes_to_remove = ("Inc", "Corporation", "Corp",
                          "Co", "Ltd", "plc", "Class A", "Class B", "Class C")
    company_info = {}

    for entry in result['metadatas']:
        company_name = re.sub(r'\s*\([^)]*\)\s*$', '', entry['Company name'])
        company_name = company_name.strip().rstrip('.,&').strip()
        for suffix in suffixes_to_remove:
            if company_name.endswith(" " + suffix):
                company_name = company_name.removesuffix(suffix)
                company_name = company_name.strip().rstrip('.,&').strip()

        company_info[company_name] = entry['id']
    logger.info(f"Normalized {len(company_info)} company names.")
    return company_info


def match_entities_to_articles(article_list: list[dict]) -> list[dict]:
    company_info = _normalize_company_names()
    logger.info(f"Matching entities across {len(article_list)} articles...")
    for article in article_list:
        matches = []
        for name, ticker in company_info.items():
            pattern = r'\b' + re.escape(name) + r'\b'
            title_hits = re.findall(pattern, article['title'], re.IGNORECASE)
            text_hits = re.findall(pattern, article['text'], re.IGNORECASE)
            if title_hits or text_hits:
                confidence = 0
                if title_hits:
                    confidence = 0.5
                if text_hits:
                    confidence = confidence + min(0.1 * len(text_hits), 0.5)
                if confidence > 0.7:
                    matches.append({"ticker": ticker, "name": name, "confidence": confidence})

        article['entities'] = matches
        if matches:
            logger.debug(f"[{article.get('title', '')[:50]}] matched {[m['ticker'] for m in matches]}")

    logger.info("Entity matching complete.")
    return article_list
