import chromadb
import os
from src.config import CHROMA_DB_PATH, ARTICLES_CACHE_PATH, PROCESSED_ARTICLES_PATH
from src.data_handler import load_from_json, save_to_json
from src.sentiment import generate_sentiment_report
from loguru import logger
import re

def normalize_company_names()-> dict:
    logger.info("Loading and normalizing company names from ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db.get_collection(name="sp500_companies")
    result = collection.get(include=["metadatas"])
    suffixes_to_remove = ("Inc","Corporation", "Corp",
                           "Co", "Ltd", "plc", "Class A", "Class B",
                           "Class C")
    company_info = {}

    #remove suffixes from company names eg Apple Inc -> Apple
    # write regex to exclude trailing brackets and words in then eg (Class A)
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

def match_entities_to_articles(company_info: dict, article_list: list[dict]) -> list[dict]:
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
                if confidence > 0.5:
                    matches.append({"ticker": ticker, "name": name, "confidence": confidence})

        article['entities'] = matches
        if matches:
            logger.debug(f"[{article.get('title','')[:50]}] matched {[m['ticker'] for m in matches]}")

    logger.info("Entity matching complete.")
    return article_list

def process_unprocessed_articles()-> None:
    company_info = normalize_company_names()
    for file_name in os.listdir(ARTICLES_CACHE_PATH):
        file_path = os.path.join(ARTICLES_CACHE_PATH, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith('.json'):
            continue

        article_list = load_from_json(file_name)
        processed_articles = match_entities_to_articles(company_info, article_list)
        processed_articles_with_sentiment = generate_sentiment_report(processed_articles)
        save_to_json(processed_articles_with_sentiment, folder_path=PROCESSED_ARTICLES_PATH, file_name=file_name)
        os.remove(file_path)
        logger.info(f"Processed {file_name}: {len(processed_articles_with_sentiment)} articles -> {PROCESSED_ARTICLES_PATH}")

if __name__ == "__main__":
    process_unprocessed_articles()
