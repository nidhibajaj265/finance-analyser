from datetime import datetime
from datasketch import MinHash, MinHashLSH
import feedparser
import hashlib
from newspaper import Article
import asyncio
import json
import os
from src.config import ARTICLES_CACHE_PATH
from loguru import logger

async def fetch_articles() -> list[dict]:
    articles_list = []
    feeds = []
    yahoo_finance_rss = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US'
    investopedia_rss = 'https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline'

    #not working right now
    #market_watch_rss = 'http://feeds.marketwatch.com/marketwatch/topstories/'
    try:
       
        feeds = await asyncio.gather(
        fetch_feed(yahoo_finance_rss),
        fetch_feed(investopedia_rss))
    except Exception as e:
        logger.error(f"Error fetching feeds: {e}")
        return articles_list

    for feed in feeds:
       for article in feed.entries:
            article_dict = {
                #using .get() to avoid KeyError if the key is missing       
                'title': article.get('title', ''), 
                'link': article.get('link', ''),
                'published': article.get('published', ''),
                'summary': article.get('summary', ''),
                'id': hashlib.md5(article.get('link', '').encode("utf-8")).hexdigest()[:12]
            }
            articles_list.append(article_dict)

    # extract artcle content will not be called here as its an async function 
    # and without await just a coroutine will be created
    # gathering all the coroutines created for each article to run them concurrently
    coroutines_for_asyncio_gather = [extract_article_content(article) for article in articles_list]
    await asyncio.gather(*coroutines_for_asyncio_gather)
    logger.info(f"Fetched {len(articles_list)} articles. Extracting content...")

    articles_list = remove_duplicates(articles_list)
    logger.info(f"{len(articles_list)} articles after deduplication")
    return articles_list

async def extract_article_content(article: dict) -> None:
    logger.info(f"Extracting content for article {article['title']}")
    url = article.get('link', '')
    if not url:
        article['text'] = ""
        return

    loop = asyncio.get_event_loop()
    try:
        content = await loop.run_in_executor(None, Article, url)
        await asyncio.get_event_loop().run_in_executor(None, content.download)
        await asyncio.get_event_loop().run_in_executor(None, content.parse)
        article['text'] = content.text
    except Exception as e:
        logger.warning(f"Could not extract content from {url}: {e}")
        article['text'] = ""
       
async def fetch_feed(url: str) -> feedparser.FeedParserDict:
    
    # May need to use to be usable with market watch but 
    # dont like the idea so skipping for now
    # # Browser headers to bypass basic anti-bot blocks
    # headers = {
    # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # "Accept-Language": "en-US,en;q=0.9",
    # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    # }
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, feedparser.parse, url)

def _minhash(text:str) -> MinHash:
    m = MinHash(num_perm=128)
    for token in set(text.lower().split()):
        m.update(token.encode("utf8"))
    return m

def remove_duplicates(articles_list: list[dict]) -> list[dict]:

    minLSH = MinHashLSH(threshold=0.8, num_perm=128)
    unique_articles_list = []
    for i, article in enumerate(articles_list):
        if article.get('text', '') != '':
            text = article.get('text', '')
        else:
            article['text'] = article.get('summary', '') + " " + article.get('title', '')
            text = article.get('text', '')

        mh = _minhash(text)
        if not minLSH.query(mh):
            minLSH.insert(f"m_{i}", mh)
            unique_articles_list.append(article)

    save_to_json(unique_articles_list)

    return unique_articles_list

def save_to_json(articles_list: list[dict], folder_path: str = ARTICLES_CACHE_PATH, file_name: str = None):
    if file_name is None:
        file_name = datetime.now().strftime('%Y%b%d_%H%M%S') + '.json'
    file_path = os.path.join(folder_path, file_name)
    logger.info(f"Saving {len(articles_list)} articles to {file_path}")
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(articles_list, file, indent=4)

def load_from_json(file_name:str, folder_path: str = ARTICLES_CACHE_PATH) -> list[dict]:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        article_dict = json.load(file)

    return article_dict

if __name__ == "__main__":
    articles_fetched = asyncio.run(fetch_articles())
    save_to_json(articles_fetched)
