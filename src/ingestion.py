import feedparser
from newspaper import Article
import asyncio

async def fetch_articles() -> list[dict]:
    articles_list = []
    feeds = []
    yahoo_finance_rss = 'https://finance.yahoo.com/news/rssindex'
    try:
        # For HTTP get requests like RSS feeds You must use "async with" for the session and request 
    #     # blocks, and you must use await on the final variable (like response.text() or response.json()) 
    #     # to actually read the server's response.
    #     async with feedparser.parse(yahoo_finance_rss) as yahoo_feed:
    #         await feeds.append(yahoo_feed)

    # except Exception as e:
    #     print(f"Error fetching Yahoo Finance RSS: {e}")


    # market_watch_rss = 'http://feeds.marketwatch.com/marketwatch/topstories/'
    # try:
    #     async with feedparser.parse(market_watch_rss) as market_watch_feed:
    #         await feeds.append(market_watch_feed)

    # except Exception as e:
    #     print(f"Error fetching Market Watch RSS: {e}")
    
    yahoo_feed, reuters_feed = await asyncio.gather(
    fetch_feed(yahoo_url),
    fetch_feed(reuters_url)
)


    for feed in feeds:
       for article in feed.entries:
            article_dict = {
                #using .get() to avoid KeyError if the key is missing
                'title': article.get('title', ''), 
                'link': article.get('link', ''),
                'published': article.get('published', ''),
                'summary': article.get('summary', ''),
                'text': extract_article_content(article.get('link', ''))
            }
            articles_list.append(article_dict)
    print(f"Fetched {articles_list[0]}")
    return articles_list

def extract_article_content(url :str) -> str:
    if not url:
        return ""
    
    content = Article(url)
    try:
        content.download()
        content.parse()
        return content.text

    except Exception as e:
        print(f"Error fetching article content from {url}: {e}")
        return ""
    
async def fetch_feed(url: str) -> None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, feedparser.parse, url)
    
if __name__ == "__main__":
    fetch_articles()
