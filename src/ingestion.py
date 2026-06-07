import feedparser
from newspaper import Article

def fetch_articles() -> list[dict]:
    articles_list = []
    feeds = []
    yahoo_finance_rss = 'https://finance.yahoo.com/news/rssindex'
    try:
        yahoo_feed = feedparser.parse(yahoo_finance_rss) 
        feeds.append(yahoo_feed)
    except Exception as e:
        print(f"Error fetching Yahoo Finance RSS: {e}")


    market_watch_rss = 'http://feeds.marketwatch.com/marketwatch/topstories/'
    try:
        market_watch_feed = feedparser.parse(market_watch_rss)
        feeds.append(market_watch_feed)
    except Exception as e:
        print(f"Error fetching Market Watch RSS: {e}")

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



