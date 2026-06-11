import asyncio
from src.ingestion import fetch_articles
from src.sentiment import generate_sentiment_report

articles = asyncio.run(fetch_articles())
articles_with_statement = generate_sentiment_report(articles)

print(f"Articles with sentiment are: {articles_with_statement}")
for article in articles_with_statement[:5]:
    print(f"Summary: {article.get('summary','')}")
    print(f"Sentiment: {article.get('sentiment')}")
