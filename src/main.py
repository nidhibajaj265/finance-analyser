import asyncio
from src.data_handler import fetch_articles
from src.entities import process_unprocessed_articles
from src.events import update_article_with_event_info

asyncio.run(fetch_articles())
process_unprocessed_articles()
update_article_with_event_info()


