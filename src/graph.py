import json
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.signals import Signal
from src.config import SIGNALS_PATH
from src.data_handler import fetch_articles
from src.sentiment import generate_sentiment_report
from src.entities import match_entities_to_articles
from src.events import classify_articles
from src.signals import generate_signals
from loguru import logger

class GraphState(TypedDict):
    articles: list[dict]
    signals: list[Signal]

graph = StateGraph(GraphState)

async def ingest_node(state: GraphState) -> dict:
    articles = await fetch_articles()
    return {"articles": articles}

def after_ingest(state: GraphState) -> str:
    if not state['articles']:
        logger.warning("No articles fetched, skipping pipeline.")
        return END
    return "sentiment_analysis"

def sentiment_node(state: GraphState) -> dict:
    articles = generate_sentiment_report(state['articles'])
    return {"articles": articles}

def entity_recognition_node(state: GraphState) -> dict:
    articles = match_entities_to_articles(state['articles'])
    return {"articles": articles}

def article_event_classification_node(state: GraphState) -> dict:
    articles = classify_articles(state['articles'])
    return {"articles": articles}

def signal_generation_node(state: GraphState) -> dict:
    signals = generate_signals(state['articles'])
    return {"signals": signals}

def save_node(state: GraphState) -> dict:
    articles_by_ticker: dict[str, list[dict]] = {}
    for article in state["articles"]:
        for entity in article.get("entities", []):
            ticker = entity.get("ticker")
            if ticker:
                articles_by_ticker.setdefault(ticker, []).append({
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                })

    file_name = datetime.now().strftime('%Y%b%d_%H%M%S') + '.json'
    file_path = f"{SIGNALS_PATH}/{file_name}"

    with open(file_path, "w") as f:
        json.dump({
            "signals": [s.model_dump(mode="json") for s in state["signals"]],
            "articles_by_ticker": articles_by_ticker,
        }, f, indent=4)
    logger.info(f"Saved {len(state['signals'])} signals to {file_path}")
    return {}

graph.add_node("ingestion", ingest_node)
graph.add_node("sentiment_analysis", sentiment_node)
graph.add_node("entity_recognition", entity_recognition_node)
graph.add_node("article_event_classification", article_event_classification_node)
graph.add_node("signal_generation", signal_generation_node)
graph.add_node("save", save_node)

graph.set_entry_point("ingestion")
graph.add_conditional_edges("ingestion", after_ingest)
graph.add_edge("sentiment_analysis", "entity_recognition")
graph.add_edge("entity_recognition", "article_event_classification")
graph.add_edge("article_event_classification", "signal_generation")
graph.add_edge("signal_generation", "save")
graph.add_edge("save", END)

app = graph.compile()