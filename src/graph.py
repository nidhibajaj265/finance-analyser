from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.signals import Signal
from src.data_handler import fetch_articles
from src.sentiment import generate_sentiment_report
from src.entities import match_entities_to_articles
from src.events import classify_articles
from src.signals import generate_signals
from src.supabase_client import save_signals, save_articles, consolidate_signals
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
    save_signals(signals)
    save_articles(state['articles'])
    consolidate_signals()
    return {"signals": signals}

graph.add_node("ingestion", ingest_node)
graph.add_node("sentiment_analysis", sentiment_node)
graph.add_node("entity_recognition", entity_recognition_node)
graph.add_node("article_event_classification", article_event_classification_node)
graph.add_node("signal_generation", signal_generation_node)

graph.set_entry_point("ingestion")
graph.add_conditional_edges("ingestion", after_ingest)
graph.add_edge("sentiment_analysis", "entity_recognition")
graph.add_edge("entity_recognition", "article_event_classification")
graph.add_edge("article_event_classification", "signal_generation")
graph.add_edge("signal_generation", END)

app = graph.compile()