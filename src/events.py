import json
import os
from loguru import logger
from src.config import PROCESSED_ARTICLES_PATH
from src.data_handler import load_from_json, save_to_json
from src.llm import call_llm

MAX_CHARS_FOR_TEXT = 1500

VALID_CATEGORIES = {"earnings", "M&A", "regulatory_action", "management_change", "product_launch", "litigation", "macro_shock", "other"}
VALID_SEVERITIES = {"low", "medium", "high"}

FALLBACK_EVENT = {"category": "other", "severity": "low", "rationale": "Could not classify article."}

SYSTEM_PROMPT = """You are a financial event classifier. Classify the news into exactly one category and a severity.
The "category" value MUST be exactly one of these 8 values: earnings, M&A, regulatory_action, management_change, product_launch, litigation, macro_shock, other.
Do not use any other category name.
Severity rules: M&A and regulatory_action are usually high; litigation medium-high;
earnings high if a clear beat/miss, else medium; routine product_launch low.
macro_shock is for macro shocks that can move entire markets: wars, geopolitical crises, pandemics, major central bank decisions, natural disasters — almost always high severity.
Return ONLY valid JSON: {"category": "...", "severity": "...", "rationale": "..."}.

Examples:
News: "Acme Corp to acquire Beta Inc for $4B in all-cash deal."
{"category": "M&A", "severity": "high", "rationale": "Large all-cash acquisition."}

News: "GadgetCo unveils new mid-range phone at annual event."
{"category": "product_launch", "severity": "low", "rationale": "Routine product reveal."}

News: "Acme Corp beats Q3 earnings estimates, raises full-year guidance."
{"category": "earnings", "severity": "high", "rationale": "Clear earnings beat with raised guidance."}

News: "Regulators fine BankCo $500M for compliance failures."
{"category": "regulatory_action", "severity": "high", "rationale": "Major regulatory penalty against the company."}

News: "BankCo CEO steps down, board names interim successor."
{"category": "management_change", "severity": "medium", "rationale": "Departure of a top executive."}

News: "Shareholders file class-action lawsuit against TechCo over data breach."
{"category": "litigation", "severity": "medium", "rationale": "Active lawsuit, outcome uncertain."}

News: "Russia launches full-scale invasion, global markets plunge."
{"category": "macro_shock", "severity": "high", "rationale": "Geopolitical conflict triggering broad market sell-off."}

News: "Fed raises interest rates by 75bps, largest hike in 28 years."
{"category": "macro_shock", "severity": "high", "rationale": "Major central bank policy shift with systemic market impact."}

News: "TechCo to present at upcoming industry investor conference."
{"category": "other", "severity": "low", "rationale": "Routine announcement with no direct financial impact."}
"""

USER_PROMPT = """Now classify:
News: "{news_text}"
"""

def update_article_with_event_info()-> list[dict]:
    logger.info("Classifying events for processed articles...")
    for file_name in os.listdir(PROCESSED_ARTICLES_PATH):
        file_path = os.path.join(PROCESSED_ARTICLES_PATH, file_name)

        if not os.path.isfile(file_path) or not file_name.endswith('.json'):
            continue

        articles = load_from_json(file_name, PROCESSED_ARTICLES_PATH)
        logger.info(f"Classifying {len(articles)} articles in {file_name}...")
        for article in articles:
            result = classify_event(article)
            article['category'] = result.get('category')
            article['severity'] = result.get('severity')
            article['rationale'] = result.get('rationale')

        save_to_json(articles, PROCESSED_ARTICLES_PATH, file_name)
    logger.info("Event classification complete.")

def classify_event(article: dict) -> dict:
    text_to_classify = article['title'] + " " + article['text'][:MAX_CHARS_FOR_TEXT]
    result = call_llm(USER_PROMPT.format(news_text=text_to_classify), SYSTEM_PROMPT)

    try:
        parsed_result = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"LLM returned invalid JSON for [{article.get('title','')[:50]}], using fallback.")
        return FALLBACK_EVENT

    if (parsed_result.get('category') not in VALID_CATEGORIES or
        parsed_result.get('severity') not in VALID_SEVERITIES):
        logger.warning(f"Invalid category/severity from LLM for [{article.get('title','')[:50]}], using fallback.")
        return FALLBACK_EVENT

    logger.debug(f"[{article.get('title','')[:50]}] -> {parsed_result.get('category')} / {parsed_result.get('severity')}")
    return {
            'category': parsed_result.get('category'),
            'severity': parsed_result.get('severity'),
            'rationale': parsed_result.get('rationale', '')
           }

if __name__ == '__main__':
    update_article_with_event_info()
