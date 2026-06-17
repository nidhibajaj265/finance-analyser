import json
from loguru import logger
from src.llm import call_llm

MAX_CHARS_FOR_TEXT = 1500

VALID_CATEGORIES = {"earnings", "M&A", "regulatory_action", "management_change", "product_launch", "litigation", "macro_shock", "other"}
VALID_SEVERITIES = {"low", "medium", "high"}
VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}

FALLBACK_EVENT = {"category": "other", "severity": "low", "direction": "neutral", "rationale": "Could not classify article."}

SYSTEM_PROMPT = """You are a financial event classifier. Classify the news into exactly one category, a severity, and a direction for the named company or market.
The "category" value MUST be exactly one of these 8 values: earnings, M&A, regulatory_action, management_change, product_launch, litigation, macro_shock, other.
Do not use any other category name.
Severity rules: M&A and regulatory_action are usually high; litigation medium-high;
earnings high if a clear beat/miss, else medium; routine product_launch low.
macro_shock is for macro shocks that can move entire markets: wars, geopolitical crises, pandemics, major central bank decisions, natural disasters — almost always high severity.
Direction rules:
- "bullish": event is likely positive for the company's stock (earnings beat, acquisition premium, FDA approval, favourable ruling).
- "bearish": event is likely negative for the company's stock (earnings miss, regulatory fine, lawsuit loss, CEO scandal).
- "neutral": outcome is genuinely unclear or mixed (pending investigation, routine product reveal, management change with no obvious signal).
Return ONLY valid JSON: {"category": "...", "severity": "...", "direction": "...", "rationale": "..."}.

Examples:
News: "Acme Corp to acquire Beta Inc for $4B in all-cash deal."
{"category": "M&A", "severity": "high", "direction": "bullish", "rationale": "Large all-cash acquisition signals growth confidence."}

News: "GadgetCo unveils new mid-range phone at annual event."
{"category": "product_launch", "severity": "low", "direction": "neutral", "rationale": "Routine product reveal with no clear financial impact."}

News: "Acme Corp beats Q3 earnings estimates, raises full-year guidance."
{"category": "earnings", "severity": "high", "direction": "bullish", "rationale": "Clear earnings beat with raised guidance."}

News: "Regulators fine BankCo $500M for compliance failures."
{"category": "regulatory_action", "severity": "high", "direction": "bearish", "rationale": "Major penalty directly hurts earnings and reputation."}

News: "BankCo CEO steps down, board names interim successor."
{"category": "management_change", "severity": "medium", "direction": "neutral", "rationale": "Leadership change with uncertain outcome."}

News: "Shareholders file class-action lawsuit against TechCo over data breach."
{"category": "litigation", "severity": "medium", "direction": "bearish", "rationale": "Active lawsuit creates financial and reputational risk."}

News: "FDA approves PharmaCo's new cancer drug."
{"category": "regulatory_action", "severity": "high", "direction": "bullish", "rationale": "Approval unlocks a major revenue stream."}

News: "Russia launches full-scale invasion, global markets plunge."
{"category": "macro_shock", "severity": "high", "direction": "bearish", "rationale": "Geopolitical conflict triggering broad market sell-off."}

News: "Fed raises interest rates by 75bps, largest hike in 28 years."
{"category": "macro_shock", "severity": "high", "direction": "bearish", "rationale": "Aggressive rate hike tightens financial conditions."}

News: "TechCo to present at upcoming industry investor conference."
{"category": "other", "severity": "low", "direction": "neutral", "rationale": "Routine announcement with no direct financial impact."}
"""

USER_PROMPT = """Now classify:
News: "{news_text}"
"""

def classify_articles(articles: list[dict]) -> list[dict]:
    logger.info(f"Classifying events for {len(articles)} articles...")
    for article in articles:
        result = classify_event(article)
        article.update(result)
    logger.info("Event classification complete.")
    return articles


def classify_event(article: dict) -> dict:
    text_to_classify = article['title'] + " " + article['text'][:MAX_CHARS_FOR_TEXT]
    result = call_llm(USER_PROMPT.format(news_text=text_to_classify), SYSTEM_PROMPT)

    try:
        parsed_result = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"LLM returned invalid JSON for [{article.get('title','')[:50]}], using fallback.")
        return FALLBACK_EVENT

    if (parsed_result.get('category') not in VALID_CATEGORIES or
        parsed_result.get('severity') not in VALID_SEVERITIES or
        parsed_result.get('direction') not in VALID_DIRECTIONS):
        logger.warning(f"Invalid fields from LLM for [{article.get('title','')[:50]}], using fallback.")
        return FALLBACK_EVENT

    logger.debug(f"[{article.get('title','')[:50]}] -> {parsed_result.get('category')} / {parsed_result.get('severity')} / {parsed_result.get('direction')}")
    return {
            'category': parsed_result.get('category'),
            'severity': parsed_result.get('severity'),
            'direction': parsed_result.get('direction'),
            'rationale': parsed_result.get('rationale', '')
           }
