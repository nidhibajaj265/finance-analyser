from src.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
from src.signals import Signal, DIRECTION_VALUE, BULLISH_THRESHOLD, BEARISH_THRESHOLD
from loguru import logger
from datetime import datetime, timedelta, timezone

# Defining the half life for articles, since we are considering only 
# last seven days articles for consolidated confidence
HALF_LIFE_DAYS = 3 

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_signals(entity_signals: list[Signal]) -> None:
    if not entity_signals:
        logger.warning("No signals to save to Database")
        return
    
    rows = [s.model_dump(mode='json') for s in entity_signals]
    supabase.table('signals').insert(rows).execute()
    logger.info(f"Saved {len(rows)} to DB")

ARTICLE_COLUMNS = ["title", "summary", "link", "category", "severity", "direction", "sentiment", "entities"]

def save_articles(articles: list[dict]) -> None:
    if not articles:
        logger.warning("No articles to save to Database")
        return

    run_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {col: article.get(col) for col in ARTICLE_COLUMNS} | {"run_at": run_at}
        for article in articles
    ]
    supabase.table('articles').insert(rows).execute()
    logger.info(f"Saved {len(rows)} articles to DB")

def fetch_latest_run_articles() -> list[dict]:
    response = (
        supabase.table('articles')
        .select('*')
        .order('run_at', desc=True)
        .limit(200)
        .execute())

    rows = response.data
    if not rows:
        logger.info("No articles found in DB")
        return []

    latest_run = rows[0]['run_at']
    latest = [r for r in rows if r['run_at'] == latest_run]
    logger.info(f"Fetched {len(latest)} articles from the latest run")
    return latest

def fetch_recent_signals(days: int = 7) ->list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    response = (
        supabase.table('signals')
        .select('*')
        .gte('timestamp', cutoff.isoformat())
        .order('timestamp', desc=True)
        .execute())
    
    logger.info(f"Fetched {len(response.data)} signals from DB")
    return response.data

def fetch_consolidated_signals() -> list[dict]:
    response = (
        supabase.table('consolidated_signals')
        .select('*')
        .order('confidence', desc=True)
        .execute())

    logger.info(f"Fetched {len(response.data)} consolidated signals from DB")
    return response.data

def fetch_signals_for_ticker(ticker: str, days: int = 7) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    response = (
        supabase.table('signals')
        .select('*')
        .eq('ticker', ticker)
        .gte('timestamp', cutoff.isoformat())
        .order('timestamp', desc=True)
        .execute())

    logger.info(f"Fetched {len(response.data)} signals for {ticker} from DB")
    return response.data

def consolidate_signals(days: int = 7) -> None:
    recent_signals = fetch_recent_signals(days)

    if not recent_signals:
        logger.warning("No recent signals to consolidate")
        return
    
    signals_per_entity: dict[str,list[dict]] = {}
    for signal in recent_signals:
        signals_per_entity.setdefault(signal["ticker"],[]).append(signal)

    consolidated_signals: list[dict] = []

    now = datetime.now(timezone.utc)
    for ticker, ticker_signals in signals_per_entity.items():
        # TODO(revisit): confidence = avg conviction × consensus strength. Come back to this
        # logic — consider folding in coverage (number of articles/runs backing the signal).
        total_recency = 0.0                # Σ recency                    -> conviction denominator
        total_conf = 0.0                   # Σ (confidence × recency)     -> conviction numerator
        weighted_conf_with_direction = 0.0 # Σ (direction × confidence × recency)
        for s in ticker_signals:
            age_days = (now - datetime.fromisoformat(s["timestamp"])).total_seconds() / 86400
            recency = 0.5 ** (age_days/HALF_LIFE_DAYS)
            confidence_with_recency = recency * s["confidence"]
            total_recency += recency
            total_conf += confidence_with_recency
            weighted_conf_with_direction += DIRECTION_VALUE[s["signal"]] * confidence_with_recency

        # direction_score in [-1, 1]: which way the signals lean and how strongly they agree.
        direction_score = weighted_conf_with_direction / total_conf if total_conf else 0.0
        # avg_conviction in [0, 1]: how confident the signals were on average (recency-weighted).
        avg_conviction = total_conf / total_recency if total_recency else 0.0
        # final confidence rewards BOTH strong conviction AND agreement.
        consolidated_confidence = avg_conviction * abs(direction_score)

        if direction_score > BULLISH_THRESHOLD:
            label = "bullish"
        elif direction_score < BEARISH_THRESHOLD:
            label = "bearish"
        else:
            label = "neutral"

        consolidated_signals.append({
            "ticker": ticker,
            "entity": ticker_signals[0]["entity"],   # rows are newest-first, so [0] is latest
            "signal": label,
            "confidence": round(consolidated_confidence, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    supabase.table("consolidated_signals").upsert(consolidated_signals).execute()
    logger.info(f"Consolidated signals {len(consolidated_signals)} added to DB")
    

