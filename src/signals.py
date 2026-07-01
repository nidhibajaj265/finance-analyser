# Signal generation
# For each company (ticker) mentioned across today's articles, produce one bullish/bearish/neutral signal.
#
# Direction is owned entirely by the LLM event direction (which comes from the event type).
# FinBERT does NOT vote on direction — it only confirms or conflicts:
#   - LLM and FinBERT point the same way  -> keep the signal at full strength
#   - they conflict                       -> halve the confidence (CONFLICT_PENALTY)
#   - FinBERT is neutral                  -> no penalty
# So a bearish event stays bearish no matter what FinBERT says; FinBERT only moves confidence.
#
# Everything is weighted by severity and by recency (newer articles count more).
# Output: one Signal per ticker, ready to feed into the LangGraph pipeline.

from email.utils import parsedate_to_datetime
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
from loguru import logger

class Signal(BaseModel):
    entity: str
    ticker: str
    signal: Literal['bullish', 'bearish', 'neutral']
    confidence: float = Field(ge=0, le=1)
    evidence: list[dict] = Field(default_factory=list)
    timestamp: datetime
    technical_info: list[str] = Field(default_factory=list)

class ArticleInfoForSignal(BaseModel):
    direction: Literal["bullish", "bearish", "neutral"]      # LLM event direction — owns the signal direction
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)     # FinBERT signed sentiment — used for confirmation only
    severity_weight: float = Field(default=0.3, ge=0.0, le=1.0)  # low=0.3, medium=0.6, high=1.0
    recency: float = Field(default=1.0, ge=0.0, le=1.0)      # 1.0 = brand new, halves every HALF_LIFE_DAYS
    title: str | None = None
    summary: str | None = None
    link: str | None = None

DIRECTION_VALUE: dict[str, float] = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
SEVERITY_WEIGHT: dict[str, float] = {"low": 0.3, "medium": 0.6, "high": 1.0}

BULLISH_THRESHOLD = 0.2
BEARISH_THRESHOLD = -0.2
HALF_LIFE_DAYS = 2.0      # an article this old counts half as much as a brand-new one
COVERAGE_FULL = 5         # this many articles = full coverage; fewer damps confidence
CONFLICT_PENALTY = 0.5    # FinBERT pointing the opposite way to the LLM halves confidence


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def recency_weight(published: str | None, now: datetime) -> float:
    """1.0 for a brand-new article, halving every HALF_LIFE_DAYS. Missing/unparseable date -> 1.0 (no penalty)."""
    if not published:
        return 1.0
    try:
        pub = parsedate_to_datetime(published)
    except (TypeError, ValueError, IndexError):
        logger.warning(f"Could not parse published date {published!r}; using recency=1.0")
        return 1.0
    if pub is None:
        return 1.0
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    age_days = max((now - pub).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def _compute_signal(ticker: str, name: str, signals: list[ArticleInfoForSignal]) -> Signal:
    # Severity- and recency-weighted sums over the ticker's articles.
    total_weight = 0.0
    weighted_direction = 0.0   # Σ w·r·direction_value  -> drives the direction (LLM only)
    weighted_finbert = 0.0     # Σ w·r·sentiment        -> FinBERT confirmation check
    for s in signals:
        weight = s.severity_weight * s.recency
        weighted_direction += weight * DIRECTION_VALUE[s.direction]
        weighted_finbert += weight * s.sentiment_score
        total_weight += weight

    dir_score = weighted_direction / total_weight if total_weight else 0.0   # in [-1, 1], LLM-only
    avg_finbert = weighted_finbert / total_weight if total_weight else 0.0

    if dir_score > BULLISH_THRESHOLD:
        aggr_direction = "bullish"
    elif dir_score < BEARISH_THRESHOLD:
        aggr_direction = "bearish"
    else:
        aggr_direction = "neutral"

    # FinBERT confirmation: compare its sign to the LLM direction's sign.
    llm_sign = _sign(DIRECTION_VALUE[aggr_direction])
    finbert_sign = _sign(avg_finbert)
    conflict = llm_sign != 0 and finbert_sign != 0 and finbert_sign != llm_sign
    conflict_mult = CONFLICT_PENALTY if conflict else 1.0

    coverage = min(len(signals) / COVERAGE_FULL, 1.0)
    confidence = round(abs(dir_score) * coverage * conflict_mult, 3)

    direction_counts = {label: sum(1 for s in signals if s.direction == label) for label in DIRECTION_VALUE}
    article_breakdown = ", ".join(f"{count} {label}" for label, count in direction_counts.items() if count > 0)
    if aggr_direction == "neutral":
        finbert_verdict = "n/a (neutral signal)"
    elif conflict:
        finbert_verdict = f"conflicts with -> ×{CONFLICT_PENALTY} confidence"
    elif finbert_sign == 0:
        finbert_verdict = "neutral on (no penalty)"
    else:
        finbert_verdict = "confirms"

    technical_info = [
        f"{article_breakdown} across {len(signals)} article(s) -> {aggr_direction} "
        f"(direction score {dir_score:+.2f}, threshold ±{BULLISH_THRESHOLD})",
        f"FinBERT {finbert_verdict} the {aggr_direction} signal (avg sentiment {avg_finbert:+.2f})",
        f"confidence {confidence:.2f} = |dir| {abs(dir_score):.2f} × coverage {coverage:.2f} "
        f"× conflict {conflict_mult:.2f} ({len(signals)}/{COVERAGE_FULL} articles)",
    ]
    evidence = [
        {"type": "article", "title": s.title, "summary": s.summary, "link": s.link}
        for s in signals
        if s.title or s.summary or s.link
    ]

    logger.debug(
        f"[{ticker}] {aggr_direction} | dir={dir_score:+.2f} | finbert={avg_finbert:+.2f} "
        f"| conflict×{conflict_mult:.2f} | confidence={confidence:.2f} | articles={len(signals)}"
    )

    return Signal(
        entity=name,
        ticker=ticker,
        signal=aggr_direction,
        confidence=confidence,
        evidence=evidence,
        timestamp=datetime.now(timezone.utc),
        technical_info=technical_info,
    )


def generate_signals(articles: list[dict], now: datetime | None = None) -> list[Signal]:
    now = now or datetime.now(timezone.utc)
    logger.info(f"Grouping {len(articles)} articles by ticker...")

    signals_for_entity: dict[str, list[ArticleInfoForSignal]] = {}
    ticket_entity_dict: dict[str, str] = {}

    for article in articles:
        article_recency = recency_weight(article.get("published"), now)
        sentiment = max(-1.0, min(1.0, article.get("sentiment", 0.0)))  # guard against float drift out of [-1, 1]

        for entity in article.get("entities", []):
            ticker = entity.get("ticker")
            if not ticker:
                continue

            article_signal = ArticleInfoForSignal(
                direction=article.get("direction", "neutral"),
                sentiment_score=sentiment,
                severity_weight=SEVERITY_WEIGHT.get(article.get("severity", "low"), 0.3),
                recency=article_recency,
                title=article.get("title"),
                summary=article.get("summary"),
                link=article.get("link"),
            )
            signals_for_entity.setdefault(ticker, []).append(article_signal)
            ticket_entity_dict.setdefault(ticker, entity.get("name", ticker))

    logger.info(f"Grouped into {len(signals_for_entity)} tickers: {list(signals_for_entity.keys())}")

    results = [
        _compute_signal(ticker, ticket_entity_dict[ticker], ticker_signals)
        for ticker, ticker_signals in signals_for_entity.items()
    ]
    logger.info(f"Generated {len(results)} signals.")
    return results
