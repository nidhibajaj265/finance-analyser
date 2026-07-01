# Top gainers
# Once a day (via a scheduled GitHub Action), rank the biggest stock gainers over 7D / 1M / 1Y / 5Y
# across our company universe (S&P 500 + companies discovered from the news) and store them in
# Supabase for the dashboard to read. Kept out of the hourly pipeline because it's a heavy yfinance
# pull and only needs refreshing daily.

from datetime import datetime, timezone, timedelta

import chromadb
import pandas as pd
import yfinance as yf
from loguru import logger

from src.config import CHROMA_DB_PATH

UNIVERSE_COLLECTIONS = ("sp500_companies", "tracked_companies")
TOP_N = 10
BATCH_SIZE = 80   # tickers per yfinance bulk download

# label -> lookback window used for the % change
PERIODS: dict[str, timedelta] = {
    "7D": timedelta(days=7),
    "1M": timedelta(days=30),
    "1Y": timedelta(days=365),
    "5Y": timedelta(days=365 * 5),
}


def get_universe() -> dict[str, str]:
    """Return {ticker: company name} from the S&P 500 KB plus companies discovered from the news."""
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    universe: dict[str, str] = {}
    for name in UNIVERSE_COLLECTIONS:
        try:
            coll = db.get_collection(name)
        except Exception:
            logger.debug(f"Collection '{name}' not found; skipping.")
            continue
        for meta in coll.get(include=["metadatas"])["metadatas"]:
            ticker = meta.get("id")
            if ticker:
                universe.setdefault(ticker, meta.get("Company name", ticker))
    logger.info(f"Top-gainers universe: {len(universe)} tickers.")
    return universe


def _download_closes(tickers: list[str]) -> pd.DataFrame:
    """5y of daily close prices for all tickers as one DataFrame (columns = tickers), fetched in batches."""
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        data = yf.download(batch, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):        # single-ticker batch comes back as a Series
            data = data.to_frame(name=batch[0])
        frames.append(data)
        logger.info(f"Downloaded closes for {min(i + BATCH_SIZE, len(tickers))}/{len(tickers)} tickers.")
    closes = pd.concat(frames, axis=1)
    if closes.index.tz is not None:            # drop tz so date cutoffs compare cleanly
        closes.index = closes.index.tz_localize(None)
    return closes


def _pct_change(series: pd.Series, lookback: timedelta, now: datetime) -> tuple[float, float] | None:
    """Return (percent change over the lookback window, latest price), or None if there's not enough data."""
    s = series.dropna()
    if s.empty:
        return None
    cutoff = pd.Timestamp(now - lookback)
    prior = s[s.index <= cutoff]
    if prior.empty or prior.iloc[-1] == 0:
        return None
    latest = float(s.iloc[-1])
    return (latest / float(prior.iloc[-1]) - 1) * 100, latest


def compute_top_gainers(universe: dict[str, str], top_n: int = TOP_N, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    naive_now = now.replace(tzinfo=None)
    closes = _download_closes(list(universe))

    rows: list[dict] = []
    for label, lookback in PERIODS.items():
        gains = []
        for ticker in closes.columns:
            result = _pct_change(closes[ticker], lookback, naive_now)
            if result is not None:
                pct, price = result
                gains.append((ticker, pct, price))
        gains.sort(key=lambda g: g[1], reverse=True)
        for rank, (ticker, pct, price) in enumerate(gains[:top_n], start=1):
            rows.append({
                "period": label,
                "ticker": ticker,
                "company": universe.get(ticker, ticker),
                "pct_change": round(pct, 2),
                "price": round(price, 2),
                "rank": rank,
                "computed_at": now.isoformat(),
            })
        logger.info(f"[{label}] ranked {len(gains)} tickers, kept top {min(top_n, len(gains))}.")
    return rows


def update_top_gainers() -> None:
    universe = get_universe()
    if not universe:
        logger.warning("Empty universe; skipping top gainers.")
        return
    rows = compute_top_gainers(universe)
    from src.supabase_client import save_top_gainers
    save_top_gainers(rows)


if __name__ == "__main__":
    update_top_gainers()
