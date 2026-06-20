import os
import streamlit as st


def _bridge_secrets() -> None:
    # On Streamlit Cloud, credentials live in st.secrets; src.config reads os.environ.
    # Copy them across before importing src modules (those imports build the Supabase client).
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in ("SUPABASE_URL", "SUPABASE_KEY", "HF_TOKEN"):
        try:
            if key in secrets and key not in os.environ:
                os.environ[key] = secrets[key]
        except Exception:
            pass


_bridge_secrets()

import pandas as pd
import yfinance as yf
from newspaper import Article
from src.supabase_client import fetch_consolidated_signals, fetch_signals_for_ticker
from src.llm import summarise_article_text

SIGNAL_COLOURS = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
HISTORY_DAYS = 7

st.set_page_config(page_title="Finance Analyser", layout="wide")

# Style the "Summarise" link: recolour it and pull it up close to the article summary.
st.markdown(
    """
    <style>
    button[kind="tertiary"] {
        color: #2563eb !important;
        margin-top: -14px !important;
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_page():
    st.title("Finance Analyser — Investment Signals")

    consolidated = fetch_consolidated_signals()
    if not consolidated:
        st.warning("No signals found. Run the pipeline first.")
        return

    st.subheader(f"{len(consolidated)} tickers — consolidated mood (last {HISTORY_DAYS} days)")

    header = st.columns([1, 2, 1, 1, 2])
    header[0].markdown("**Ticker**")
    header[1].markdown("**Company**")
    header[2].markdown("**Signal**")
    header[3].markdown("**Confidence**")
    header[4].markdown("**Updated**")
    st.divider()

    for row in consolidated:
        cols = st.columns([1, 2, 1, 1, 2])
        if cols[0].button(row["ticker"], key=row["ticker"]):
            st.session_state["selected_ticker"] = row["ticker"]
            st.switch_page(detail)
        cols[1].write(row["entity"])
        cols[2].write(f"{SIGNAL_COLOURS[row['signal']]} {row['signal']}")
        cols[3].write(f"{row['confidence']:.0%}")
        cols[4].write(row["updated_at"][:19].replace("T", " "))


@st.cache_data(ttl=3600)
def fetch_stock_prices(ticker: str, days: int = 7) -> pd.DataFrame:
    # hourly bars give a readable intraday movement line over the window.
    data = yf.Ticker(ticker).history(period=f"{days}d", interval="1h")
    return data[["Close"]]


@st.cache_data(ttl=86400, show_spinner=False)
def summarise_article(link: str) -> str:
    # full article text isn't stored on the signal, so fetch it on demand, then summarise.
    article = Article(link)
    article.download()
    article.parse()
    text = (article.text or "").strip()
    if not text:
        return "Could not retrieve the article text to summarise."
    return summarise_article_text(text)


def _consolidate_articles(history: list[dict]) -> list[dict]:
    # gather every article across the window, deduped by link (falling back to title).
    seen: dict[str, dict] = {}
    for signal in history:
        for item in signal.get("evidence", []) or []:
            if not isinstance(item, dict):
                continue
            key = item.get("link") or item.get("title")
            if key and key not in seen:
                seen[key] = item
    return list(seen.values())


def _consolidate_technical_info(history: list[dict]) -> list[str]:
    notes: list[str] = []
    for signal in history:
        # newer rows carry reasoning in technical_info; older backfilled rows
        # stored reasoning as plain strings inside evidence.
        candidates = signal.get("technical_info", []) or [
            e for e in (signal.get("evidence", []) or []) if isinstance(e, str)
        ]
        for note in candidates:
            if note not in notes:
                notes.append(note)
    return notes


def detail_page():
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        st.info("Pick a ticker from the Signals page.")
        return

    if st.button("← Back to signals"):
        st.switch_page(listing)

    history = fetch_signals_for_ticker(ticker, HISTORY_DAYS)
    entity = history[0]["entity"] if history else ticker
    st.title(f"{entity} ({ticker})")

    if not history:
        st.info("No individual signals in the selected window.")
        return

    st.subheader(f"Price — last {HISTORY_DAYS} days")
    prices = fetch_stock_prices(ticker, HISTORY_DAYS)
    if prices.empty:
        st.info("No price data available from Yahoo Finance for this ticker.")
    else:
        st.line_chart(prices, y="Close")

    articles = _consolidate_articles(history)
    if articles:
        st.subheader("Related articles")
        summarised = st.session_state.setdefault("summarised", set())
        for article in articles:
            title = article.get("title") or "Untitled"
            link = article.get("link")
            st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
            if article.get("summary"):
                st.write(article["summary"])
            if link:
                if st.button("Summarise", key=f"sum_{link}", type="tertiary"):
                    summarised.add(link)
                if link in summarised:
                    with st.spinner("Summarising…"):
                        st.caption(summarise_article(link))

    notes = _consolidate_technical_info(history)
    if notes:
        st.divider()
        corner = st.columns([1, 4])
        with corner[0].popover("⚙️ technical info"):
            for note in notes:
                st.markdown(f"- {note}")


listing = st.Page(list_page, title="Signals", url_path="signals", default=True)
detail = st.Page(detail_page, title="Ticker detail", url_path="ticker")
st.navigation([listing, detail], position="hidden").run()
