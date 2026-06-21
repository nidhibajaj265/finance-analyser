import os
from collections import Counter
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
from src.supabase_client import (
    fetch_consolidated_signals,
    fetch_signals_for_ticker,
    fetch_latest_run_articles,
)
from src.llm import summarise_article_text

SIGNAL_COLOURS = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
SIGNAL_BG = {"bullish": "green", "bearish": "red", "neutral": "gray"}
HISTORY_DAYS = 7
SIGNALS_PER_PAGE = 10


def signal_badge(signal: str) -> str:
    return f":{SIGNAL_BG[signal]}-background[{SIGNAL_COLOURS[signal]} {signal}]"


def confidence_text(conf: float) -> str:
    colour = "green" if conf >= 0.5 else "orange" if conf >= 0.2 else "gray"
    return f":{colour}[**{conf:.0%}**]"


def article_tags(article: dict) -> str:
    # direction (coloured), event category, then the matched company tickers — shown on top of the article.
    parts = []
    direction = article.get("direction")
    if direction in SIGNAL_BG:
        parts.append(signal_badge(direction))
    category = article.get("category")
    if category:
        parts.append(f":violet-background[{category}]")
    for entity in article.get("entities", []) or []:
        ticker = entity.get("ticker")
        if ticker:
            parts.append(f":blue-background[{ticker}]")
    return " ".join(parts)

st.set_page_config(page_title="Finance Analyser", layout="wide")

# Style the "Summarise" link: recolour it and pull it up close to the article summary.
st.markdown(
    """
    <style>
    /* trim the default empty space above the page heading */
    .block-container { padding-top: 2rem !important; }
    /* page heading colour */
    .block-container h1 { color: #93c5fd !important; }
    button[kind="tertiary"] {
        color: #60a5fa !important;
        margin-top: -14px !important;
        padding-top: 0 !important;
    }
    /* ---- compact styling, scoped to the signals table only ---- */
    /* pull the table up close to the header row */
    .st-key-signalrows { margin-top: -0.5rem; }
    /* tighten vertical spacing so the ticker rows sit closer together */
    .st-key-signalrows div[data-testid="stVerticalBlock"] { gap: 0rem; }
    /* shrink the bordered row cards: minimal internal padding */
    .st-key-signalrows div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0rem 0.6rem !important; }
    /* alternating row colours (zebra striping) */
    .st-key-signalrows [class*="st-key-signalrow_even_"] { background-color: #1e293b !important; }
    .st-key-signalrows [class*="st-key-signalrow_odd_"]  { background-color: #0f172a !important; }
    /* smaller, tighter buttons so rows aren't tall */
    .st-key-signalrows div[data-testid="stButton"] button {
        padding: 0rem 0.4rem !important;
        min-height: 0 !important;
        line-height: 1.2 !important;
        font-size: 0.8rem !important;
    }
    /* remove default top/bottom margin on cell text and shrink it to match */
    .st-key-signalrows div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0 !important;
        font-size: 0.8rem !important;
        line-height: 1.2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_page():
    st.title("📈 Next-Gen Finance")
    st.caption("AI-generated investment signals")

    consolidated = fetch_consolidated_signals()
    if not consolidated:
        st.warning("No signals found. Run the pipeline first.")
        return

    # KPI summary row
    counts = Counter(row["signal"] for row in consolidated)
    latest = max((row["updated_at"] for row in consolidated), default="")
    kpis = st.columns(4)
    kpis[0].metric("Tickers tracked", len(consolidated), border=True)
    kpis[1].metric("🟢 Bullish", counts.get("bullish", 0), border=True)
    kpis[2].metric("🔴 Bearish", counts.get("bearish", 0), border=True)
    kpis[3].metric("⚪ Neutral", counts.get("neutral", 0), border=True)
    if latest:
        st.caption(f"Last updated {latest[:19].replace('T', ' ')} UTC")

    st.divider()
    st.subheader(f"Signals — consolidated mood (last {HISTORY_DAYS} days)")

    total_pages = max(1, (len(consolidated) + SIGNALS_PER_PAGE - 1) // SIGNALS_PER_PAGE)
    page = max(0, min(st.session_state.get("signals_page", 0), total_pages - 1))
    start = page * SIGNALS_PER_PAGE
    page_rows = consolidated[start:start + SIGNALS_PER_PAGE]

    header = st.columns([1, 2, 1, 1, 2], vertical_alignment="center")
    header[0].markdown("**Ticker**")
    header[1].markdown("**Company**")
    header[2].markdown("**Signal**")
    header[3].markdown("**Confidence**")
    header[4].markdown("**Updated**")

    with st.container(key="signalrows"):
        for i, row in enumerate(page_rows):
            parity = "even" if i % 2 == 0 else "odd"
            with st.container(border=True, key=f"signalrow_{parity}_{i}"):
                cols = st.columns([1, 2, 1, 1, 2], vertical_alignment="center")
                if cols[0].button(row["ticker"], key=row["ticker"]):
                    st.session_state["selected_ticker"] = row["ticker"]
                    st.session_state["selected_row"] = row
                    st.switch_page(detail)
                cols[1].write(row["entity"])
                cols[2].markdown(signal_badge(row["signal"]))
                cols[3].markdown(confidence_text(row["confidence"]))
                cols[4].write(row["updated_at"][:19].replace("T", " "))

    prev, info, nxt = st.columns([1, 2, 1])
    if prev.button("← Prev", disabled=page == 0, use_container_width=True):
        st.session_state["signals_page"] = page - 1
        st.rerun()
    info.markdown(f"<div style='text-align:center'>Page {page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    if nxt.button("Next →", disabled=page >= total_pages - 1, use_container_width=True):
        st.session_state["signals_page"] = page + 1
        st.rerun()

    st.divider()
    st.subheader("📰 Latest news")
    articles = fetch_latest_run_articles()
    if not articles:
        st.info("No articles from the latest run yet.")
        return

    for article in articles:
        st.markdown(article_tags(article))
        title = article.get("title") or "Untitled"
        link = article.get("link")
        st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
        if article.get("summary"):
            st.write(article["summary"])
        st.divider()


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

    row = st.session_state.get("selected_row", {})
    if row.get("ticker") == ticker:
        mood, conf = st.columns([1, 1])
        mood.markdown(f"#### {signal_badge(row['signal'])}")
        conf.metric("Consolidated confidence", f"{row['confidence']:.0%}")

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
