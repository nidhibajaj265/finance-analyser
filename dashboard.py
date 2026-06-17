import json
import os
import streamlit as st
from src.config import SIGNALS_PATH

SIGNAL_COLOURS = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def load_latest_signals() -> dict:
    files = sorted(
        [f for f in os.listdir(SIGNALS_PATH) if f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return {}
    with open(os.path.join(SIGNALS_PATH, files[0])) as f:
        return json.load(f)


st.set_page_config(page_title="Finance Analyser", layout="wide")
st.title("Finance Analyser — Investment Signals")

data = load_latest_signals()

if not data:
    st.warning("No signals found. Run the pipeline first.")
    st.stop()

signals = data.get("signals", [])
articles_by_ticker = data.get("articles_by_ticker", {})

st.subheader(f"{len(signals)} signals generated")

cols = st.columns([1, 2, 1, 1, 2])
cols[0].markdown("**Ticker**")
cols[1].markdown("**Company**")
cols[2].markdown("**Signal**")
cols[3].markdown("**Confidence**")
cols[4].markdown("**Timestamp**")

st.divider()

selected_ticker = st.session_state.get("selected_ticker")

for signal in signals:
    cols = st.columns([1, 2, 1, 1, 2])
    if cols[0].button(signal["ticker"], key=signal["ticker"]):
        st.session_state["selected_ticker"] = signal["ticker"]
        selected_ticker = signal["ticker"]
    cols[1].write(signal["entity"])
    cols[2].write(f"{SIGNAL_COLOURS[signal['signal']]} {signal['signal']}")
    cols[3].write(f"{signal['confidence']:.0%}")
    cols[4].write(signal["timestamp"][:19].replace("T", " "))

if selected_ticker:
    st.divider()
    selected = next((s for s in signals if s["ticker"] == selected_ticker), None)
    if selected:
        st.subheader(f"{selected['entity']} ({selected_ticker})")

        st.markdown("**Evidence**")
        for point in selected["evidence"]:
            st.markdown(f"- {point}")

        articles = articles_by_ticker.get(selected_ticker, [])
        if articles:
            st.markdown(f"**Articles ({len(articles)})**")
            for article in articles:
                with st.expander(article["title"] or "Untitled"):
                    st.write(article["summary"] or "No summary available.")
