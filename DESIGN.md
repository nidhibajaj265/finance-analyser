# Design Document — Autonomous Financial News Sentiment Analyzer & Investment Signal Generator

**Author:** (your name)
**Date:** 2026-06-06
**Status:** Draft v2.0
**Audience:** This document is written for a first-time builder of agentic systems. It explains both *what* to build and *why*, including foundational concepts. Read it top to bottom once before writing any code.

> **Tech-stack note (v2.0):** This version assumes you develop **locally in VS Code** (a normal Python project, not a Colab notebook) and that you run the machine-learning models **as hosted services** — either the **Hugging Face Inference API** (serverless, free tier) or a **paid LLM API** — rather than downloading and running the models on your own GPU. That removes the need for a GPU and for 4-bit quantization entirely. Wherever this matters, the document gives you **two paths** — "HF path" (free, more setup) and "Paid-API path" (costs money, simplest) — and a recommendation.

---

## Table of Contents

1. [What You Are Building (Plain English)](#1-what-you-are-building-plain-english)
2. [Foundational Concepts (Glossary)](#2-foundational-concepts-glossary)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Environment Setup](#4-environment-setup)
5. [Data Sources & How to Get Them](#5-data-sources--how-to-get-them)
6. [The RAG Knowledge Base](#6-the-rag-knowledge-base)
7. [Shared State: The Heart of LangGraph](#7-shared-state-the-heart-of-langgraph)
8. [The Agents, One by One](#8-the-agents-one-by-one)
9. [The LangGraph State Machine](#9-the-langgraph-state-machine)
10. [Models & Hosted Inference (HF API / Paid API)](#10-models--hosted-inference-hf-api--paid-api)
11. [The Signal Generation Algorithm](#11-the-signal-generation-algorithm)
12. [The Dashboard (Gradio)](#12-the-dashboard-gradio)
13. [Evaluation & Backtesting](#13-evaluation--backtesting)
14. [Phased Implementation Roadmap](#14-phased-implementation-roadmap)
15. [Risks, Pitfalls & Beginner Tips](#15-risks-pitfalls--beginner-tips)
16. [Mapping to Deliverables](#16-mapping-to-deliverables)
17. [Appendix A: Data Schemas](#appendix-a-data-schemas)
18. [Appendix B: Prompt Templates](#appendix-b-prompt-templates)

---

## 1. What You Are Building (Plain English)

You are building a small **automated financial analyst**. Every time it runs, it:

1. **Reads the financial news** from public internet feeds (like a person scanning Yahoo Finance headlines).
2. **Figures out which companies/sectors each article is about** (e.g., "this is about Apple, ticker AAPL").
3. **Decides whether the news sounds good or bad** for those companies (sentiment).
4. **Classifies what kind of event happened** (earnings report? a merger? a lawsuit?).
5. **Produces an investment signal** — a recommendation like *bullish / bearish / neutral* — with a **confidence score** and the **evidence** behind it.
6. **Writes a daily briefing** summarizing everything, and shows it on a **dashboard**.

The word **"agentic"** means the system is built as a set of cooperating **agents** (specialized components), and it can make **decisions about its own flow** — for example, routing earnings news to a special analyzer, or *pausing to ask a human* before emitting a high-impact signal. It is not a single straight-line script; it is a small graph of steps with branching logic.

> **Important framing for a beginner:** Do not try to build everything at once. This document describes the *final* system, but Section 14 gives you a **phased roadmap** so you build it in small, testable pieces. Build Phase 1, see it work, then move on.

---

## 2. Foundational Concepts (Glossary)

Read this section even if some terms feel familiar — the project uses them in specific ways.

| Term | What it means here |
|---|---|
| **Agent** | A focused software component with one job (e.g., "the Sentiment Agent"). In this project an agent is usually just a **Python function** that takes the shared state, does its work (possibly calling a model), and returns updates to the state. It does **not** have to be a fancy autonomous LLM loop. |
| **LangGraph** | A library for wiring agents together as a **graph** (nodes = steps, edges = transitions). It manages a **shared state** object that flows through the nodes and supports branching, loops, and pausing. |
| **LangChain** | A toolkit of helpers for working with language models (prompt templates, output parsers, model wrappers). LangGraph builds on top of it. You will use a thin slice of LangChain. |
| **Node** | One step in the LangGraph graph = one agent function. |
| **Edge** | A connection saying "after node A, go to node B." A **conditional edge** chooses the next node based on the current state (this is where the "decision-making" lives). |
| **State** | A single Python object (a typed dictionary) that holds *everything* the pipeline knows so far: the articles, extracted entities, sentiments, events, signals. Each node reads from it and writes to it. |
| **LLM (Large Language Model)** | A text-generation model, used here for event classification and report writing. You call it as a hosted service. Pick **one**: **Mistral-7B-Instruct** via the Hugging Face Inference API (free tier), **or** a paid API such as **Anthropic Claude** (most capable, simplest, costs money). See [Section 10](#10-models--hosted-inference-hf-api--paid-api). |
| **FinBERT** | A smaller, specialized model that classifies a sentence as *positive / negative / neutral* **for finance**. Not generative — it just labels text. Called via the **Hugging Face Inference API** here (`ProsusAI/finbert`). |
| **Embedding** | A list of numbers (a vector) that represents the *meaning* of a piece of text. Similar meanings → similar vectors. Used for search. |
| **Vector database / ChromaDB** | A database that stores embeddings and lets you find "the most similar stored text to my query." |
| **RAG (Retrieval-Augmented Generation)** | A pattern: before asking the model a question, you **retrieve** relevant reference text from the vector DB and paste it into the prompt, so the model answers using facts instead of guessing. Here it gives the model context about S&P 500 companies. |
| **NER (Named Entity Recognition)** | A model that finds names of things (companies, people) inside text. |
| **Hosted inference / Inference API** | Running a model **on someone else's servers** and calling it over the internet (an HTTP request), instead of downloading the model and running it on your own machine. You send text, you get back the model's output. No GPU needed on your side. Two flavours here: the **Hugging Face Inference API** and **paid LLM APIs** (e.g., Anthropic Claude, OpenAI). |
| **API key / token** | A secret password-like string that authenticates your requests to a hosted service. Kept out of your code, in a `.env` file. |
| **Rate limit** | A cap on how many requests (or tokens) a hosted service lets you send per minute/day. Hit it and you get a `429` error and must wait/retry. |
| **Quantization (4-bit NF4)** | *(No longer used in this design.)* A trick to shrink a model to fit in limited GPU memory. It only matters when you run the model **yourself** on a GPU. Because we now use **hosted inference**, the provider handles all of this — you never load or quantize a model. Kept in the glossary only so you recognize the term if you read the original brief. |
| **Human-in-the-loop (HITL)** | The pipeline **pauses** and waits for a person to approve/reject before continuing — used for high-impact signals. |
| **MinHash / deduplication** | A fast way to detect that two articles are near-duplicates so you don't process the same story twice. |
| **Pydantic** | A Python library to define structured data objects with validation. Your final signals are Pydantic objects, guaranteeing they always have the right fields. |
| **Backtesting** | Checking whether your signals would have matched real past price movements, using historical prices from `yfinance`. |

---

## 3. High-Level Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │            RAG Knowledge Base                │
                         │   (ChromaDB: S&P 500 profiles, sectors,      │
                         │    financial terminology — as embeddings)    │
                         └──────────────▲───────────────▲──────────────┘
                                        │ retrieve      │ retrieve
                                        │ context       │ context
   RSS / APIs                           │               │
   ┌────────┐    ┌──────────────┐  ┌────┴──────┐  ┌─────┴───────┐  ┌──────────────┐  ┌──────────────┐
   │ Yahoo  │    │ 1. News      │  │ 2. Entity │  │ 3. Sentiment│  │ 4. Event     │  │ 5. Signal    │
   │ Finance│───▶│ Ingestion    │─▶│ Recognition│─▶│ Analysis    │─▶│ Detection    │─▶│ Generation   │
   │ SEC    │    │ Agent        │  │ Agent     │  │ Agent       │  │ Agent        │  │ Agent        │
   │ Market │    │ (feedparser, │  │ (NER +    │  │ (FinBERT,   │  │ (LLM via HF  │  │ (sentiment + │
   │ Watch  │    │  newspaper3k,│  │  ChromaDB │  │  hosted)    │  │  or paid API)│  │  event +     │
   └────────┘    │  MinHash)    │  │  linking) │  │             │  │              │  │  price corr.)│
                 └──────────────┘  └───────────┘  └─────────────┘  └──────┬───────┘  └──────┬───────┘
                                                                          │                 │
                                                          severity=high?  │                 │
                                                       ┌──────────────────┘                 │
                                                       ▼                                     ▼
                                              ┌─────────────────┐                  ┌──────────────────┐
                                              │ Human-Review    │                  │ 6. Briefing      │
                                              │ Interrupt Node  │                  │ Report Generator │
                                              │ (pause → approve)│                 │ (Markdown/PDF)   │
                                              └─────────────────┘                  └────────┬─────────┘
                                                                                            │
                       event type = "earnings"? ──▶ Earnings Sub-Agent ──┐                  ▼
                                                                         └───────▶ ┌──────────────────┐
                                                                                   │ Gradio Dashboard │
                                                                                   │ (heatmap, time-  │
                                                                                   │  line, signals)  │
                                                                                   └──────────────────┘
```

**The flow in one sentence:** News comes in → we clean and dedupe it → identify companies → score sentiment → classify the event → combine everything into a signal (pausing for human review on high-impact ones) → write a briefing → display on a dashboard.

**Two pieces of "intelligence" to notice:**
- The **RAG knowledge base** sits to the side and is *queried* by the Entity and Sentiment agents — it is not a step in the line, it is a reference library.
- The **conditional edges** (human-review interrupt, earnings sub-agent) are what make this *agentic* rather than a plain pipeline.

---

## 4. Environment Setup

You are building a **normal local Python project in VS Code**. Because every ML model runs as a **hosted service** (you call it over the internet), you do **not** need a GPU, CUDA, PyTorch, or quantization on your machine. A plain laptop is enough.

### 4.1 Install Python + VS Code
1. Install **Python 3.11+** (check with `python3 --version`).
2. Install **VS Code**, then its **Python extension** (Microsoft). This gives you the run button, debugger, and the integrated terminal (`` Ctrl+` ``).
3. Optional but recommended: the **Jupyter** extension, so you can also run `.ipynb` notebooks inside VS Code if you want a notebook deliverable later.

### 4.2 Create the project and a virtual environment
A **virtual environment** (`venv`) is an isolated folder of packages just for this project, so it doesn't clash with other Python work on your computer. In the VS Code terminal, from your project folder:

```bash
python3 -m venv .venv            # create the isolated environment
source .venv/bin/activate        # activate it (macOS/Linux). Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```
When active, your terminal prompt shows `(.venv)`. In VS Code, also pick this interpreter: `Cmd/Ctrl+Shift+P → "Python: Select Interpreter" → .venv`.

### 4.3 Suggested project layout
```
finance-analyser/
├── .venv/                 # virtual environment (never commit this)
├── .env                   # secret API keys (never commit this)
├── .gitignore             # must list .venv and .env
├── requirements.txt       # pinned package list
├── DESIGN.md              # this document
├── data/                  # cached KB, sentiment trajectory, price history
└── src/
    ├── config.py          # loads keys from .env, model choices
    ├── ingestion.py       # News Ingestion Agent
    ├── entities.py        # Entity Recognition Agent
    ├── sentiment.py       # Sentiment Analysis Agent (FinBERT via HF API)
    ├── events.py          # Event Detection Agent (LLM via HF or paid API)
    ├── signals.py         # Signal Generation Agent
    ├── briefing.py        # Briefing Report Generator
    ├── kb.py              # builds/queries the ChromaDB knowledge base
    ├── graph.py           # the LangGraph wiring (nodes + edges)
    └── app.py             # the Gradio dashboard entry point
```
> One file per agent mirrors the architecture and makes each phase of the roadmap (Section 14) a self-contained file you can test alone.

### 4.4 Packages to install
Note what is **gone** versus the original brief: no `torch`, no `accelerate`, no `bitsandbytes` — those are only for running models locally on a GPU. Put this in `requirements.txt`:

```text
langchain
langgraph
langchain-community
huggingface_hub          # client for the Hugging Face Inference API
chromadb
sentence-transformers    # small embedding model — fine to run locally (CPU)
feedparser
newspaper3k
lxml_html_clean
yfinance
datasketch
gradio
pandas
matplotlib
pydantic
python-dotenv            # loads your .env file
datasets                 # to download Financial PhraseBank / FiQA for evaluation
# --- add ONE of these depending on your LLM choice (Section 10) ---
anthropic                # only if you use the paid Claude API
# openai                 # only if you use the paid OpenAI API
```
Install with `pip install -r requirements.txt`.

Notes for beginners:
- `newspaper3k` sometimes needs `lxml_html_clean` (added above) and `nltk` data: run once `python -c "import nltk; nltk.download('punkt')"`.
- `sentence-transformers` downloads one **small** embedding model (~80 MB) that runs comfortably on your CPU — this is the *one* model we run locally, because it's tiny and gets called a lot. Everything heavy (FinBERT, the LLM) stays hosted.
- You can later split into `requirements.txt` (runtime) and a `requirements-dev.txt` (e.g. `pytest`, `ruff`) — not required for v1.

### 4.5 Secrets / API keys — the `.env` file
**Never paste keys into your code.** Put them in a `.env` file (and add `.env` to `.gitignore` so it's never committed):

```text
# .env
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxx     # only if using the paid Claude path
SEC_USER_AGENT=finance-analyser your.email@example.com
```
Load them in `src/config.py`:
```python
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")   # may be None on the HF-only path
SEC_USER_AGENT = os.environ["SEC_USER_AGENT"]
```
Which keys you need:
- **RSS feeds** and **`yfinance`** — no key.
- **SEC EDGAR** — no key, but you must send a descriptive **User-Agent** with your email (their politeness policy).
- **Hugging Face Inference API** — create a free token at huggingface.co → Settings → Access Tokens. Some models (e.g. Mistral) require accepting their license on the model page first.
- **Paid LLM API** (only if you choose that path) — an API key from the provider's console (e.g. Anthropic). This one bills you per token.

---

## 5. Data Sources & How to Get Them

You have two categories: **reference data** (loaded once, used to build the knowledge base and to evaluate) and **live data** (fetched every run).

### 5.1 Live news (the input every run)
| Source | How | Notes |
|---|---|---|
| **Yahoo Finance RSS** | `feedparser.parse("https://finance.yahoo.com/news/rssindex")` or per-ticker feeds | Free, no key. Per-ticker: `https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US` |
| **MarketWatch RSS** | `feedparser.parse("http://feeds.marketwatch.com/marketwatch/topstories/")` | Free. |
| **SEC EDGAR** | EDGAR full-text search API / filings RSS, with your User-Agent header | Use for earnings/8-K filings. Heavier; add for the earnings sub-agent. |

`feedparser.parse(url)` returns an object with `.entries`, each having `.title`, `.link`, `.summary`, `.published`. You then fetch the **full article text** from `.link` using `newspaper3k`:
```python
from newspaper import Article
a = Article(url); a.download(); a.parse()
text = a.text
```

### 5.2 Reference / evaluation datasets
| Dataset | Role | How to load |
|---|---|---|
| **Financial PhraseBank** | Primary benchmark to measure your sentiment accuracy | `load_dataset("financial_phrasebank", "sentences_allagree")` |
| **FiQA Sentiment** | Secondary sentiment evaluation | `load_dataset("...")` (search HF hub; aspect-based) |
| **S&P 500 company list** | To build the knowledge base + entity linking | Scrape Wikipedia's S&P 500 table with `pandas.read_html`, or use a static CSV |
| **yfinance price history** | Backtesting signals vs. real prices | `yfinance.Ticker("AAPL").history(period="30d")` |
| **Synthetic news** | Few-shot examples / event-detection testing | Generate with Mistral or hand-write a handful |

> **Beginner tip:** Download and cache reference data to your local `data/` folder once. Re-downloading every run wastes time and may hit rate limits.

---

## 6. The RAG Knowledge Base

### 6.1 Why it exists
When an article says *"the iPhone maker beat estimates,"* a naive system doesn't know "iPhone maker" = Apple = AAPL = Technology sector. The knowledge base gives agents **background facts** to:
- **Link entities** ("iPhone maker" → AAPL) more accurately.
- **Interpret sentiment** with sector context.

### 6.2 What goes in it
Three kinds of documents, each embedded as a vector and stored in ChromaDB:
1. **Company profiles** — one per S&P 500 company: name, ticker, sector, a 1–2 sentence description, common aliases ("Google/Alphabet", "Facebook/Meta").
2. **Sector descriptions** — one per GICS sector (e.g., "Information Technology: companies in software, hardware...").
3. **Financial terminology** — short definitions ("EPS beat", "guidance", "downgrade", "share buyback") so the model interprets jargon.

### 6.3 How to build it (one-time setup step)
```python
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("all-MiniLM-L6-v2")   # small, fast, good enough
client = chromadb.Client()                            # in-memory; or PersistentClient(path=...)
collection = client.get_or_create_collection("sp500_kb")

docs   = [...]   # list of strings (profiles, sectors, terms)
ids    = [...]   # unique id per doc, e.g. "AAPL", "sector_tech", "term_eps"
metas  = [...]   # dict per doc, e.g. {"type":"company","ticker":"AAPL","sector":"Tech"}
embs   = embedder.encode(docs).tolist()

collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
```

### 6.4 How agents query it
```python
q_emb = embedder.encode(["iPhone maker beat estimates"]).tolist()
res = collection.query(query_embeddings=q_emb, n_results=3)
# res["documents"], res["metadatas"] → top matching company/sector context
```
The Entity agent uses the returned ticker/metadata to link; the Sentiment agent pastes the returned descriptions into context.

> **Concept check:** RAG = "look it up before you answer." The vector DB is your lookup index, keyed by *meaning* rather than exact words.

---

## 7. Shared State: The Heart of LangGraph

Before writing any agent, design the **state object** — the single dictionary that flows through the whole graph. Every agent reads some fields and writes others. Getting this right early saves enormous pain.

```python
from typing import TypedDict, List, Optional
from datetime import datetime

class Article(TypedDict):
    id: str
    title: str
    url: str
    text: str
    published: str
    source: str

class EntityMention(TypedDict):
    name: str
    ticker: Optional[str]
    sector: Optional[str]
    confidence: float

class FinanceState(TypedDict):
    # --- filled by Ingestion ---
    raw_articles: List[Article]
    deduped_articles: List[Article]
    # --- filled by Entity agent ---
    article_entities: dict          # article_id -> List[EntityMention]
    # --- filled by Sentiment agent ---
    entity_sentiments: dict         # (article_id, ticker) -> {label, score}
    sentiment_trajectory: dict      # ticker -> rolling 7-day series
    # --- filled by Event agent ---
    events: List[dict]              # {article_id, category, severity, rationale}
    # --- filled by Signal agent ---
    signals: List[dict]             # Pydantic Signal dumped to dict
    # --- control / HITL ---
    needs_human_review: bool
    human_decision: Optional[str]   # "approve" / "reject" / None
    # --- output ---
    briefing_markdown: Optional[str]
```

> **Why TypedDict?** LangGraph passes this object between nodes. Each node returns a *partial* dict of fields to update, and LangGraph merges it. Typing it keeps you (and the reader) clear on what exists when.

**Design rule:** A node should only *add* to state, never silently overwrite another agent's data. Keep fields append-only where possible.

---

## 8. The Agents, One by One

Each agent below is described with: **purpose → inputs → outputs → method → beginner notes**. Each agent is implemented as a function `def agent_x(state: FinanceState) -> dict:` that returns the fields it updates.

### 8.1 News Ingestion Agent
- **Purpose:** Collect fresh articles and remove duplicates.
- **Inputs:** A list of RSS feed URLs (config).
- **Outputs:** `raw_articles`, `deduped_articles`.
- **Method:**
  1. For each feed URL, `feedparser.parse(url)` → loop `.entries`.
  2. For each entry, fetch full text with `newspaper3k`. Wrap in `try/except` — some pages fail; skip gracefully.
  3. **Deduplicate with MinHash (datasketch):** compute a MinHash signature from each article's shingles (word n-grams), put them in an `MinHashLSH` index, and drop articles whose Jaccard similarity to an already-seen one exceeds a threshold (e.g., 0.8). This catches the same wire story republished by multiple outlets.
- **Beginner notes:**
  - Limit articles per run (e.g., 20–30) while developing, so iterations are fast and you don't hit GPU limits.
  - Cache fetched articles by URL to avoid re-downloading.
  - Respect feeds: add small delays; set a User-Agent.

```python
from datasketch import MinHash, MinHashLSH

def _minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for token in set(text.lower().split()):
        m.update(token.encode("utf8"))
    return m

def dedupe(articles, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    for i, art in enumerate(articles):
        mh = _minhash(art["text"])
        if not lsh.query(mh):           # no near-duplicate already indexed
            lsh.insert(f"a{i}", mh)
            kept.append(art)
    return kept
```

### 8.2 Entity Recognition Agent
- **Purpose:** Find which companies/people/sectors each article mentions and link them to canonical tickers.
- **Inputs:** `deduped_articles`, the ChromaDB knowledge base.
- **Outputs:** `article_entities`.
- **Method:**
  1. Extract candidate names from the article text. Options (pick one):
     - **Easy starter (hosted):** call an **NER model on the HF Inference API**, e.g. `hf.token_classification(text, model="dslim/bert-base-NER")` (or a finance NER model). Aggregate sub-tokens into whole names. No local model.
     - **Even simpler starter (no model):** fuzzy/exact match candidate phrases against your S&P 500 name+alias list (from the knowledge base). Cheapest and surprisingly effective for v1.
     - **Paid-API alternative:** ask your `call_llm` (Claude) to extract company mentions as JSON — one call can return names *and* tickers, folding steps 1–2 together.
     - The brief mentions a *fine-tuned* NER model — that's a separate GPU training task; start with a pretrained/hosted one and note fine-tuning as future work.
  2. **Link** each candidate to an S&P 500 entry: embed the candidate name, query ChromaDB, take the top match if similarity is high enough; record the ticker + sector + a confidence score.
  3. Handle aliases via the knowledge base (e.g., "Alphabet"/"Google" both → GOOGL).
- **Beginner notes:** Entity linking is the hardest accuracy problem here. Start simple (exact + fuzzy name match against the S&P 500 list), then add embedding-based linking. Always store a **confidence**; low-confidence links should weaken downstream signals.

### 8.3 Sentiment Analysis Agent
- **Purpose:** Score how positive/negative the news is — per sentence, then aggregated per entity.
- **Inputs:** `deduped_articles`, `article_entities`.
- **Outputs:** `entity_sentiments`, `sentiment_trajectory`.
- **Method:**
  1. Split each article into sentences.
  2. Call **FinBERT** (`ProsusAI/finbert`) on each sentence **via the Hugging Face Inference API** (see Section 10.2) → {positive, negative, neutral} with probabilities. (No local model load — it's an HTTP call.)
  3. **Aggregate to entity level.** The brief says *attention-weighted averaging*. Concretely:
     - Weight sentences that mention the entity (or its ticker) more heavily.
     - Optionally weight by FinBERT confidence and by sentence position (headline/lead sentences matter more).
     - Produce a single signed score per (article, entity): e.g., `score = mean(weight_i * (P_pos_i - P_neg_i))`.
  4. **Trajectory:** maintain a per-ticker rolling **7-day** series of daily aggregated sentiment (store to a file under `data/` so it persists across runs). This lets you show "sentiment is trending up for AAPL."
- **Beginner notes:**
  - FinBERT has a **512-token limit** — work sentence by sentence, not whole articles. (One sentence per API call is also the natural caching unit.)
  - Map labels to a number for averaging: positive=+1, neutral=0, negative=−1, scaled by probability.
  - "Attention-weighted" can start as "weight = 1 if the sentence mentions the entity, else 0.2." Don't over-engineer first.

### 8.4 Event Detection Agent
- **Purpose:** Classify the *type* of event and its market-impact severity.
- **Inputs:** `deduped_articles`, retrieved KB context.
- **Outputs:** `events` (with `category`, `severity`, `rationale`).
- **Event taxonomy (categories):** `earnings`, `M&A`, `regulatory_action`, `management_change`, `product_launch`, `litigation`, `other`.
- **Severity levels:** `low / medium / high`.
- **Method:** Call your **`call_llm` wrapper** (Mistral on HF, or Claude on the paid path — Section 10) with **few-shot prompting**: give it 3–5 labeled examples, then the new article, and ask for a JSON answer `{category, severity, rationale}`. On the HF path, parse the returned text with a tolerant JSON parser. On the Claude path, prefer **structured outputs** (`messages.parse` with a Pydantic schema, Section 10.3) so the JSON is guaranteed valid — no parsing needed. (Prompt in Appendix B.)
- **Beginner notes:**
  - Force **structured JSON output** and validate it. On the HF/Mistral path the model sometimes adds prose — strip code fences, use a robust parser, and retry once on failure. The Claude structured-outputs path avoids this entirely.
  - Severity is subjective; define clear rules in the prompt ("M&A and regulatory action are usually high; routine product launches are low").
  - This is the node that feeds the two conditional branches (high-severity → human review; earnings → sub-agent), so its output drives the graph's decisions.

### 8.5 Earnings Analysis Sub-Agent (conditional)
- **Purpose:** A specialized deeper analysis when the event is an **earnings** report.
- **Triggered by:** conditional edge when `category == "earnings"`.
- **Method:** Pull the relevant SEC EDGAR filing or yfinance earnings data, extract EPS beat/miss and guidance, and produce a richer evidence note that feeds the Signal agent. Start minimal (just note "earnings event, beat/miss if detectable") and expand later.

### 8.6 Signal Generation Agent
- **Purpose:** Combine everything into a final, structured investment signal.
- **Inputs:** `entity_sentiments`, `events`, plus **historical price correlation** from `yfinance`.
- **Outputs:** `signals` — a list of validated **Pydantic** objects.
- **Method:** See the full algorithm in [Section 11](#11-the-signal-generation-algorithm).
- **Output schema (Pydantic):**
  ```python
  from pydantic import BaseModel, Field
  from typing import Literal, List
  class Signal(BaseModel):
      entity: str
      ticker: str
      signal: Literal["bullish", "bearish", "neutral"]
      confidence: float = Field(ge=0, le=1)
      evidence: List[str]
      timestamp: str
  ```

### 8.7 Briefing Report Generator
- **Purpose:** Write a human-readable **daily market intelligence briefing**.
- **Inputs:** All of state (sentiments, events, signals).
- **Outputs:** `briefing_markdown` (and a PDF export).
- **Method:** Either template-fill with Pandas/string formatting (reliable, free) **or** call your `call_llm` wrapper (Mistral on HF, or Claude on the paid path) to write a narrative summary **from the structured data** (nicer prose, but verify it doesn't hallucinate numbers — feed it the exact numbers and ask only for phrasing). Export Markdown → PDF with a simple converter. The briefing is one call per day, so even on the paid path its cost is negligible — fine to use the top model here.

---

## 9. The LangGraph State Machine

This is where the agents become an *agentic system*. You declare nodes, then edges (including conditional ones), then compile and run.

### 9.1 Building the graph
```python
from langgraph.graph import StateGraph, START, END

g = StateGraph(FinanceState)

# 1. Register each agent as a node
g.add_node("ingest",   news_ingestion_agent)
g.add_node("entities", entity_recognition_agent)
g.add_node("sentiment", sentiment_agent)
g.add_node("events",   event_detection_agent)
g.add_node("earnings", earnings_subagent)
g.add_node("human_review", human_review_node)
g.add_node("signals",  signal_generation_agent)
g.add_node("briefing", briefing_agent)

# 2. The linear backbone
g.add_edge(START, "ingest")
g.add_edge("ingest", "entities")
g.add_edge("entities", "sentiment")
g.add_edge("sentiment", "events")

# 3. Conditional routing AFTER events
def route_after_events(state: FinanceState):
    cats = {e["category"] for e in state["events"]}
    sev  = {e["severity"] for e in state["events"]}
    if "earnings" in cats:
        return "earnings"
    if "high" in sev:
        return "human_review"
    return "signals"

g.add_conditional_edges("events", route_after_events,
                        {"earnings": "earnings",
                         "human_review": "human_review",
                         "signals": "signals"})

g.add_edge("earnings", "signals")        # earnings sub-agent then continues
g.add_edge("human_review", "signals")    # after approval, continue
g.add_edge("signals", "briefing")
g.add_edge("briefing", END)

app = g.compile(checkpointer=memory_saver)   # checkpointer enables pause/resume
```

> **Note on routing realism:** The simple `route_after_events` above returns a *single* branch. In practice you may have both earnings *and* high-severity events. Two clean options for a beginner: (a) handle one concern per node and chain checks (earnings node internally sets `needs_human_review`), or (b) do severity/earnings handling **inside** the signal agent and reserve the conditional edge purely for the human-review interrupt. Start with option (b) — it's simpler — and graduate to richer routing later.

### 9.2 The human-in-the-loop interrupt
High-severity events should pause for human approval. LangGraph supports this via an **interrupt** + **checkpointer**:
- Compile with a checkpointer (e.g., `MemorySaver()` or SQLite) so state can be saved and resumed.
- Configure the graph to **interrupt before** the `human_review` node (`interrupt_before=["human_review"]`), or call `interrupt()` inside it.
- When interrupted, the graph **stops and returns**. Your dashboard shows the pending high-impact item. A human clicks Approve/Reject, you set `state["human_decision"]`, and you **resume** with `app.invoke(None, config=...)` using the same thread id.
- **Beginner note:** HITL is the trickiest LangGraph feature. Build and test the whole graph *without* it first (let everything flow to signals). Add the interrupt last, once the rest works.

### 9.3 Running it
```python
config = {"configurable": {"thread_id": "run-2026-06-06"}}
initial = {"raw_articles": [], "deduped_articles": [], ... }   # empty state
result = app.invoke(initial, config=config)
print(result["briefing_markdown"])
```

> **Mental model:** `invoke` walks the graph from START to END, calling each node function, merging its returned dict into the state, and following edges. Conditional edges call your router function to decide where to go next.

---

## 10. Models & Hosted Inference (HF API / Paid API)

You use **three** models, but you **don't run any heavy ones yourself** — you call them over the network. There is no GPU, no `torch`, and no quantization in this design.

| Model | Role | How you run it |
|---|---|---|
| `all-MiniLM-L6-v2` (sentence-transformers) | Embeddings for the RAG knowledge base | **Locally on CPU** — it's tiny (~80 MB) and called constantly, so a network round-trip per call would be wasteful. |
| `ProsusAI/finbert` | Sentence sentiment | **Hugging Face Inference API** (hosted). |
| An instruction LLM (event classification + briefing prose) | Reasoning / generation | **Pick one path below.** |

### 10.1 The key decision: which LLM path?
The generative model (Event Detection + the briefing narrative) is the one place you choose. Both paths plug into the same agent code behind a thin wrapper function `call_llm(prompt) -> str`.

| | **HF path** (Mistral-7B via HF Inference API) | **Paid-API path** (Anthropic Claude — recommended for ease & quality) |
|---|---|---|
| Cost | Free tier (rate-limited); may queue or "cold start" | Pay per token (see pricing below) |
| Setup | One HF token; accept the model license | One API key |
| Output quality on classification/JSON | Good | Best; very reliable structured JSON |
| Reliability | Free endpoints can be slow or temporarily unavailable | High |
| Best for | Zero-budget student build | When you want it to "just work" and can spend a few dollars |

**Recommendation for a first build:** use **FinBERT on the HF Inference API** for sentiment (it's purpose-built and free), and for the *generative* LLM start on the **HF path with Mistral** to keep costs at zero; if you find the free endpoint slow or the JSON unreliable, switch the single `call_llm` wrapper to the **paid Claude path**. Because everything goes through that one function, swapping is a few lines.

### 10.2 HF path — calling hosted models on Hugging Face
The `huggingface_hub` client talks to the Inference API. **FinBERT (sentiment):**
```python
from huggingface_hub import InferenceClient
from src.config import HF_TOKEN

hf = InferenceClient(token=HF_TOKEN)

def finbert_sentiment(sentence: str) -> dict:
    # returns a list of {label, score} for positive/neutral/negative
    return hf.text_classification(sentence, model="ProsusAI/finbert")
```
**Mistral (event classification / briefing) — the `call_llm` wrapper:**
```python
def call_llm(prompt: str, max_new_tokens: int = 256) -> str:
    return hf.text_generation(
        prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=max_new_tokens,
        temperature=0.2,            # low temp = more consistent JSON
        return_full_text=False,
    )
```
Use Mistral's instruction format (`[INST] ... [/INST]`) in the prompt — see Appendix B.

### 10.3 Paid-API path — calling Anthropic Claude
Install `anthropic`, set `ANTHROPIC_API_KEY` in `.env`, then implement the **same** `call_llm` signature so the rest of the code doesn't change:
```python
import anthropic
from src.config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    resp = client.messages.create(
        model="claude-opus-4-8",                 # most capable; see table below
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in resp.content if b.type == "text")
```
For **event classification** (you want strictly-valid JSON back), prefer Claude's **structured outputs** instead of parsing free text — it guarantees the shape and removes a whole class of bugs:
```python
from pydantic import BaseModel
from typing import Literal

class EventLabel(BaseModel):
    category: Literal["earnings","M&A","regulatory_action","management_change",
                      "product_launch","litigation","other"]
    severity: Literal["low","medium","high"]
    rationale: str

resp = client.messages.parse(
    model="claude-opus-4-8",
    max_tokens=512,
    messages=[{"role": "user", "content": event_prompt(article_text)}],
    output_format=EventLabel,           # validates the response against the schema
)
label: EventLabel = resp.parsed_output  # already a typed object — no json.loads needed
```

**Current Claude models & pricing** (per 1M tokens; pick by cost vs. capability):

| Model | Model ID | Input $/1M | Output $/1M | Use it for |
|---|---|---|---|---|
| Claude Opus 4.8 | `claude-opus-4-8` | $5.00 | $25.00 | Highest quality; default when correctness matters |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | $3.00 | $15.00 | Strong balance of cost and quality for higher volume |
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1.00 | $5.00 | Cheapest/fastest; fine for simple event classification |

> **Cost intuition for a student:** classifying a few dozen short articles a day is *tiny* — typically well under a dollar a day, and far less on Haiku. Use the cheaper model for the per-article classification calls and reserve the top model for the once-a-day briefing if you want. (You can also swap to OpenAI by re-implementing only `call_llm`; this design is provider-agnostic behind that wrapper.)

### 10.4 Hosted-inference hygiene (replaces "GPU memory hygiene")
- **Cache results.** Don't re-send the same article to FinBERT or the LLM twice — store outputs keyed by article id in `data/`. This saves money, time, and rate-limit budget.
- **Handle rate limits (`429`).** Wrap calls in a small retry-with-backoff helper; the `anthropic` SDK already retries `429`/5xx automatically. For HF free endpoints, also handle "model loading" responses by waiting and retrying.
- **Keep prompts short** and `max_tokens` modest for classification (you only need a small JSON object back).
- **Batch sentiment** where the API allows passing a list, to cut the number of round-trips.

### 10.5 FinBERT fine-tuning (a deliverable) — what changes when models are hosted
The brief asks you to discuss FinBERT fine-tuning. Note the nuance: **fine-tuning means *training* a model, which the Inference API does not do — that is a separate, GPU step.** Plan:
- **Baseline (no training, do this first):** call hosted `ProsusAI/finbert` over the Financial PhraseBank test set and record accuracy/F1. This is your benchmark and satisfies the core evaluation deliverable **without any GPU**.
- **Optional fine-tune (separate task):** if you want to *report* on fine-tuning, do the training in a **one-off GPU environment** (a Colab/Kaggle notebook or a paid GPU) using HuggingFace `Trainer` — small LR (~2e-5), 2–3 epochs, held-out test split, never evaluate on training data. Then either push your fine-tuned weights to the HF Hub and call them via the Inference API (paid **Inference Endpoints** for a private model), or just report the offline before/after numbers. For a first build, the **baseline evaluation is enough** — present fine-tuning as analysis/future work and be honest that gains on PhraseBank are usually small (FinBERT was largely built on that data).

---

## 11. The Signal Generation Algorithm

This is the analytical core. Make it **transparent and rule-based first** (easy to debug and to explain in your report); you can add ML later.

**Inputs per entity (ticker):**
- `S` = aggregated sentiment score in [−1, +1] (from FinBERT).
- `E` = event severity weight: low=0.3, medium=0.6, high=1.0; and an event-type sign (e.g., litigation/regulatory lean negative, product launch/earnings beat lean positive).
- `C` = historical price correlation: over the last 30 days, how has this ticker's price reacted to similar sentiment? Compute with `yfinance` returns (see below).
- `L` = entity-linking confidence in [0, 1] (penalize uncertain links).

**Step 1 — direction:**
```
direction_score = S * event_sign        # in [-1, 1]
if direction_score >  threshold_pos: signal = "bullish"
elif direction_score < threshold_neg: signal = "bearish"
else: signal = "neutral"
```

**Step 2 — confidence (combine the signals):**
```
confidence = clip( w1*|S| + w2*E + w3*|C| , 0, 1 ) * L
# e.g., w1=0.4, w2=0.35, w3=0.25 ; L scales the whole thing down if linking was shaky
```

**Step 3 — evidence:** collect the concrete reasons — the top sentences driving sentiment, the event category/severity, and the price-correlation stat — into the `evidence` list so every signal is **explainable**.

**Computing price correlation `C` (with yfinance):**
```python
import yfinance as yf
hist = yf.Ticker(ticker).history(period="30d")
returns = hist["Close"].pct_change().dropna()
# Simple version: correlate daily sentiment series with next-day returns.
# C = correlation(sentiment_trajectory[ticker], returns.shift(-1))
```
> **Beginner caution:** With only 30 days you have very little data — treat `C` as a weak hint, not gospel. Report this limitation honestly. Avoid over-claiming predictive power.

**Step 4 — produce Pydantic `Signal` objects** (Section 8.6). Validation guarantees every signal has all fields and a confidence in [0,1].

---

## 12. The Dashboard (Gradio)

Gradio turns Python functions into a web UI with a few lines — ideal for a student demo.

**Four panels:**
1. **Sentiment heatmap by sector** — a `matplotlib` grid (rows=sectors, cols=days, color=sentiment) rendered into a Gradio `Plot`.
2. **Event timeline** — events on a time axis, colored/sized by severity markers.
3. **Signal cards** — for each signal: ticker, bullish/bearish/neutral badge, a **confidence bar**, and the evidence list (expandable).
4. **Briefing download** — a button to download the daily briefing as **Markdown / PDF**.

```python
import gradio as gr

def run_pipeline():
    result = app.invoke(empty_state(), config=config)
    return (sector_heatmap_fig(result),
            event_timeline_fig(result),
            signal_cards_html(result),
            result["briefing_markdown"])

with gr.Blocks() as demo:
    gr.Markdown("# Financial News Intelligence")
    btn = gr.Button("Run analysis")
    heatmap = gr.Plot(); timeline = gr.Plot()
    cards = gr.HTML(); briefing = gr.Markdown()
    btn.click(run_pipeline, outputs=[heatmap, timeline, cards, briefing])

demo.launch()   # opens at http://127.0.0.1:7860 in your browser
# add share=True only if you want a temporary public URL to show someone
```
Run it from the VS Code terminal with `python -m src.app` (venv active). VS Code can also debug it with breakpoints via the Run panel.

**For the human-review interrupt:** add a panel that appears when the pipeline is paused, showing the high-impact item with **Approve / Reject** buttons that set `human_decision` and resume the graph.

---

## 13. Evaluation & Backtesting

Two required evaluations — these are deliverables, so do them rigorously.

### 13.1 Sentiment accuracy (Financial PhraseBank)
- Run FinBERT over the PhraseBank test sentences; compare predicted vs. true labels.
- Report **accuracy, precision/recall/F1 per class, and a confusion matrix** (use `sklearn.metrics`).
- If you fine-tune, report **before vs. after**.

### 13.2 Signal backtesting (signal–price correlation)
- For each signal you generated, look at the entity's **actual price move** over the following 1–5 days (`yfinance`).
- Define "correct" simply: bullish signal followed by a price rise, bearish by a fall, neutral by a small move.
- Report **hit rate** and the **correlation between confidence and correctness** (do higher-confidence signals do better?).
- **Be honest about limitations:** small sample, short horizon, no transaction costs, correlation ≠ causation. A good report states these.

> **Beginner reminder:** This is an educational/research project, **not financial advice**. State that clearly in your report and dashboard.

---

## 14. Phased Implementation Roadmap

Build in this order. Each phase is runnable and testable on its own — **do not skip ahead**.

| Phase | Goal | You can demo… |
|---|---|---|
| **0. Setup** | VS Code project + `.venv` + `requirements.txt` + `.env` keys; make one successful FinBERT call and one successful `call_llm` call | "Hosted models respond to a hello-world request." |
| **1. Ingestion** | Fetch RSS, extract text, dedupe with MinHash | Print N unique articles. |
| **2. Sentiment (standalone)** | FinBERT on sentences; evaluate on PhraseBank | Accuracy number + confusion matrix. |
| **3. Knowledge base** | Build ChromaDB; query returns right company | "iPhone maker" → AAPL. |
| **4. Entities** | NER + linking to tickers | Article → list of tickers. |
| **5. Events** | Mistral few-shot classification → JSON | Article → {category, severity}. |
| **6. Signals** | Rule-based algorithm → Pydantic Signal | Validated signal objects with evidence. |
| **7. Wire LangGraph** | Connect 1–6 as a linear graph; run end-to-end | One `invoke` → briefing. |
| **8. Conditional edges** | Earnings routing + high-severity routing | Graph branches correctly. |
| **9. Human-in-the-loop** | Interrupt + resume on high-severity | Pause → approve → continue. |
| **10. Dashboard** | Gradio: heatmap, timeline, cards, download | Clickable demo. |
| **11. Backtest + report** | yfinance backtest; write technical report | Deliverables complete. |
| **12. FinBERT fine-tuning** | Fine-tune FinBERT on Financial PhraseBank in Colab/Kaggle; compare before/after accuracy; push weights to HF Hub or report offline results | Before vs. after accuracy numbers + written analysis. |

**Golden rule:** get a thin end-to-end slice working by Phase 7 (even if each agent is crude), *then* improve quality. A working ugly pipeline beats a perfect half-pipeline.

---

## 15. Risks, Pitfalls & Beginner Tips

- **Leaking your API keys** is the #1 risk now. Keep them in `.env`, add `.env` to `.gitignore`, and **never** commit or paste them. If a key ever lands in a commit, rotate it immediately.
- **Network / hosted-API failures**: calls can time out, rate-limit (`429`), or (on HF free endpoints) return "model is loading." Wrap every model call in try/except with retry-and-backoff; the `anthropic` SDK already retries `429`/5xx for you.
- **`newspaper3k` failures**: many sites block scraping. Always `try/except`; fall back to the RSS `.summary` if full text fails.
- **LLM JSON parsing**: on the HF/Mistral path the model may wrap JSON in prose or markdown fences — use a tolerant parser, strip fences, retry once with "return ONLY JSON." On the Claude path, use structured outputs (`messages.parse`) and skip parsing entirely.
- **FinBERT 512-token limit**: never feed whole articles; go sentence by sentence.
- **Rate limits**: SEC EDGAR and yfinance can throttle you, and so can the model APIs. Add delays, and **cache every model/result to `data/`** so re-runs don't re-call.
- **Cost control (paid path)**: bill is per token. Cache aggressively, use a cheaper model (Haiku) for the high-volume per-article classification, keep `max_tokens` small, and watch your provider's usage dashboard while developing.
- **Entity linking errors cascade**: a wrong ticker poisons sentiment → signal. Keep and propagate a confidence; show low-confidence links differently.
- **Don't over-engineer "attention-weighted averaging"** at first — a simple mention-based weight is fine for v1.
- **Persisting state across runs**: sentiment trajectory and price history should be saved under `data/` (e.g. CSV/JSON/SQLite). Locally this is easy — just don't keep state only in memory.
- **Reproducibility**: pin package versions in `requirements.txt`; pin the exact model IDs; cache datasets and API responses.
- **Ethics/disclaimer**: label outputs clearly as research, not investment advice.
- **Latency**: hosted calls add network round-trips. Develop with a handful of articles, cache results, and run sentiment/classification calls concurrently (e.g. a thread pool) once the logic works.

---

## 16. Mapping to Deliverables

| Required Deliverable | Where it's produced |
|---|---|
| End-to-end pipeline | The VS Code project (`src/`) + compiled LangGraph `app` (Phases 0–10). *If the brief specifically wants a notebook*, you can additionally expose the pipeline from a thin `.ipynb` that imports `src/` and calls `app.invoke(...)`. |
| 7-day market intelligence report | Sentiment trajectory (8.3) + events + signals → Briefing agent (8.7), run daily for 7 days |
| Sentiment accuracy on Financial PhraseBank | Section 13.1 (hosted FinBERT baseline — no GPU needed) |
| Backtest (signal–price correlation) | Section 13.2 |
| Technical report (FinBERT fine-tuning, entity linking, event taxonomy, signal algorithm) | Sections 10.5, 8.2, 8.4, 11 — write these up with your results |

---

## Appendix A: Data Schemas

**Event object**
```python
{
  "article_id": "yahoo_12345",
  "category": "earnings",            # one of the taxonomy categories
  "severity": "high",                # low | medium | high
  "rationale": "Company reported EPS well above consensus and raised guidance.",
  "entities": ["AAPL"]
}
```

**Signal object (Pydantic, serialized)**
```python
{
  "entity": "Apple Inc.",
  "ticker": "AAPL",
  "signal": "bullish",
  "confidence": 0.78,
  "evidence": [
     "Sentiment +0.62 across 5 mentions (FinBERT).",
     "Event: earnings, severity high — EPS beat + raised guidance.",
     "30-day sentiment–return correlation: +0.31."
  ],
  "timestamp": "2026-06-06T14:30:00Z"
}
```

**Sentiment trajectory (per ticker)**
```python
{ "AAPL": [
    {"date":"2026-05-31","score": 0.12},
    {"date":"2026-06-01","score": 0.20},
    ...
]}
```

---

## Appendix B: Prompt Templates

**Event detection (few-shot)** — the prompt body is the same on both paths; only the wrapping differs. On the **HF/Mistral** path, wrap it in Mistral's instruction format `[INST] ... [/INST]` (shown below). On the **Claude** path, drop the `[INST]` tags and pass the body as the user message — and prefer structured outputs (Section 10.3) so you don't parse JSON at all:

```
[INST]
You are a financial event classifier. Classify the news into exactly one category and a severity.
Categories: earnings, M&A, regulatory_action, management_change, product_launch, litigation, other.
Severity rules: M&A and regulatory_action are usually high; litigation medium-high;
earnings high if a clear beat/miss, else medium; routine product_launch low.
Return ONLY valid JSON: {"category": "...", "severity": "...", "rationale": "..."}.

Examples:
News: "Acme Corp to acquire Beta Inc for $4B in all-cash deal."
{"category": "M&A", "severity": "high", "rationale": "Large all-cash acquisition."}

News: "GadgetCo unveils new mid-range phone at annual event."
{"category": "product_launch", "severity": "low", "rationale": "Routine product reveal."}

Now classify:
News: "{ARTICLE_TEXT}"
[/INST]
```

**Briefing narrative (optional, Mistral)** — feed exact numbers; ask only for phrasing:

```
[INST]
Write a concise daily market-intelligence briefing in Markdown from the structured data below.
Do NOT invent numbers; use only what is given. Sections: Overview, Top Movers (by sentiment),
Notable Events, Signals (with confidence), Caveats.

DATA:
{JSON_OF_SENTIMENTS_EVENTS_SIGNALS}
[/INST]
```

---

*End of design document. Build Phase 0 next: set up the VS Code project and `.env`, then confirm one FinBERT call and one `call_llm` call return successfully. Then proceed phase by phase.*
