# Entity Recognition Agent
#
# Pipeline for each article:
#   1. Use spaCy NER (en_core_web_sm) to pull ORG mentions out of the text.
#      If spaCy/the model isn't available, fall back to plain text matching against known names.
#   2. Resolve each ORG against the S&P 500 companies stored in ChromaDB.
#   3. If it's NOT a known company: clean the span and apply cheap shape guards (reject headline
#      fragments, generic words, acronyms), then ask the LLM: is this a real company, its S&P 500
#      parent (if any), and its own stock ticker (if any)?
#         - not a company    -> drop it (the LLM filters generic words the guards missed).
#         - subsidiary       -> persist the span as an ALIAS on the parent's ChromaDB metadata so it
#                               matches instantly next time (no LLM call).
#         - has a ticker     -> validate it against yfinance; if real, add a full profile to the
#                               "tracked_companies" KB — now matchable + signal-eligible like S&P 500.
#         - private/unlisted -> add to the "news_entities" collection (name-only) to track firms in the news.
# The lookup is loaded from BOTH sp500_companies and tracked_companies, so the DB self-expands over time.

import re
import json
from datetime import datetime, timezone

import chromadb
from src.config import CHROMA_DB_PATH
from src.llm import call_llm
from loguru import logger

try:
    import spacy
except ImportError:  # spaCy is optional — we degrade to text matching without it
    spacy = None

SP500_COLLECTION = "sp500_companies"
TRACKED_COMPANIES_COLLECTION = "tracked_companies"  # non-S&P 500 firms with a yfinance-validated profile (signal-eligible)
TRACKED_COLLECTION = "news_entities"   # non-S&P 500 companies spotted in the news (name-only, no valid ticker)
SPACY_MODEL = "en_core_web_sm"
# spaCy is unreliable about company labels — it flips the same firm between ORG/PRODUCT/GPE
# depending on context. So we route by label:
#   PRIMARY  -> full resolution (S&P 500 match, else LLM subsidiary check, else track as news entity)
#   LOOKUP   -> accept ONLY if the span is already a known S&P 500 name; never LLM-check or track
#              (recovers companies mislabeled GPE/PERSON without tracking "China"/"New York" as firms)
PRIMARY_LABELS = {"ORG", "PRODUCT"}
LOOKUP_LABELS = {"GPE", "PERSON", "WORK_OF_ART", "FAC", "NORP"}
ALIAS_SEP = "|"                        # ChromaDB metadata values must be scalar, so aliases are a joined string
MAX_CHARS = 20000                      # cap text handed to spaCy per article

SUFFIXES_TO_REMOVE = (
    "Incorporated", "Corporation", "Company", "Holdings", "Group", "Inc", "Corp",
    "Co", "Ltd", "plc", "LLC", "LP", "NV", "SA", "AG", "Class A", "Class B", "Class C",
)

# Orgs that surface constantly in finance news but aren't tradable companies we want to track.
# Skipped before the LLM subsidiary check to avoid wasted calls and noise. Editable.
ORG_STOPLIST = {
    "reuters", "bloomberg", "cnbc", "cnn", "the wall street journal", "wall street journal",
    "wall street", "associated press", "federal reserve", "the federal reserve", "the fed",
    "sec", "irs", "congress", "senate", "white house", "eu", "european union",
}

_nlp = False   # False = not yet attempted; None = unavailable; otherwise the loaded pipeline
_ef = None     # lazy SentenceTransformer embedding fn for the tracked collection (same model as the KB)


def _embedding_fn():
    """Embed tracked entities with the same all-MiniLM-L6-v2 the S&P 500 KB uses (cached locally, no network)."""
    global _ef
    if _ef is None:
        from chromadb.utils import embedding_functions
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return _ef


def _load_nlp():
    """Load spaCy NER once. Returns the pipeline, or None if spaCy/the model isn't installed."""
    global _nlp
    if _nlp is not False:
        return _nlp
    if spacy is None:
        logger.warning("spaCy not installed — using text-matching fallback for entity recognition.")
        _nlp = None
        return _nlp
    try:
        _nlp = spacy.load(SPACY_MODEL, disable=["lemmatizer", "tagger", "parser"])
        logger.info(f"Loaded spaCy model '{SPACY_MODEL}' for NER.")
    except OSError:
        logger.warning(f"spaCy model '{SPACY_MODEL}' not found — using text-matching fallback.")
        _nlp = None
    return _nlp


def _normalize(name: str) -> str:
    """Strip parentheticals and corporate suffixes so 'NVIDIA Corporation' and 'Nvidia' collapse together."""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    name = name.strip().rstrip(".,&").strip()
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIXES_TO_REMOVE:
            if name.lower().endswith(" " + suffix.lower()):
                name = name[: -(len(suffix) + 1)].strip().rstrip(".,&").strip()
                changed = True
    return name


def _clean_span(span: str) -> str:
    """Tidy a raw NER span: collapse whitespace/newlines, drop possessive 's, drop a leading 'the'."""
    span = re.sub(r"\s+", " ", span).strip()               # "Zeta Global\n\nZeta" -> "Zeta Global Zeta"
    span = re.sub(r"[’']s\b", "", span)                    # "Apple's" -> "Apple", "Big Tech's" -> "Big Tech"
    span = re.sub(r"^the\s+", "", span, flags=re.IGNORECASE)
    return span.strip()


def _looks_like_company(span: str) -> bool:
    """Cheap shape check to reject non-company spans before spending an LLM call or tracking them."""
    if not span or len(span) > 40:
        return False                                       # empty or headline-length fragment
    if len(span.split()) > 4:
        return False                                       # sentence/headline fragment, not a name
    if span.islower():
        return False                                       # proper nouns aren't all-lowercase
    if re.search(r"[:;?!|(),&]", span):
        return False                                       # "Zeta Global: Which...", "Alphabet (GOOG, GOOGL", "target & stock"
    if len(span.split()) == 1 and re.search(r"\d", span):
        return False                                       # digit junk like "416x"
    if not any(c.isalpha() for c in span):
        return False
    return True


def _load_companies_into(lookup: dict[str, dict], collection) -> None:
    """Add a company collection's names, tickers, and saved aliases into the shared lookup dict."""
    result = collection.get(include=["metadatas"])
    for doc_id, meta in zip(result["ids"], result["metadatas"]):
        entry = {"ticker": meta["id"], "name": meta["Company name"], "doc_id": doc_id, "collection": collection.name}
        lookup[_normalize(meta["Company name"]).lower()] = entry
        if len(meta["id"]) >= 2:
            lookup[meta["id"].lower()] = entry
        for alias in (meta.get("aliases") or "").split(ALIAS_SEP):
            alias = alias.strip()
            if alias:
                lookup[_normalize(alias).lower()] = entry
    logger.info(f"Loaded {collection.count()} companies from '{collection.name}' (lookup now {len(lookup)} keys).")


def _extract_candidates(nlp, text: str) -> list[tuple[str, bool]]:
    """Return de-duplicated (span, is_primary) candidates, preserving first-seen order.
    is_primary=True for ORG/PRODUCT (full resolution); False for GPE/PERSON/etc (lookup-only)."""
    doc = nlp(text[:MAX_CHARS])
    out, seen = [], set()
    for ent in doc.ents:
        if ent.label_ in PRIMARY_LABELS or ent.label_ in LOOKUP_LABELS:
            name = _clean_span(ent.text)
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append((name, ent.label_ in PRIMARY_LABELS))
    return out


SUBSIDIARY_SYSTEM_PROMPT = (
    "You are a corporate-structure expert. You are given a text span pulled from a news article. "
    "First decide whether it actually names a real company, brand, or organization — NOT a generic "
    "word, acronym, market index, government body, or headline fragment. If it is a company, decide "
    "whether it is a subsidiary/brand/division/trade name of a larger publicly traded PARENT company, "
    "and give its stock ticker if it is itself publicly traded. "
    'Return ONLY JSON: {"is_company": true or false, "parent_company": "<full parent company name, or null>", '
    '"ticker": "<the company\'s own stock ticker if publicly traded, else null>"}. '
    "Set parent_company to null if it is independent or the ultimate parent. Set ticker to null if it is "
    "private, not listed, or you are unsure — never guess a ticker."
)


def _check_subsidiary(org: str, lookup: dict[str, dict]) -> dict:
    """Ask the LLM if `org` is a real company and, if so, its S&P 500 parent and/or its own ticker.
    Returns {"is_company": bool, "parent": <lookup entry or None>, "ticker": <str or None>}."""
    try:
        data = json.loads(call_llm(f'Text span: "{org}"', SUBSIDIARY_SYSTEM_PROMPT))
    except Exception as e:  # LLM/network/JSON — never let a lookup failure break the pipeline
        logger.warning(f"Subsidiary check failed for {org!r}: {e}")
        return {"is_company": False, "parent": None, "ticker": None}
    parent_name = data.get("parent_company")
    parent = lookup.get(_normalize(parent_name).lower()) if parent_name else None
    if parent:
        logger.info(f"'{org}' resolved as subsidiary of {parent['name']} ({parent['ticker']}).")
    ticker = data.get("ticker") or None
    return {"is_company": bool(data.get("is_company")), "parent": parent, "ticker": ticker}


def _add_alias(collection, parent: dict, alias: str) -> None:
    """Persist `alias` onto the parent company's ChromaDB metadata so it matches without an LLM call next time."""
    try:
        record = collection.get(ids=[parent["doc_id"]], include=["metadatas"])
        meta = record["metadatas"][0]
        aliases = [a for a in (meta.get("aliases") or "").split(ALIAS_SEP) if a]
        if alias not in aliases:
            aliases.append(alias)
            meta["aliases"] = ALIAS_SEP.join(aliases)
            collection.update(ids=[parent["doc_id"]], metadatas=[meta])
            logger.info(f"Added alias '{alias}' to {parent['name']} metadata.")
    except Exception as e:
        logger.warning(f"Could not persist alias '{alias}' for {parent['name']}: {e}")


def _fetch_yf_profile(ticker: str) -> dict | None:
    """Validate an LLM-suggested ticker against yfinance. Returns a profile dict, or None if it's not a real listed stock."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.debug(f"yfinance lookup failed for {ticker!r}: {e}")
        return None
    name = info.get("longName") or info.get("shortName")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    quote_type = info.get("quoteType")
    # Require a real name + a price, and (when yfinance says) that it's an equity — guards against
    # hallucinated tickers, indices, currencies, and delisted symbols.
    if not name or price is None or (quote_type and quote_type != "EQUITY"):
        logger.debug(f"Ticker {ticker!r} rejected (name={name!r}, price={price}, type={quote_type}).")
        return None
    return {
        "ticker": ticker.upper(),
        "name": name,
        "sector": info.get("sector") or "Unknown",
        "sub_industry": info.get("industry") or "Unknown",
        "description": info.get("longBusinessSummary") or f"{name} — company tracked from the news.",
    }


def _promote_company(companies_collection, profile: dict) -> dict:
    """Add a yfinance-validated company as a full profile in the tracked_companies KB. Returns its lookup entry."""
    doc_id = f"tracked_{profile['ticker']}"
    text = (f"ticker: {profile['ticker']} | Company name: {profile['name']} | Sector: {profile['sector']} | "
            f"Sub industry: {profile['sub_industry']} | Description: {profile['description']}")
    meta = {"id": profile["ticker"], "Company name": profile["name"], "Sector": profile["sector"],
            "Sub industry": profile["sub_industry"], "Description": profile["description"]}
    try:
        companies_collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta])  # upsert dedups by ticker
        logger.info(f"Promoted '{profile['name']}' ({profile['ticker']}) into the tracked_companies KB.")
    except Exception as e:
        logger.warning(f"Could not promote company {profile['ticker']}: {e}")
    return {"ticker": profile["ticker"], "name": profile["name"], "doc_id": doc_id,
            "collection": companies_collection.name}


def _track_unknown(tracked_collection, org: str) -> None:
    """Record a non-S&P 500 company in the news_entities collection (dedup by slug, bump a mention count)."""
    eid = "news_" + re.sub(r"[^a-z0-9]+", "_", org.lower()).strip("_")
    if eid == "news_":
        return
    try:
        existing = tracked_collection.get(ids=[eid], include=["metadatas"])
        now = datetime.now(timezone.utc).isoformat()
        if existing["ids"]:
            meta = existing["metadatas"][0]
            meta["mentions"] = int(meta.get("mentions", 1)) + 1
            meta["last_seen"] = now
            tracked_collection.update(ids=[eid], metadatas=[meta])
        else:
            tracked_collection.add(
                ids=[eid],
                documents=[org],
                metadatas=[{"name": org, "mentions": 1, "first_seen": now, "last_seen": now}],
            )
            logger.info(f"Tracking new non-S&P 500 entity from the news: '{org}'.")
    except Exception as e:
        logger.warning(f"Could not track unknown entity '{org}': {e}")


def _resolve_org(org, lookup, collections, tracked_collection, cache, allow_discovery=True) -> dict | None:
    """Resolve one span to a known company (S&P 500 or previously-tracked), else discover it via the LLM:
    subsidiary -> alias on its parent; new public company -> yfinance-validated profile; else name-only track.
    allow_discovery=False restricts to a direct lookup match (no LLM, no yfinance, no tracking)."""
    norm = _normalize(org).lower()
    if not norm or norm in ORG_STOPLIST:
        return None
    if norm in lookup:
        return {"ticker": lookup[norm]["ticker"], "name": lookup[norm]["name"], "confidence": 0.9}
    if not allow_discovery:
        return None   # non-ORG label (GPE/PERSON/...) that isn't a known company name — ignore it
    # Short all-caps spans not in the KB are almost always acronyms the small model mislabels as
    # ORG (IPO, CEO, GDP, AI...) rather than companies — skip them so we don't LLM-check or track them.
    if org.isupper() and len(org) <= 4:
        return None
    # Cheap shape filter: drop headline/sentence fragments and junk before spending an LLM call.
    if not _looks_like_company(org):
        return None

    if norm not in cache:                                  # per-run cache avoids repeat LLM calls
        cache[norm] = _check_subsidiary(org, lookup)
    result = cache[norm]

    if not result["is_company"]:                          # LLM says it isn't a real company — drop it
        return None

    if result["parent"]:                                  # subsidiary of a known company -> alias on parent
        parent = result["parent"]
        _add_alias(collections[parent["collection"]], parent, org)
        lookup[norm] = parent                             # match instantly for the rest of this run
        return {"ticker": parent["ticker"], "name": parent["name"], "confidence": 0.8}

    if result["ticker"]:                                  # its own ticker -> maybe promote
        tkey = result["ticker"].lower()
        if tkey in lookup:                                # already a known company (LLM gave an existing ticker)
            entry = lookup[tkey]
            lookup[norm] = entry
            return {"ticker": entry["ticker"], "name": entry["name"], "confidence": 0.8}
        profile = _fetch_yf_profile(result["ticker"])     # validate via yfinance before promoting
        if profile:
            # reuse an existing entry if yfinance's normalized ticker is already known, else promote
            entry = lookup.get(profile["ticker"].lower()) or _promote_company(
                collections[TRACKED_COMPANIES_COLLECTION], profile)
            lookup[norm] = entry                          # matchable by name + ticker from now on
            lookup[entry["ticker"].lower()] = entry
            return {"ticker": entry["ticker"], "name": entry["name"], "confidence": 0.8}

    _track_unknown(tracked_collection, org)               # real company, private/unlisted -> name-only track
    return None


def _match_by_text(article: dict, lookup: dict[str, dict]) -> list[dict]:
    """Fallback used when spaCy isn't available: regex-scan for known names/tickers (case-sensitive, S&P 500 only)."""
    text = article.get("title", "") + " " + article.get("text", article.get("summary", ""))
    matches, seen = [], set()
    for entry in {e["ticker"]: e for e in lookup.values()}.values():
        # match on the normalized name (drop "Inc."/etc — a trailing "." would break the \b anchor)
        norm_name = _normalize(entry["name"])
        variants = {norm_name, norm_name.upper()}
        if len(entry["ticker"]) >= 2:
            variants.add(entry["ticker"])
        pattern = r"\b(?:" + "|".join(re.escape(v) for v in variants) + r")\b"
        if re.search(pattern, text) and entry["ticker"] not in seen:
            seen.add(entry["ticker"])
            matches.append({"ticker": entry["ticker"], "name": entry["name"], "confidence": 0.7})
    return matches


def match_entities_to_articles(article_list: list[dict]) -> list[dict]:
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    sp500_collection = db.get_collection(name=SP500_COLLECTION)
    companies_collection = db.get_or_create_collection(name=TRACKED_COMPANIES_COLLECTION, embedding_function=_embedding_fn())
    tracked_collection = db.get_or_create_collection(name=TRACKED_COLLECTION, embedding_function=_embedding_fn())
    collections = {SP500_COLLECTION: sp500_collection, TRACKED_COMPANIES_COLLECTION: companies_collection}

    lookup: dict[str, dict] = {}
    _load_companies_into(lookup, sp500_collection)      # S&P 500 base KB
    _load_companies_into(lookup, companies_collection)  # companies discovered from past news runs
    nlp = _load_nlp()
    subsidiary_cache: dict[str, dict] = {}

    mode = "spaCy NER" if nlp else "text-match fallback"
    logger.info(f"Matching entities across {len(article_list)} articles (mode: {mode})...")

    for article in article_list:
        if nlp:
            raw_text = article.get("title", "") + ". " + article.get("text", article.get("summary", ""))
            text = re.sub(r"\s+", " ", raw_text).strip()   # collapse scraped whitespace/newlines/tables
            matches, seen = [], set()
            for org, is_primary in _extract_candidates(nlp, text):
                resolved = _resolve_org(org, lookup, collections, tracked_collection, subsidiary_cache,
                                        allow_discovery=is_primary)
                if resolved and resolved["ticker"] not in seen:
                    seen.add(resolved["ticker"])
                    matches.append({k: resolved[k] for k in ("ticker", "name", "confidence")})
        else:
            matches = _match_by_text(article, lookup)

        article["entities"] = matches
        if matches:
            logger.debug(f"[{article.get('title', '')[:50]}] matched {[m['ticker'] for m in matches]}")

    logger.info("Entity matching complete.")
    return article_list
