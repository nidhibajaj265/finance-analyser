# Finance Analyser — Claude Code Context

## What this project is
An Autonomous Financial News Sentiment Analyzer & Investment Signal Generator. Full spec in DESIGN.md.

## Collaboration mode
Pair-programming mode (switched from Socratic teaching on 2026-06-14): write code directly and iterate together rather than waiting for Nidhi to write it first. Still question everything — design choices, edge cases, naming, assumptions — like a critical pair programmer, not a silent implementer. Don't settle for the basic/obvious approach; push for the better one.

## Current status
Phases 0-3 COMPLETE. **Phase 4 — Entities** (next up — entity linking using the company-profile ChromaDB collection)

### Phases overview (from DESIGN.md section 14)
0. Setup → 1. Ingestion → 2. Sentiment standalone → 3. Knowledge base → 4. Entities → 5. Events → 6. Signals → 7. Wire LangGraph → 8. Conditional edges → 9. HITL → 10. Dashboard → 11. Backtest

## Resume instruction
At the start of a new session, read this file and the memory files, then ask Nidhi to paste whatever she was last working on so you can pick up from there.
