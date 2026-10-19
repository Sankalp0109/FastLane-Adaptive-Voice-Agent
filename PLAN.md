# FastLane — Implementation Plan

Source: `Chaturvedi_and_sons (1).pdf` (LMA Group Project 2026 proposal, team Chaturvedi and Sons).

FastLane is a voice agent for booking, changing and cancelling **movie tickets, bus tickets and
restaurant tables** for Indian-English callers. A LoRA-tuned **small model** (Qwen2.5-1.5B/3B)
handles every turn first and either replies (optionally with a tool call) or emits
`ESCALATE_LLM` / `ESCALATE_STAFF`. All actions go through one **tool and policy gateway**.

## Guiding decisions

| Decision | Choice | Why |
|---|---|---|
| Language / layout | Python ≥3.11, `src/fastlane` package, `pytest` | One codebase for world, agent, eval, training |
| Practice world | One SQLite DB, deterministic seed, fixed `TODAY = 2026-10-15 (Thu)` | Reproducible gold states; Dussehra/Diwali fall in the booking window |
| Config of systems | One `SystemConfig` dataclass with flags; FastLane, B1–B3, A1–A8 are presets | "Exactly one part removed" is enforced by construction |
| Model backends | `LLMClient` protocol: Anthropic (large), OpenAI-compatible / HF local (small), `ScriptedClient` (tests) | Runs offline in CI; swap models without touching agent code |
| Safety | Enforced in the gateway, not in prompts (except variant A3) | Hypothesis 5 |
| Timing | Every model/tool call is wrapped in a timer that records into the call trace | p50/p95 per turn comes for free from traces |

## Repository layout (target)

```
fastlane/
  pyproject.toml  README.md  PLAN.md
  src/fastlane/
    config.py              # TODAY, paths, SystemConfig + presets (FastLane, B1-B3, A1-A8)
    world/                 # Phase 1: practice booking system
      schema.sql  seed.py  db.py  names.py
    gateway/               # Phase 2: tools + policy enforcement
      tools.py  rules.py  dates.py  alternatives.py  record.py  gateway.py
    retrieval/             # Phase 3: RAG
      docs/  index.py  early_lookup.py  history.py
    models/                # Phase 4: LLM clients + prompts
      base.py  anthropic_client.py  small_client.py  scripted.py  prompts.py  parsing.py
    agent/                 # Phase 4: orchestrator + router
      orchestrator.py  router.py  templates.py  trace.py
    eval/                  # Phase 5: test calls, simulated caller, scoring
      scenarios.py  caller_sim.py  runner.py  metrics.py  report.py
    training/              # Phase 6: data + LoRA
      sgd.py  taskmaster.py  teacher.py  labels.py  noise.py  build_dataset.py  train_lora.py
    speech/                # Phase 7: voice pipeline
      pipeline.py  stt_compare.py  web/
  tests/
  data/                    # generated: world.db, test_calls.jsonl, training sets (git-ignored)
  notebooks/train_lora_colab.ipynb
```

## Phases

### Phase 0 — Scaffolding
- `pyproject.toml`, venv, `pytest`, `.gitignore`, `README.md`, `git init`.
- `config.py` with `TODAY`, paths, and the `SystemConfig` flag set.
- **Done when:** `pytest` runs (empty) and `python -m fastlane --help` works.

### Phase 1 — Practice world (SQLite)
- Schema: `customers`, `cinemas`, `screens`, `movies`, `shows`, `show_seats`, `bus_routes`,
  `bus_trips`, `bus_seats`, `restaurants`, `restaurant_tables`, `table_slots`, `bookings`,
  `booking_items`, `sms_log`, `handoffs`.
- Domains per Table 4: 4 cinemas × ~12 shows/day with seat maps; 6 bus routes, sleeper/seater,
  fares; 3 restaurant branches, tables by size/time.
- Deterministic seed (fixed RNG) with Indian names/cities, ~500 synthetic past bookings for
  returning callers, some sold-out shows/trips/slots planted for the "needs alternative" scenarios.
- `World` API: snapshot/restore (each test call starts from the same DB), `state_digest()` for gold
  comparison.
- **Done when:** `fastlane world build` creates `data/world.db`; tests check counts, determinism,
  sold-out plants, snapshot/restore.

### Phase 2 — Tool and policy gateway + booking record
- Tools (pydantic arg models → JSON schemas for tool calling): `search_inventory`, `resolve_date`,
  `find_alternatives`, `create_booking`, `modify_booking`, `cancel_booking`, `lookup_booking`,
  `get_caller_history`, `search_policy`, `send_sms`, `handoff_to_staff`.
- Rules in code: movie seat limit + cancellation cut-off; bus refund % by hours to departure +
  passenger details; restaurant party-size limits + deposit for large groups.
- Safety: read-back confirmation required in the previous turn for any change; identity check
  (phone match) for modify/cancel; cannot touch other callers' bookings; no payment tools;
  SMS logged only.
- `BookingRecord`: typed per-domain record (film/route/branch, date, time, seats/party, name,
  phone, status of read-back) shared by small model, large model and staff.
- Read-back templates; error injection hook (10% of calls) and one-retry policy.
- Variant hooks: `enforce_rules=False` (A3), `date_tools=False` (A2).
- **Done when:** unit tests cover each rule, read-back gating, identity check, date resolution
  ("next Friday", "day after tomorrow", "Diwali"), alternatives for sold-out inventory.

### Phase 3 — Retrieval (RAG) and caller history
- Write the ~60-page policy corpus as Markdown: fares, refund rules, FAQs, allergen-tagged menus,
  festival calendar, and one policy in **old and current** versions (with effective dates).
- Chunking + BM25 index (no heavy deps) with optional embedding reranker; "not covered" detection
  by score threshold; current-version preference.
- Early lookup: start search on partial transcripts, reuse if the final transcript matches
  (variant A6 disables it). Caller-history index over past bookings (A7 disables it).
- **Done when:** a small labelled policy-QA set gives Recall@3 / accuracy numbers; tests cover
  old-vs-new policy choice and "not covered".

### Phase 4 — Models, router and orchestrator
- `LLMClient` protocol with timing + token accounting. Backends: Anthropic (large, mid-tier model,
  prompt caching), OpenAI-compatible endpoint / HF transformers for Qwen + LoRA (small),
  `ScriptedClient` for deterministic tests.
- Small-model output format: reply + optional tool call, or `ESCALATE_LLM` / `ESCALATE_STAFF`,
  with an escalation score (from token logprob) and a threshold tuned on validation.
- Orchestrator: per-turn loop, hard-coded escalation triggers (tool error, no availability,
  changed earlier choice, unconfident policy match, score > threshold) and staff triggers
  (payment dispute, complaint, out-of-scope, two tool failures in a row).
- Large model gets booking record + last 2–3 turns + same tools, then returns control.
- Speed-ups: template read-backs/goodbyes, one retry, short context, cached prompts, streaming + filler.
- Systems: FastLane, B1 (large only, rules in prompt), B2 (untuned small only), B3 (large model as
  router), A1–A8 presets.
- **Done when:** a full scripted call runs end-to-end through every system preset with `ScriptedClient`,
  producing a trace with per-turn timings and route decisions.

### Phase 5 — Test set, simulated caller, evaluation
- Generate the 360 test calls (120/domain) with the category mix from §8: simple 120, dates 45,
  sold-out 45, changes-mind 30, policy 40, returning 40, red-flag/out-of-scope 25, rule-breaking 15.
  Each script: persona, goal, details revealed only when asked, hard turns marked, gold final DB state.
- LLM caller simulator (text), 3 seeds per call.
- Runner: systems × calls × seeds, resumable, JSONL traces; injected tool errors in 10% of calls.
- Metrics (Table 7): p50/p95 model time per turn, cost/tokens, large-model call rate, success (exact
  DB-state match), router precision/recall on hard turns, tool-use correctness, retrieval metrics,
  state correctness, safety (unsafe/unconfirmed bookings, missed hand-offs). 95% CIs.
- Report: success-vs-p95 plot per system + unsafe-action table.
- **Done when:** `fastlane eval --system fastlane --calls 10` produces a metrics JSON and plot.

### Phase 6 — Training data and LoRA
- Converters for Schema-Guided Dialogue (Movies/Buses/Restaurants) and Taskmaster-1
  (movie/restaurant) into our turn format, with easy/hard labels from dialogue acts.
- Teacher-generated practice calls on our world (keep only those reaching gold state).
- Rule-based labels (reply / tool call / ESCALATE_LLM / ESCALATE_STAFF); 300-turn double-label
  sheet + Cohen's κ script; speech-noise copies (~30%) via TTS→STT round trip or simulated ASR noise.
- QLoRA training script (PEFT/TRL) + Colab notebook; all-domain adapter and movies+buses adapter (A8).
- **Done when:** dataset builder outputs train/val JSONL; training script runs a smoke step on CPU
  with a tiny model.

### Phase 7 — Speech layer
- Pipecat pipeline: Silero VAD → streaming STT → FastLane orchestrator → sentence-streamed TTS,
  browser mic page; partial transcripts feed early lookup.
- STT comparison script (Sarvam Saaras, cloud en-IN, faster-whisper) on Svarah + own recordings:
  WER overall and on names/places/numbers/dates.
- **Done when:** local demo runs in a browser with a text fallback.

### Phase 8 — Experiments and report
- Run all 12 systems, collect metrics, produce figures/tables for the report; README results section.

## Status

| Phase | Status |
|---|---|
| 0 Scaffolding | in progress |
| 1 Practice world | todo |
| 2 Gateway + record | todo |
| 3 Retrieval | todo |
| 4 Models + orchestrator | todo |
| 5 Evaluation | todo |
| 6 Training | todo |
| 7 Speech | todo |
| 8 Experiments | todo |
