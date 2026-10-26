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
| 0 Scaffolding | done |
| 1 Practice world | done — `fastlane world build`, 12 tests |
| 2 Gateway + record | done — 11 tools, rules, read-back gate, identity, error injection; 78 tests total |
| 3 Retrieval | done — 25 docs (~21 pages at 450 words; target ~60, add `.md` files to `retrieval/docs/`), BM25 + version handling + "not covered", early lookup; 87 tests total |
| 4 Models + orchestrator | done — Groq `gpt-oss-120b` (large), Ollama `qwen2.5:3b` (small, base until LoRA), all 12 systems run; `fastlane chat`; 119 tests total |
| 5 Evaluation | done — 360-call test set (`data/test_calls.jsonl`, every gold outcome verified achievable), LLM caller (Qwen2.5-7B), resumable runner, Table 7 metrics, report + plot; 135 tests total. Free-tier pilot done (below) |
| 6 Training | v1 and v2 trained on Kaggle and served with MLX; v2 offline: route 98.9%, escalation recall 92.3%, args 92.5% (FastLane config 98.4%); A8 transfers routing but not restaurant arguments; **data v3 (A2 date fix) prepared, not trained** (see docs/EXPERIMENT_LOG.md §8) |
| 7 Speech | todo |
| 8 Experiments | todo |

### Phase 4 findings (first real calls)

- Small model (base Qwen2.5-3B, Ollama, M4): 0.8–1.5 s per step with a ~1k-token prompt; 5 s cold start.
  Log-probabilities are available, so the escalation score is real. The untuned model over-escalates
  easy turns and invents option ids, which is the gap LoRA (Phase 6) has to close (A1 vs FastLane).
- Large model (Groq free tier): ~1–2.5 s per call, but the free tier allows 8k tokens/min and 1k requests/day,
  and each large call carries ~3k tokens of tool schemas + prompt. Waiting is logged as `throttle_ms` and is
  not counted as model time. Free tier is enough for development and pilots, not for 13k simulated calls.
- Staff hand-off on "two failed tool calls in a row" counts booking-system failures only; model mistakes
  (bad arguments, unknown tools) escalate to the large model instead.

### Phase 3 retrieval baseline (67-question QA set, `fastlane.retrieval.evaluate`)

| Recall@3 | Top-1 | Answer in context | Covered when answerable | Current-policy choice | "Not covered" detection |
|---|---|---|---|---|---|
| 0.967 | 0.817 | 0.883 | 0.933 | 1.00 | 0.714 (only 7 items; noisy) |

Thresholds (`min_score=5.0`, `min_coverage=0.6`) were tuned on this same set, so these numbers are optimistic.
A held-out QA set and an optional dense reranker are follow-ups.

### Phase 5 notes

- Success = the set of bookings created or changed anywhere in the database matches one acceptable gold outcome
  exactly (domain-specific fields; bus passengers by first name), plus hand-off and policy-answer expectations.
- Hard-turn labels for router metrics: caller-text patterns + the simulated caller's own tags + any no-availability
  result in the turn (`fastlane.eval.labels`). The same labeller will produce routing targets for training data.
- Bugs found by the first real calls and fixed: Groq rejects tool calls server-side when an enum doesn't match
  (now lenient + normalised, and rejections are fed back to the model); the large model lost details across turns
  (write-tool arguments now fill the booking record); "Friday 23rd October" resolved to the coming Friday
  (explicit dates now win); the base small model sent refund-rule questions to staff (prompt clarified).

### Pilot (24 stratified calls, FastLane with the *untuned* Qwen2.5-3B, Groq gpt-oss-120b, Qwen2.5-7B caller)

| Success (95% CI) | p50 / p95 model ms per turn | Large-model turns | Router precision / hard recall | Unsafe-write calls | Missed hand-offs |
|---|---|---|---|---|---|
| 79.2% (60–91%) | 2,882 / 6,892 | 60.7% | 25.3% / 90.5% | 0% | 0/3 |

- Every category succeeded except policy (0/3: the untuned small model hands refund-rule questions to staff)
  and one sold-out and one change-of-mind call. Injected tool errors: 3/3 recovered.
- ~6 large-model calls and ~13k large-model tokens per call; ~110 s rate-limit wait per call on Groq's free tier.
- Large-model fallback: Groq `gpt-oss-120b` first; when Groq would make a call wait more than 10 s, that call goes to
  Gemini `gemini-3.5-flash` (4–7 s per call vs 1–2 s on Groq). `large_model_mix` in the report shows which model
  answered, so any result can be split by model. Set `FASTLANE_LARGE_BACKUP=none` to disable.

### Phase 6 notes

- **Practice-call data** (`fastlane train data`): both sides are code (rule-based caller + oracle agent) and the
  calls run through the real orchestrator and gateway, so every example is the exact small-model input plus the
  ideal JSON. Only calls that end in the planned final state are kept (645/650).
- **Routing targets:** relative dates and policy questions are handled by the small model *when it has the tool*
  (A2 prompts without `resolve_date` -> `llm`); changed choices, "my usual", refused details -> `llm`; red flags,
  deposit waivers, over-limit groups -> `staff`; no availability / tool errors escalate in code.
- **Mix:** prompts of FastLane 60%, A2 15%, A4 10%, A6 5%, A7 10% so one adapter works with every tool list/context.
  Escalation steps are repeated 3x in train (they are ~5% of steps); validation keeps the natural mix.
- **Speech noise:** 30% of training examples get a copy with mis-transcribed caller lines (rule-based; replace with
  a TTS->STT round trip in Phase 7).
- **Compact prompt:** the tuned model gets a 643-token system prompt instead of 1,062 (less prefill latency);
  used only when the tuned model is really loaded.
- **Not done yet:** Schema-Guided Dialogue / Taskmaster-1 conversion (public data rows of Table 5);
  the 300-turn double annotation (`fastlane train label-sheet`, then `fastlane train kappa`) needs two people.
- **Bugs found while building the data, fixed in the agent:** `parse_time("10:00")` returned 22:00 (24-hour tool
  times are now literal); "…44, male, that's all" was taken as a goodbye (goodbyes now need a short turn without
  numbers).

### Adapter v1 results (offline step accuracy, `fastlane train check`)

| Same 80 held-out steps | Untuned Qwen2.5-3B (full prompt) | Tuned v1 (compact prompt) |
|---|---|---|
| Route accuracy | 62.5% | 97.5% |
| Escalation precision | 15% | 100% |
| Right tool | 51% | 100% |
| Exact tool arguments | 25% | 87% |
| p50 / p95 latency | 2.7 s / 9.3 s (Ollama) | 2.1 s / 3.8 s (MLX) |

On all 273 validation steps: route 98.5%, tool 99%, arguments 88%. Every remaining error involved dates:
a date said several turns earlier had scrolled out of the 6-line context and was never stored in the booking
record; A2 date turns were not escalated; and `resolve_date` read "Friday the 23rd" as the coming Friday.
Fixed for v2: spoken dates go in the `date` slot and are resolved into the record by code, searches accept the
caller's words, A2 escalates any spoken date, and day-of-month beats the weekday word.

**Serving:** this Ollama build cannot import fine-tuned Qwen2 weights or LoRA adapters, so the small models
(tuned and untuned) are 4-bit MLX models served by `fastlane serve`; `fastlane train finish` merges a downloaded
adapter and converts it.
