# FastLane

**A voice agent that uses a small model for easy turns and a large model only for hard ones.**

FastLane books, changes and cancels movie tickets, bus tickets and restaurant tables for Indian-English callers. A
fine-tuned 3B model (Qwen2.5-3B-Instruct with a LoRA adapter) answers every caller turn first. It passes only the hard
turns to a large API model (`openai/gpt-oss-120b` on Groq) or to a human, and every booking action goes through a tool
and policy gateway that enforces the booking rules and a spoken read-back in code.

LMA Group Project 2026, team **Chaturvedi and Sons** (IIIT Hyderabad).

| | |
|---|---|
| Final report | [report/final/fastlane_final_report.pdf](report/final/fastlane_final_report.pdf) |
| Trained adapters | [huggingface.co/Sankalp0109/fastlane-adapters](https://huggingface.co/Sankalp0109/fastlane-adapters) |
| Raw results of every run | [results/](results/) |
| Training notebook (Kaggle) | [notebooks/train_lora_kaggle.ipynb](notebooks/train_lora_kaggle.ipynb) |
| Poster | [poster/fastlane_poster.pdf](poster/fastlane_poster.pdf) |

## Contents

1. [How FastLane works](#how-fastlane-works)
2. [Final results](#final-results)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [Reproducing the final results](#reproducing-the-final-results)
8. [Training the small model](#training-the-small-model)
9. [How the evaluation works](#how-the-evaluation-works)
10. [The practice world and its rules](#the-practice-world-and-its-rules)
11. [Configuration](#configuration)
12. [Command reference](#command-reference)
13. [Troubleshooting](#troubleshooting)
14. [Team](#team)

## How FastLane works

A voice agent has about a second to reply before callers talk over it. Sending every turn to a large API model is
slow and expensive, yet most turns in a booking call are easy: a date, a seat count, a name, a "yes" to a read-back.
Only a few are hard: "next Friday", a sold-out show, a caller who changes their mind, a request that breaks a rule.
FastLane decides **for each turn** which model should reply.

```
caller speech ─► Silero VAD + Whisper / Sarvam STT ─► SMALL MODEL (Qwen2.5-3B + LoRA)
                                                        │  reply / tool call / pass on
                                     easy ◄─────────────┼─────────────► hard: LARGE MODEL (gpt-oss-120b)
                                       │                └─────────────► red flag: STAFF hand-off with a summary
                                       ▼                                         │
                        TOOL AND POLICY GATEWAY ◄──────────────────────────────────┘
        search · dates · alternatives · book / change / cancel · policy search · caller history
        rules in code · read-back gate · identity check · grounding checks · booking record
                                       │
                                       ▼
                         reply ─► Kokoro TTS (sentence by sentence)
```

- **One small model does two jobs.** For every turn it writes one JSON object: a route (`self`, `llm` or `staff`),
  the details heard this turn, an optional tool call and the words to say. The route, together with an escalation
  score from its token log-probabilities, decides whether the large model is needed. There is no separate router
  call, so an easy turn pays for only one small-model step.
- **Code also passes turns on.** A tool error, no availability, a changed earlier choice, an unsure policy match, a
  repeated tool call, or a reply that claims a booking change the gateway did not make all send the turn to the
  large model. Two failed tool calls in a row go to staff.
- **Neither model writes to the booking system directly.** Every create, change or cancel goes through the gateway.
  The gateway first returns a read-back that must be spoken to the caller, and the change only goes through after the
  caller confirms it in the next turn. Rules (ticket limits, refund windows, party sizes, deposits) are checked in
  code, not in the prompt.
- **Routine work is done in code.** Tools resolve dates ("next Friday", "Diwali") and find alternatives for sold-out
  shows. A typed booking record is shared by both models. Read-backs and goodbyes are templates that need no model.
- **The small model is trained on practice calls run through the real agent.** A rule-based caller and an oracle
  agent play each practice call through the real orchestrator and gateway. So the training inputs match what the model
  sees at inference exactly, and producing them costs nothing.

## Final results

Final runs on 10 November 2026, with one seed. Every system uses `openai/gpt-oss-120b` on Groq (paid tier, no backup
model) as the large model and the same simulated caller (Qwen2.5-7B). FastLane uses adapter v3. FastLane and B1 ran
on 120 stratified test calls, the other systems on 48 of these calls. The samples are nested and include the 24 calls
used during development. "FastLane on the same calls" is how many of the same calls FastLane completed, and the
paired test is an exact McNemar test.

| System | Calls | Success | FastLane on the same calls | Paired test p | Large-model calls per call | p50 / p95 per turn (s) | Calls with an unconfirmed booking change |
|---|---|---|---|---|---|---|---|
| **FastLane** | 120 | **108 (90.0%)** | – | – | 1.76 | 1.67 / 6.81 | **0** |
| B1 large model only | 120 | 98 (81.7%) | 108 | 0.087 | 8.02 | 1.04 / 3.01 | 44 |
| B2 untuned small model only | 48 | 14 (29.2%) | 46 | <0.001 | 0 | 1.65 / 16.08 | 0 |
| B3 large model as router | 48 | 44 (91.7%) | 46 | 0.50 | 6.94 | 1.66 / 7.66 | 0 |
| A1 no fine-tuning | 48 | 21 (43.8%) | 46 | <0.001 | 1.88 | 1.53 / 5.52 | 0 |
| A2 no date/alternative tools | 48 | 41 (85.4%) | 46 | 0.13 | 3.35 | 2.61 / 8.65 | 0 |
| A3 rules only in the prompt | 48 | 35 (72.9%) | 46 | 0.001 | 2.21 | 2.28 / 7.50 | 31 |
| A4 no booking record | 48 | 42 (87.5%) | 46 | 0.22 | 2.77 | 1.76 / 6.55 | 0 |
| A5 no reasoning on hard turns | 48 | 44 (91.7%) | 46 | 0.63 | 1.71 | 1.70 / 7.18 | 0 |
| A6 no early lookup | 48 | 43 (89.6%) | 46 | 0.25 | 1.69 | 1.68 / 7.11 | 0 |
| A7 no caller history | 48 | 45 (93.8%) | 46 | 1.00 | 2.33 | 1.81 / 7.03 | 0 |
| FastLane, adapter v5 (restaurants) | 40 | 37 (92.5%) | – | – | 0.90 | 1.60 / 7.39 | 0 |
| A8 movies+buses adapter v5 (restaurants) | 40 | 37 (92.5%) | 37 (v5) | 1.00 | 1.82 | 1.76 / 7.54 | 0 |

**What the results show:**

- **Safety:** with the rules and the read-back in code, FastLane made no booking change the caller had not confirmed.
  B1 made such changes in 44 of 120 calls, and A3 in 31 of 48 (paired tests p = 1.1e-13 and 9.3e-10). Many of these
  calls still reached the right final state, so success alone hides the risk.
- **Quality and cost:** FastLane succeeded in 8 more calls out of 120 than B1, which is not significant (p = 0.087).
  It made 78% fewer large-model calls and used 79% fewer large-model tokens.
- **Speed:** B1 is faster (p95 3.0 s vs 6.8 s per turn), because one step of the local 3B model takes longer than a
  call to gpt-oss-120b on Groq. The latency hypothesis does not hold on this hardware. Routing saves cost, quota and
  unsafe writes, not time.
- **Fine-tuning is essential:** without it (A1) success falls to 21 of 48. Offline, fine-tuning raises route accuracy
  from 89.1% to 99.3% and exact tool arguments from 26.8% to 99.0% on 267 held-out steps.
- **Tools in code halve large-model use:** without the date and alternative tools (A2), large-model calls per call
  double (1.67 → 3.35).
- **A new domain works:** an adapter trained only on movies and buses does as well on restaurants as the all-domains
  adapter (37 of 40 each), but with twice the large-model calls.
- **Failures:** we read every failed transcript. Half of FastLane's 12 failures come from the simulated caller not
  following its script ([results/final/failure_analysis.md](results/final/failure_analysis.md)). After four agent
  fixes, FastLane was rerun on the same 120 calls and got 105/120, which is within run-to-run variation
  ([results/final/final_fix/](results/final/final_fix/)).
- **Routing-label agreement:** Cohen's κ = 0.17 between two annotators, and 0.65 / 0.42 for each against the rule
  labels used for training ([results/kappa/agreement.md](results/kappa/agreement.md)). The disagreement is about
  design choices nobody wrote down; on plain bookings, dates and sold-out turns the annotators agree on 127 of 128
  turns.

## Repository layout

| Path | Contents |
|---|---|
| `src/fastlane/config.py` | Fixed "today" (2026-10-15), festival calendar, model roles, and the 12 system presets |
| `src/fastlane/world/` | SQLite schema, deterministic seeder, `World` (an in-memory copy for each call) |
| `src/fastlane/gateway/` | The 11 tools, rules in code, read-back gate, identity checks, date and alternative tools, booking record |
| `src/fastlane/retrieval/` | Policy documents (`docs/`, 25 files), BM25 index, early lookup, 67-question retrieval evaluation |
| `src/fastlane/models/` | Model clients (Groq, Gemini, Ollama and MLX through one OpenAI-compatible client), prompts, output parsing |
| `src/fastlane/agent/` | The orchestrator (per-turn routing), templates, traces |
| `src/fastlane/eval/` | Test-call generator, simulated caller, scorer, runner, metrics, reports |
| `src/fastlane/training/` | Practice-call generator, speech-noise copies, LoRA training (Kaggle/Colab/MLX), merging and conversion, offline step check, Cohen's κ |
| `src/fastlane/speech/` | Voice demo: browser page, WebSocket server, Silero VAD, speech recognition (MLX Whisper, faster-whisper, Sarvam), Kokoro TTS, WER comparison |
| `tests/` | 218 unit and flow tests (no model or network needed) |
| `notebooks/` | Kaggle and Colab training notebooks |
| `data/test_calls.jsonl` | The 360 scripted test calls (the practice world `data/world.db` is built locally) |
| `results/final/` | Final runs: per-call rows (`*.jsonl`), reports, run scripts, failure analysis, `final_tables.py` / `.json` |
| `results/kappa/` | Filled label sheet, κ values and the agreement analysis |
| `results/adapter_v1/` … `adapter_v5/` | Offline checks, loss curves and data reports for each adapter and data version |
| `results/pilot_*`, `results/mid_baselines/`, `results/v5_vs_v3_24calls/` | Development runs (pilot and 24-call runs) |
| `results/stt_comparison/`, `results/roundtrip_noise/`, `results/voice_demo/` | Speech recognition comparison, speech-noise data, a demo call |
| `report/final/` | Final report (LaTeX source, PDF, figure script and figures) |
| `poster/` | A1 poster (LaTeX source, PDF, figure script and figures) |

## Requirements

| What | Needed for | Notes |
|---|---|---|
| Python 3.11+ | everything | developed with Python 3.13 |
| Apple Silicon Mac (16 GB) | running the small model | the small models run as 4-bit MLX models (`mlx-lm`); developed on an M4 MacBook Air |
| [Ollama](https://ollama.com) with `qwen2.5:7b` | evaluation runs | plays the simulated caller |
| `GROQ_API_KEY` | anything that calls the large model | the final runs need Groq's paid tier; the free tier allows about 200k tokens a day, and one B1 call uses about 20k |
| `GOOGLE_API_KEY` | optional | Gemini as the backup large model (used during development only) |
| `SARVAM_API_KEY` | optional | Sarvam speech recognition or voices in the voice demo |
| `HF_TOKEN` | optional | the Svarah speech set (a gated dataset on Hugging Face) for the recogniser comparison |
| Kaggle account (free T4 GPU) | optional | only to train the adapters yourself |

Keys are read from the environment or from a `.env` file in the repository root. They are never printed or logged.

```bash
# .env (not committed)
GROQ_API_KEY=...
GOOGLE_API_KEY=...      # optional
SARVAM_API_KEY=...      # optional
HF_TOKEN=...            # optional
```

## Installation

```bash
git clone https://github.com/Sankalp0109/FastLane-Adaptive-Voice-Agent.git
cd FastLane-Adaptive-Voice-Agent
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,small,eval]"     # core, tests, model clients, plots
.venv/bin/pip install mlx-lm torch transformers peft   # small model: serving (MLX) and merging the adapters
.venv/bin/pip install -e ".[speech]"             # optional: voice demo and speech tools

.venv/bin/fastlane world build      # practice world -> data/world.db
.venv/bin/pytest -q                 # 218 tests in about 20 s, no model or network needed
```

## Quick start

**Without any model:** inspect the systems and query the policy index.

```bash
.venv/bin/fastlane systems
.venv/bin/fastlane docs search "refund if I cancel the bus 10 hours before"
```

The search returns the matching policy section with a confidence score. In this case it is the refund table of the
current bus refund policy, which gives a 25% refund for 4–12 hours before departure.

**With the models:** after the setup in steps 1 and 2 of [Reproducing the final results](#reproducing-the-final-results)
(adapters converted, `fastlane serve` running, `GROQ_API_KEY` set), talk to the agent in text:

```bash
.venv/bin/fastlane chat --system fastlane -v                      # interactive; -v shows routes and tool calls
.venv/bin/fastlane chat --system fastlane --say "I want two tickets for Kaala Samudram tomorrow evening" "yes"
.venv/bin/fastlane chat --system B1                               # any system: fastlane, B1-B3, A1-A8
```

**Voice demo** (needs the speech extras):

```bash
.venv/bin/fastlane voice                 # open http://127.0.0.1:8765, allow the microphone, and speak
.venv/bin/fastlane voice --stt sarvam    # Sarvam recogniser (better on Indian names; needs SARVAM_API_KEY)
```

The demo is half duplex: the caller's turn ends after 600 ms of silence. It also has a typed-text fallback.

## Reproducing the final results

**1. Get the small models.** The three trained LoRA adapters (125 MB each, too large for GitHub) are on the Hugging
Face Hub at [`Sankalp0109/fastlane-adapters`](https://huggingface.co/Sankalp0109/fastlane-adapters):

| Folder | Training data | Used for |
|---|---|---|
| `v3/` | all three domains, data v3 | FastLane in the final runs |
| `v5/` | all three domains, data v5 | FastLane v5, the matched pair for A8 |
| `mb-v5/` | movies and buses only, data v5 | ablation A8 |

`fastlane train finish` merges an adapter into `Qwen/Qwen2.5-3B-Instruct` (downloaded from Hugging Face on first use)
and converts the result to a 4-bit MLX model in `runs/`. A1 and B2 use the untuned base, converted the same way.

```bash
.venv/bin/hf download Sankalp0109/fastlane-adapters --local-dir adapters
.venv/bin/fastlane train finish adapters/v3 --name fastlane-qwen2.5-3b            # -> runs/mlx-fastlane-qwen2.5-3b
.venv/bin/fastlane train finish adapters/v5 --name fastlane-qwen2.5-3b-v5         # -> runs/mlx-fastlane-qwen2.5-3b-v5
.venv/bin/fastlane train finish adapters/mb-v5 --name fastlane-qwen2.5-3b-mb-v5   # -> runs/mlx-fastlane-qwen2.5-3b-mb-v5
.venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen2.5-3B-Instruct \
    --mlx-path runs/mlx-qwen2.5-3b-base -q --q-bits 4                              # untuned base (A1, B2)
```

**2. Start the servers.** Keep this running in its own terminal. One MLX server on port 8080 serves every small
model, and loads other model folders when they are requested.

```bash
.venv/bin/fastlane serve --model runs/mlx-fastlane-qwen2.5-3b
ollama pull qwen2.5:7b          # the simulated caller; Ollama must be running
```

**3. Run the experiments.** Runs can be resumed: rerunning a script skips calls already in
`data/runs/<run>/<system>.jsonl`. The final runs took about 5 hours on Groq's paid tier.

```bash
zsh results/final/run_final.sh        # FastLane, B1 (120 calls); B2, B3, A1-A7 (48); A8 pair (40) -> data/runs/final/, data/runs/final_v5/
zsh results/final/run_final_fix.sh    # FastLane with the current code on the same 120 calls -> data/runs/final_fix/
```

You can also run single systems:

```bash
.venv/bin/fastlane eval run --system fastlane B1 --run myrun --calls 120     # stratified sample of 120 calls
.venv/bin/fastlane eval run --system A2 --run myrun --calls 48 --domain buses
.venv/bin/fastlane eval run --system fastlane --run myrun --only mov-policy-085   # one call by id
```

**4. Tables, figures and the report.**

```bash
.venv/bin/fastlane eval report --run final           # REPORT.md, summary.json, plot in data/runs/final/
.venv/bin/python results/final/final_tables.py       # every number of the final tables -> results/final/final_tables.json
.venv/bin/fastlane train kappa results/kappa/label_sheet_filled.csv   # Cohen's kappa (0.17 / 0.65 / 0.42)
.venv/bin/python report/final/make_figures_final.py  # report/final/figures/
cd report/final && pdflatex fastlane_final_report && pdflatex fastlane_final_report
```

**What to expect.** `final_tables.py` and the κ command read the files stored in `results/`, so they reproduce the
reported numbers exactly without running anything. To score a fresh run, copy `data/runs/final/*.jsonl` into
`results/final/final/` first. A fresh run gives somewhat different numbers, because neither Groq nor the simulated
caller is deterministic: a rerun of FastLane changed the outcome of 13 of 120 calls. The `final` run used the code
from before the fixes of 10 November, so the current code reproduces the `final_fix` system exactly, not `final`.

## Training the small model

You only need this to train the adapters yourself. Each adapter takes about 3.3 hours on Kaggle's free T4.

**1. Generate the training data.** A rule-based caller and an oracle agent play 650 practice calls through the real
orchestrator and gateway. Every small-model step is saved as a pair: the exact input, and the ideal JSON output.
Calls that do not end in the planned final state are dropped (645 of 650 are kept). About 30% of steps also get a copy
with speech-recognition errors.

```bash
.venv/bin/fastlane train data                 # -> data/train/all_domains/ and data/train/movies_buses/
.venv/bin/fastlane train roundtrip            # optional (data v5): speak each caller line with Kokoro, transcribe it back
.venv/bin/fastlane train renoise              # optional (data v5): use those transcripts as the noise copies
```

**2. Train on Kaggle.** Settings: QLoRA with a 4-bit base, rank 16, alpha 32, dropout 0.05 on all seven projection
matrices, learning rate 2e-4 (cosine, 3% warm-up), effective batch 16, 2 epochs, sequence length 2,048. The loss
covers only the target JSON.

1. `.venv/bin/fastlane train colab-bundle` writes `data/train/colab_bundle.zip`.
2. Upload the zip as a Kaggle dataset.
3. Open [notebooks/train_lora_kaggle.ipynb](notebooks/train_lora_kaggle.ipynb), attach the dataset, choose a T4 GPU
   and run *Save & Run All*.
4. Download the `adapter/` folder from the notebook's output.

[notebooks/train_lora_colab.ipynb](notebooks/train_lora_colab.ipynb) does the same on Colab. On a Mac, you can train
with MLX directly instead, with `fastlane train mlx`; the final results do not use this option.

**3. Convert and check.**

```bash
.venv/bin/fastlane train finish path/to/adapter --name my-adapter                 # -> runs/mlx-my-adapter
.venv/bin/fastlane train check --model mlx:runs/mlx-my-adapter --out check.json  # offline step accuracy on held-out steps
```

The offline check replays held-out validation steps. It reports valid JSON, route accuracy, escalation precision and
recall, the right tool, and exact tool arguments. It needs no large model and no simulated caller.

## How the evaluation works

**The test set.** [data/test_calls.jsonl](data/test_calls.jsonl) holds 360 scripted calls, 120 per domain, none of
them used for training:

| Category | Calls |
|---|---|
| Simple book / change / cancel | 120 |
| Relative dates and festivals | 45 |
| Sold out (needs an alternative) | 45 |
| Policy questions | 40 |
| Returning callers | 40 |
| Change of mind | 30 |
| Red flags and out-of-scope requests | 25 |
| Rule-breaking requests (e.g. waiving a deposit) | 15 |

Each script gives a persona, a goal, an opening line, details revealed only when asked, the marked hard moments, and
the gold final state of the database. 10% of the calls get an injected tool failure. Qwen2.5-7B plays the caller,
following the script.

**Success.** A call succeeds when the bookings created or changed match an acceptable gold outcome exactly, and any
expected hand-off and policy answer happened. Success rates come with Wilson 95% intervals. Systems are compared with
FastLane on the same calls with an exact McNemar test.

**Other metrics.** Model time per turn (p50 / p95; time spent waiting on rate limits is excluded), share of turns and
calls sent to the large model, large-model tokens, router precision and recall on the marked hard turns, invalid tool
calls, success on calls with injected errors, unconfirmed and rule-breaking writes, missed hand-offs, and early-lookup
statistics.

**The 12 systems** (`fastlane systems`). Each baseline is another way to build the same agent. Each ablation is
FastLane with exactly one part removed or changed.

| System | Description |
|---|---|
| `fastlane` | Fine-tuned small model answers easy turns and passes hard ones to the large model or staff |
| `B1` | Large model only; rules in the prompt |
| `B2` | Small model only, no fine-tuning; large model never used |
| `B3` | Large model asked to route each turn as easy or hard before it is answered |
| `A1` | No fine-tuning; small model prompted |
| `A2` | No date and alternative tools |
| `A3` | No safety checks; rules only in the prompt, no read-back |
| `A4` | No booking record; models see the chat history only |
| `A5` | No step-by-step reasoning on hard turns |
| `A6` | No early lookup |
| `A7` | No caller history |
| `A8` | Adapter trained on movies and buses only; tested on restaurants |

## The practice world and its rules

One SQLite database is built from a fixed seed, with a fixed "now" of Thursday 15 October 2026, 11:00. Dussehra and
Diwali therefore fall inside the 30-day booking window.

- **Movies:** 4 cinemas × 3 screens × 4 shows a day, 9 films (one releasing on Diwali weekend), 1,440 shows with seat
  maps.
- **Buses:** Hyderabad to and from Bengaluru, Vijayawada and Chennai, 4 trips a day, sleeper and seater, 720 trips.
- **Restaurants:** 3 Tandoor Tales branches, 45 tables of 2–12, 90-minute seatings.
- **Callers:** 200 customers, 510 past bookings (for returning callers), 70 upcoming bookings to change or cancel.
- **Planted sell-outs** for the "needs an alternative" calls: *Kaala Samudram* Fri 16 Oct 20:30 everywhere,
  Friday-night Bengaluru sleepers, and Diwali dinner.
- **Policy documents:** 25 documents (about 21 pages) of fares, refund rules, FAQs, allergen-tagged menus, a festival
  calendar, and an old and a current bus refund policy. They are searched with BM25.

**Rules enforced in code:**

- **Limits:** at most 10 movie tickets, and changes up to 4 hours before the show. At most 6 bus seats, with a name,
  age and gender for every passenger.
- **Refunds:** bus refunds of 90 / 75 / 50 / 25% by hours to departure.
- **Restaurants:** party limits of 12, 10 and 8 by branch, and a deposit for large groups.
- **Read-back:** before any change, confirmed by the caller in the next turn.
- **Identity:** the booking's phone number is required to change it.
- **Grounding:**
  - the booked venue and time must have been said or offered in the call;
  - counts must be numbers the caller said;
  - a bus origin other than Hyderabad must have been named by the caller.

There is no payment tool, and SMS messages are only logged.

## Configuration

Which model plays each role is set in `MODEL_ROLES` in [src/fastlane/config.py](src/fastlane/config.py), as
`provider:model`. Each role can be overridden with an environment variable `FASTLANE_<ROLE>`:

| Role | Default | Variable |
|---|---|---|
| Large model | `groq:openai/gpt-oss-120b` | `FASTLANE_LARGE` |
| Backup large model | `gemini:gemini-3.5-flash` (`none` turns it off) | `FASTLANE_LARGE_BACKUP` |
| Tuned small model | `mlx:runs/mlx-fastlane-qwen2.5-3b` | `FASTLANE_SMALL_LORA` |
| Movies+buses small model (A8) | `mlx:runs/mlx-fastlane-qwen2.5-3b-mb` | `FASTLANE_SMALL_LORA_MB` |
| Untuned small model (A1, B2) | `mlx:runs/mlx-qwen2.5-3b-base` | `FASTLANE_SMALL` |
| Simulated caller | `ollama:qwen2.5:7b` | `FASTLANE_CALLER` |

Providers: `groq`, `gemini`, `ollama` and `mlx` (the `fastlane serve` server). For example, to run FastLane with the
v5 adapter and no backup model:

```bash
FASTLANE_SMALL_LORA=mlx:runs/mlx-fastlane-qwen2.5-3b-v5 FASTLANE_LARGE_BACKUP=none \
  .venv/bin/fastlane eval run --system fastlane --run v5test --calls 24
```

## Command reference

| Command | What it does |
|---|---|
| `fastlane systems` | List FastLane, baselines B1–B3 and ablations A1–A8 |
| `fastlane world build` | Build the deterministic SQLite practice world (`data/world.db`) |
| `fastlane docs build` | Regenerate the policy documents that are generated from the world's data |
| `fastlane docs search "QUERY" [--domain D]` | Search the policy index |
| `fastlane testset build` | Regenerate the 360 test calls (`data/test_calls.jsonl`) |
| `fastlane chat --system S [-v] [--say "..." ...]` | Talk to a system in text (interactive, or a scripted list of caller lines) |
| `fastlane voice [--stt mlx\|faster-whisper\|sarvam] [--tts kokoro\|sarvam]` | Browser voice demo at http://127.0.0.1:8765 |
| `fastlane serve [--model PATH] [--port 8080]` | Serve the small models (MLX, OpenAI-compatible, with log-probabilities) |
| `fastlane eval run --system S ... --run NAME [--calls N] [--domain D] [--category C] [--only ID ...]` | Simulated calls on a stratified sample; resumable; results in `data/runs/NAME/` |
| `fastlane eval report --run NAME` | `REPORT.md`, `summary.json` and the success-vs-p95 plot for a run |
| `fastlane train data` | Generate practice-call training data (`data/train/`) |
| `fastlane train roundtrip` / `renoise` | Speech-noise copies from a TTS → STT round trip (data v5) |
| `fastlane train colab-bundle` | Zip the training data and script for the Kaggle/Colab notebook |
| `fastlane train mlx` | LoRA training on Apple Silicon with MLX |
| `fastlane train finish ADAPTER --name NAME` | Merge an adapter and convert it to a 4-bit MLX model (`--ollama` for Ollama instead) |
| `fastlane train check --model M` | Offline step accuracy on held-out validation steps |
| `fastlane train label-sheet` / `kappa SHEET` | Export 300 turns for two annotators / compute Cohen's κ |
| `fastlane speech wer --set tts\|svarah` | Compare recognisers: WER and key-detail accuracy |

Every command has `--help`.

## Troubleshooting

- **Success looks like A1 (about 45%):** if the MLX server is not running, or a model folder in `runs/` is missing,
  FastLane quietly falls back to the untuned small model, and then to Ollama's `qwen2.5:3b`. Before a run, check that
  `curl -s http://localhost:8080/v1/models` answers and that the `runs/mlx-…` folders from step 1 exist.
- **Groq rate limits:** on the free tier the client waits as Groq's response headers ask, and the wait is logged
  separately from model time. A run may then switch to the Gemini backup if `GOOGLE_API_KEY` is set. For runs
  comparable with the report, use the paid tier and `FASTLANE_LARGE_BACKUP=none`.
- **MLX fails with an MPI error under Anaconda:** FastLane sets `MLX_MPI_LIBNAME` for its own MLX calls. If you call
  `mlx_lm` yourself, run `export MLX_MPI_LIBNAME=/nonexistent/libmpi.dylib` first.
- **The simulated caller does not answer:** check that Ollama is running and `qwen2.5:7b` has been pulled.
- **Svarah download fails:** accept the dataset's terms on Hugging Face and put `HF_TOKEN` in `.env`.

## Team

Noel Alex Jacob, Aviral Tyagi, Sankalp Chaturvedi and KSPSVLN Siddardha Kumar Kavuri (IIIT Hyderabad). LMA Group
Project 2026.
