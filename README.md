# FastLane

A voice agent that books, changes and cancels movie tickets, bus tickets and restaurant tables for
Indian-English callers. A fine-tuned small model answers easy turns and passes hard ones to a large
model or to staff. LMA Group Project 2026, team Chaturvedi and Sons. See [PLAN.md](PLAN.md) for the
phase plan and status, and [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for every experiment, failure and fix
(with raw numbers in [results/](results/)).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/fastlane world build      # practice world -> data/world.db
.venv/bin/pytest -q
```

## Commands

| Command | What it does |
|---|---|
| `fastlane systems` | List FastLane, baselines B1-B3 and variants A1-A8 |
| `fastlane world build` | Build the deterministic SQLite practice world |
| `fastlane docs build` | Regenerate the data-backed policy documents |
| `fastlane docs search "refund if I cancel the bus 10 hours before"` | Query the policy index |
| `fastlane chat --system fastlane -v` | Talk to a system in text (`--say "..." "..."` for a scripted run) |
| `fastlane voice` | Browser voice demo at http://127.0.0.1:8765 (mic → VAD → Whisper → agent → Kokoro TTS, text fallback); needs `pip install -e ".[speech]"`, `fastlane serve` running, and the Groq key |
| `fastlane testset build` | Regenerate the 360 test calls (`data/test_calls.jsonl`) |
| `fastlane eval run --system fastlane B1 --run pilot --calls 24` | Simulated calls, resumable, results in `data/runs/<run>/` |
| `fastlane eval report --run pilot` | `REPORT.md`, `summary.json` and the success-vs-p95 plot |

## Training the small model (LoRA)

| Step | Command / place |
|---|---|
| Build training data | `fastlane train data` (practice calls through the real agent; `data/train/`) |
| **Kaggle (3B, recommended)** | `fastlane train colab-bundle`, upload `data/train/colab_bundle.zip` as a Kaggle dataset, run [notebooks/train_lora_kaggle.ipynb](notebooks/train_lora_kaggle.ipynb) with *Save & Run All* |
| Colab | [notebooks/train_lora_colab.ipynb](notebooks/train_lora_colab.ipynb) |
| Apple Silicon (MLX) | `fastlane train mlx --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --iters 1861` |
| Use the result | `fastlane train ollama-import fastlane-qwen2.5-3b.Q4_K_M.gguf --name fastlane-qwen2.5-3b` |
| Routing-label agreement | `fastlane train label-sheet`, two annotators fill it, `fastlane train kappa data/train/label_sheet.csv` |

Models are set in `src/fastlane/config.py` (`MODEL_ROLES`) or with `FASTLANE_LARGE=groq:openai/gpt-oss-120b` style
environment variables. Keys are read from `.env` (`GROQ_API_KEY`, `GOOGLE_API_KEY`); local models run on Ollama.

## Layout

| Path | Contents |
|---|---|
| `src/fastlane/config.py` | Fixed "today" (2026-10-15), festival calendar, system presets |
| `src/fastlane/world/` | SQLite schema, deterministic seeder, `World` (per-call in-memory copies) |
| `src/fastlane/gateway/` | Tools, rules in code, read-back gate, identity checks, date and alternative tools, booking record |
| `src/fastlane/retrieval/` | Policy documents, BM25 index, early lookup, retrieval evaluation |
| `src/fastlane/models/` | Model clients (Groq/Gemini/Ollama via one OpenAI-compatible client), prompts, output parsing |
| `src/fastlane/agent/` | The orchestrator (per-turn routing), templates, traces |
| `src/fastlane/eval/` | Test-call generator, simulated caller, scorer, runner, metrics, report |
| `tests/` | Unit and flow tests |

## The practice world

- **Movies:** 4 cinemas × 3 screens × 4 shows/day, 9 films (one releasing on Diwali weekend).
- **Buses:** Hyderabad–Bengaluru/Vijayawada/Chennai in both directions, 4 trips/day, sleeper and seater.
- **Restaurants:** 3 Tandoor Tales branches, tables of 2–12, 90-minute seatings.
- 200 customers, 510 past bookings for returning callers, 70 upcoming bookings to change or cancel.
- Planted sell-outs (e.g. *Kaala Samudram* Fri 16 Oct 20:30 everywhere; Friday-night Bengaluru sleepers;
  Diwali dinner) for the "needs an alternative" test calls.
