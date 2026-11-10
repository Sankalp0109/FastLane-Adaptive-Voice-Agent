# FastLane

A voice agent that books, changes and cancels movie tickets, bus tickets and restaurant tables for
Indian-English callers. A fine-tuned small model answers easy turns and passes hard ones to a large
model or to staff. LMA Group Project 2026, team Chaturvedi and Sons. See [PLAN.md](PLAN.md) for the
phase plan and status, and [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for every experiment, failure and fix
(with raw numbers in [results/](results/)). The final report is
[report/final/fastlane_final_report.pdf](report/final/fastlane_final_report.pdf).

## Final results (10 November 2026)

One seed; every system uses `openai/gpt-oss-120b` on Groq (paid tier, no backup model) as the large model and the
same simulated caller (Qwen2.5-7B). FastLane uses adapter v3. FastLane and B1 ran on 120 stratified test calls, the
other systems on 48 of these calls; the samples are nested and include the 24 calls used during development.

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

- **Safety:** with rules and read-back in code, FastLane made no booking change the caller had not confirmed. B1 made
  such changes in 44 of 120 calls and A3 in 31 of 48 (paired tests p = 1.1e-13 and 9.3e-10).
- **Quality and cost:** FastLane succeeded in 8 more calls out of 120 than B1, which is not significant (p = 0.087).
  It used 78% fewer large-model calls.
- **Speed:** B1 is faster (p95 3.0 s vs 6.8 s per turn), because one step of the local 3B model takes longer than a
  call to gpt-oss-120b on Groq. The latency hypothesis does not hold on this hardware.
- **Failures:** half of FastLane's 12 failures come from the simulated caller (`results/final/failure_analysis.md`).
  After four agent fixes, FastLane was rerun on the same 120 calls and got 105/120, which is within run-to-run
  variation (`results/final/final_fix/`).
- **Routing-label agreement:** Cohen's κ = 0.17 between two annotators and 0.65 / 0.42 against the rule labels
  (`results/kappa/`).

Raw rows, reports and the matched comparison are in `results/final/` (`final/`, `final_v5/`, `final_fix/`,
`comparison_matched.md`, `final_tables.json`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,small,eval]"   # add ",speech" for the voice demo
.venv/bin/fastlane world build      # practice world -> data/world.db
.venv/bin/pytest -q                 # 218 tests, no model or network needed
```

## Reproducing the final results

You need an Apple Silicon Mac (the small model runs with MLX, so `mlx-lm` must be installed in `.venv`), Ollama with
`qwen2.5:7b` (the simulated caller), and `GROQ_API_KEY` in `.env`. On the Groq free tier, the final runs hit the
daily token limit.

```bash
# 1. Small models as 4-bit MLX models (adapters from the Kaggle notebook; base Qwen/Qwen2.5-3B-Instruct)
.venv/bin/fastlane train finish kaggle_output_4/fastlane-qwen2.5-3b/adapter --name fastlane-qwen2.5-3b        # v3 -> runs/mlx-fastlane-qwen2.5-3b
.venv/bin/fastlane train finish kaggle_output_6/fastlane-qwen2.5-3b/adapter --name fastlane-qwen2.5-3b-v5     # v5
.venv/bin/fastlane train finish kaggle_output_6/fastlane-qwen2.5-3b-mb/adapter --name fastlane-qwen2.5-3b-mb-v5  # A8
#    The untuned base used by A1 and B2 is runs/mlx-qwen2.5-3b-base (Qwen2.5-3B-Instruct converted to 4-bit MLX).

# 2. Serve the small models (port 8080; other model folders are loaded on request)
.venv/bin/fastlane serve --model runs/mlx-fastlane-qwen2.5-3b

# 3. Final runs (resumable; about 5 hours on the paid tier), then the after-fix run
ollama pull qwen2.5:7b
zsh results/final/run_final.sh        # writes data/runs/final/ and data/runs/final_v5/
zsh results/final/run_final_fix.sh    # needs the code from 10 Nov (current code); writes data/runs/final_fix/

# 4. Tables, figures and the report
.venv/bin/fastlane eval report --run final
.venv/bin/python results/final/final_tables.py
.venv/bin/python report/final/make_figures_final.py
cd report/final && pdflatex fastlane_final_report && pdflatex fastlane_final_report
```

`results/final/final_tables.py` reads the copies stored in `results/final/`. To use a fresh run, copy
`data/runs/final/*.jsonl` there first. A fresh run gives different numbers: Groq and the simulated caller are not
deterministic, and a rerun of FastLane changed the outcome of 13 of 120 calls. The `final` run used the code before
the fixes of 10 November, so with the current code only `final_fix` reproduces exactly that system.

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
| `fastlane eval run --system fastlane B1 --run final --calls 120` | Simulated calls on a stratified sample (`--domain`, `--only ID ...` to filter), resumable, results in `data/runs/<run>/` |
| `fastlane eval report --run final` | `REPORT.md`, `summary.json` and the success-vs-p95 plot |
| `fastlane serve --model runs/mlx-fastlane-qwen2.5-3b` | Serve the small models (MLX, port 8080) |

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
