#!/bin/zsh
# Final runs (2026-11-10), Groq paid tier. One seed, current code (sections 21 and 24 fixes), adapter v3 for FastLane.
# Large model: groq:openai/gpt-oss-120b ONLY (Gemini backup off, so every system sees the same large model).
# Samples are stratified and nested: --calls 48 is a subset of --calls 120; the 24 development calls are in both.
# Resumable: rerunning the script skips calls already in data/runs/<run>/<system>.jsonl.
cd /Users/sankakpchaturvedi/IIITH/assignments/LMA/LMA_GROUP_PROJECT
export FASTLANE_LARGE_BACKUP=none
FL=.venv/bin/fastlane
stamp() { date '+%m-%d %H:%M:%S'; }

# 1. Core comparison (H1, H2): FastLane and B1 on 120 calls.
for sys in fastlane B1; do
  echo "=== final $sys 120 $(stamp)"; $FL eval run --system $sys --run final --calls 120
done
# 2. Other baselines and main ablations on 48 calls.
for sys in B3 B2 A1 A2 A3; do
  echo "=== final $sys 48 $(stamp)"; $FL eval run --system $sys --run final --calls 48
done
$FL eval report --run final

# 3. A8 matched pair (same data version v5): all-domains v5 vs movies+buses v5 on 40 restaurant calls.
echo "=== final_v5 fastlane(v5) 40 restaurants $(stamp)"
FASTLANE_SMALL_LORA=mlx:runs/mlx-fastlane-qwen2.5-3b-v5 \
  $FL eval run --system fastlane --run final_v5 --domain restaurants --calls 40
echo "=== final_v5 A8(mb-v5) 40 restaurants $(stamp)"
FASTLANE_SMALL_LORA_MB=mlx:runs/mlx-fastlane-qwen2.5-3b-mb-v5 \
  $FL eval run --system A8 --run final_v5 --domain restaurants --calls 40
$FL eval report --run final_v5

# 4. Remaining ablations (A4-A7) on 48 calls.
for sys in A4 A5 A6 A7; do
  echo "=== final $sys 48 $(stamp)"; $FL eval run --system $sys --run final --calls 48
done
$FL eval report --run final
echo "=== all done $(stamp)"
