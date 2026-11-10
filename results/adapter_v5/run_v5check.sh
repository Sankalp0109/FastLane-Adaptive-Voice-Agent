#!/bin/zsh
# 24-call end-to-end check of adapter v5 vs v3, both with the current code (section 21 fixes).
# One MLX server per model on :8080. Runs: data/runs/v5check (v5) and data/runs/v3check (v3).
cd /Users/sankakpchaturvedi/IIITH/assignments/LMA/LMA_GROUP_PROJECT
IDS=(bus-change_mind-191 bus-dates-161 bus-policy-201 bus-red_flag-228 bus-returning-214 bus-rule_break-236 bus-simple-121 bus-sold_out-176 mov-change_mind-071 mov-dates-041 mov-policy-081 mov-red_flag-108 mov-returning-095 mov-rule_break-116 mov-simple-001 mov-sold_out-056 res-change_mind-311 res-dates-281 res-policy-321 res-red_flag-347 res-returning-334 res-rule_break-356 res-simple-241 res-sold_out-296)
RES=(res-change_mind-311 res-dates-281 res-policy-321 res-red_flag-347 res-returning-334 res-rule_break-356 res-simple-241 res-sold_out-296)
F=.venv/bin/fastlane
serve() {
  pkill -f "mlx_lm server"; sleep 2
  $F serve --model $1 > data/runs/serve_check.log 2>&1 &
  until curl -s localhost:8080/v1/models >/dev/null; do sleep 2; done
  echo "=== server up: $1 $(date '+%m-%d %T')"
}
serve runs/mlx-fastlane-qwen2.5-3b-v5
echo "=== fastlane v5 $(date '+%m-%d %T')"
FASTLANE_SMALL_LORA=mlx:runs/mlx-fastlane-qwen2.5-3b-v5 $F eval run --system fastlane --run v5check --only $IDS
serve runs/mlx-fastlane-qwen2.5-3b-mb-v5
echo "=== A8 mb-v5 $(date '+%m-%d %T')"
FASTLANE_SMALL_LORA_MB=mlx:runs/mlx-fastlane-qwen2.5-3b-mb-v5 $F eval run --system A8 --run v5check --only $RES
serve runs/mlx-fastlane-qwen2.5-3b
echo "=== fastlane v3 $(date '+%m-%d %T')"
$F eval run --system fastlane --run v3check --only $IDS
$F eval report --run v5check; $F eval report --run v3check
echo "=== done $(date '+%m-%d %T')"
