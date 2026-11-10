#!/bin/zsh
# Offline checks of the v5 adapters: one MLX server per model on :8080, then `fastlane train check`.
cd /Users/sankakpchaturvedi/IIITH/assignments/LMA/LMA_GROUP_PROJECT
R=results/adapter_v5; F=.venv/bin/fastlane
serve() {
  pkill -f "mlx_lm server" ; sleep 2
  $F serve --model $1 > data/runs/serve_$2.log 2>&1 &
  until curl -s localhost:8080/v1/models >/dev/null; do sleep 2; done
  echo "=== server up: $1 $(date +%T)"
}
serve runs/mlx-fastlane-qwen2.5-3b-v5 v5
echo "=== v5 clean $(date +%T)"
$F train check --model mlx:runs/mlx-fastlane-qwen2.5-3b-v5 --val data/train/all_domains/val.jsonl --out $R/check_v5_val_clean.json --save-rows $R/rows_v5_val_clean.jsonl || echo "FAILED v5 clean"
echo "=== v5 roundtrip $(date +%T)"
$F train check --model mlx:runs/mlx-fastlane-qwen2.5-3b-v5 --val data/train/all_domains/val_roundtrip.jsonl --out $R/check_v5_val_roundtrip.json --save-rows $R/rows_v5_val_roundtrip.jsonl || echo "FAILED v5 roundtrip"
serve runs/mlx-fastlane-qwen2.5-3b-mb-v5 mbv5
echo "=== mb-v5 restaurants $(date +%T)"
$F train check --model mlx:runs/mlx-fastlane-qwen2.5-3b-mb-v5 --val data/train/all_domains/val.jsonl --domain restaurants --out $R/check_mb_v5_on_restaurants.json --save-rows $R/rows_mb_v5_on_restaurants.jsonl || echo "FAILED mb-v5"
serve runs/mlx-fastlane-qwen2.5-3b-mb mbv2
echo "=== mb-v2 restaurants $(date +%T)"
$F train check --model mlx:runs/mlx-fastlane-qwen2.5-3b-mb --val data/train/all_domains/val.jsonl --domain restaurants --out $R/check_mb_v2_on_v5val_restaurants.json --save-rows $R/rows_mb_v2_on_v5val_restaurants.jsonl || echo "FAILED mb-v2"
pkill -f "mlx_lm server"
echo "=== done $(date +%T)"
