#!/bin/zsh
# Mid-submission runs: baselines and ablations on the 24 pilot calls (one seed), run name "mid".
cd "$(dirname "$0")/../.."   # repo root
IDS=(bus-change_mind-191 bus-dates-161 bus-policy-201 bus-red_flag-228 bus-returning-214 bus-rule_break-236 bus-simple-121 bus-sold_out-176 mov-change_mind-071 mov-dates-041 mov-policy-081 mov-red_flag-108 mov-returning-095 mov-rule_break-116 mov-simple-001 mov-sold_out-056 res-change_mind-311 res-dates-281 res-policy-321 res-red_flag-347 res-returning-334 res-rule_break-356 res-simple-241 res-sold_out-296)
RES=(res-change_mind-311 res-dates-281 res-policy-321 res-red_flag-347 res-returning-334 res-rule_break-356 res-simple-241 res-sold_out-296)
# Cheapest first (mostly local small model), large-model-heavy baselines last.
for sys in fastlane B2 A3 A2; do
  echo "=== $sys $(date +%H:%M:%S)"
  .venv/bin/fastlane eval run --system $sys --run mid --only $IDS
done
echo "=== A8 (restaurant calls only) $(date +%H:%M:%S)"
.venv/bin/fastlane eval run --system A8 --run mid --only $RES
for sys in B3 B1; do
  echo "=== $sys $(date +%H:%M:%S)"
  .venv/bin/fastlane eval run --system $sys --run mid --only $IDS
done
echo "=== report $(date +%H:%M:%S)"
.venv/bin/fastlane eval report --run mid
echo "=== all done $(date +%H:%M:%S)"
