#!/bin/zsh
# B1, A3 (rerun: ungated tool descriptions) and B3 on the same 24 calls as the other mid runs (run name "mid"), resumable.
cd "$(dirname "$0")/../.."   # repo root
IDS=(bus-change_mind-191 bus-dates-161 bus-policy-201 bus-red_flag-228 bus-returning-214 bus-rule_break-236 bus-simple-121 bus-sold_out-176 mov-change_mind-071 mov-dates-041 mov-policy-081 mov-red_flag-108 mov-returning-095 mov-rule_break-116 mov-simple-001 mov-sold_out-056 res-change_mind-311 res-dates-281 res-policy-321 res-red_flag-347 res-returning-334 res-rule_break-356 res-simple-241 res-sold_out-296)
for sys in B1 A3 B3; do
  echo "=== $sys $(date '+%m-%d %H:%M:%S')"
  .venv/bin/fastlane eval run --system $sys --run mid --only $IDS
done
.venv/bin/fastlane eval report --run mid
echo "=== done $(date '+%m-%d %H:%M:%S')"
