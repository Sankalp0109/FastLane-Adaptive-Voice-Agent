#!/bin/zsh
# After-fix run (2026-11-10): FastLane (adapter v3) on the same 120 calls as run `final`, with the three fixes of
# section 27 (origin must be heard; completion claim without a write; negative policy answer). Groq only.
cd "$(dirname "$0")/../.."   # repo root
export FASTLANE_LARGE_BACKUP=none
echo "=== final_fix fastlane 120 $(date '+%m-%d %H:%M:%S')"
.venv/bin/fastlane eval run --system fastlane --run final_fix --calls 120
.venv/bin/fastlane eval report --run final_fix
echo "=== all done $(date '+%m-%d %H:%M:%S')"
