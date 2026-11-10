# Final runs — failure analysis (hand-checked transcripts)

Run `final` (2026-11-10), Groq gpt-oss-120b only, one seed. Causes: **caller** = the simulated caller did not follow its
script (no change of mind, wrong acceptance, new request); **scorer** = correct behaviour scored wrong; **agent** = the
agent's error.

## FastLane (adapter v3), 120 calls: 108 succeeded (90.0%), 0 run errors, 0 unsafe writes

| Call | Cause | What happened |
|---|---|---|
| mov-change_mind-073 | caller | never changed 8:30 PM → 5 PM; agent booked what was asked |
| mov-change_mind-074 | caller | never changed 8:30 PM → 1:30 PM |
| res-change_mind-313 | caller | never changed 9:30 PM → 7:30 PM |
| res-change_mind-315 | caller | never changed 8 PM → 7 PM |
| mov-policy-081 | caller | after the correct answer ("inside the 4-hour window") asked to cancel without a reference → hand-off; script said no booking action |
| bus-sold_out-179 | caller | goal: accept the 8 AM bus "whatever the type"; caller kept asking for an 8 AM *non-AC sleeper* (does not exist) after the agent offered the 8 AM AC seater → hand-off |
| mov-policy-083 | scorer | "a two-year-old does not need a ticket" is correct; gold wants the threshold ("3") stated |
| bus-simple-122 | agent | **invented origin**: caller never said where from; small model searched Bengaluru→Hyderabad, large model kept it; 7:30 PM exists from Chennai; booked 9 PM after the caller said Chennai |
| bus-change_mind-195 | agent | **invented origin** again (Bengaluru instead of Vijayawada, never asked); booked BLR-HYD 10 PM; gateway blocked two attempts but read-back showed the wrong origin and the caller accepted |
| res-returning-337 | agent | **false completion claim**: cancel blocked by the gateway (`needs_readback`); caller's "That's correct, but…" was not taken as consent; small model then said "Your booking was cancelled" — nothing was cancelled |
| mov-policy-085 | agent | search_policy returned the cinema guide ("Mall parking (Rs 60 for 4 hours)"); small model answered "No parking facility" — unfaithful to the retrieved text |
| res-policy-325 | agent (safe) | "Is the meat halal?" did not match the doc heading "Is the food halal?" confidently → hand-off to staff (safe, but the answer was in the corpus) |

**Summary:** 6 caller, 1 scorer, 5 agent (2 invented origin, 1 false completion claim, 1 unfaithful policy answer,
1 retrieval miss handed off safely). On the agent's own errors FastLane is 115/120 (95.8%).

**Fix candidates (after all final runs, not mid-run):**
1. Bus search needs a caller-stated origin: the gateway rejects `origin` not heard in the transcript (like the
   heard-number check of section 21) → the agent asks "Where are you travelling from?".
2. Completion claims ("cancelled", "confirmed", "booked") only from templates after a successful write; a small-model
   reply claiming completion with no successful write this turn goes to the large model.
3. Policy answers that contradict the retrieved passage: route small-model policy answers to the large model unless the
   answer overlaps the top passage.

**Development vs unseen calls:** 23/24 (95.8%) on the 24 development calls, **85/96 (88.5%)** on the 96 calls never
used for development. By category: change_mind 10/15 (4 caller), policy 11/15, simple 14/15, returning 14/15,
sold_out 14/15; dates, red_flag and rule_break 15/15.

## B1 (large model only, rules in the prompt), 120 calls: 98 succeeded (81.7%), 0 run errors

- **Booking changes made without a read-back to the caller:** 53 in 44/120 calls (37%; mid run 8/24 = 33%). 33 of these calls were still scored
  successful, so success alone hides the risk. By category: returning 15, change_mind 10, sold_out 7, dates 6, simple 4,
  rule_break 2.
- **Duplicate bookings** (same booking made twice): 7 calls (bus-change_mind-191, bus-rule_break-237, res-sold_out-297,
  bus-dates-163, mov-dates-044, bus-sold_out-180, mov-simple-005). All failed.
- **Rule-violating writes:** 3 calls.
- Failures shared with FastLane's caller/scorer list: mov-change_mind-073 (caller), bus-sold_out-179 (caller), mov-policy-083
  (scorer). The other 19 are listed in the summary script output (outcome wrong 17, hand-off where none was needed 6;
  overlapping).

## B3 (large model as router), 48 calls: 44 succeeded (91.7%), 0 run errors, 0 unsafe writes

FastLane on the same 48 calls: 46/48 (95.8%). B3 failures: mov-policy-081, bus-simple-122 (both also FastLane failures),
res-policy-321, bus-sold_out-177.

## B2 (untuned small model only), 48 calls: 14 succeeded (29.2%), 0 run errors, 0 unsafe writes

24/48 calls hit the 16-turn limit (the model loops on questions or tool calls). Same rate as the 24-call mid run (29.2%).

## A1 (no fine-tuning: untuned small model + large model), 48 calls: 21 succeeded (43.8%), 0 run errors, 0 unsafe writes

FastLane on the same 48: 46/48. A1 sends 18.1% of turns to the large model (FastLane 17.6%), 29 of them only after an
invalid tool call; 18/48 calls hit the 16-turn limit. The untuned model rarely escalates by itself (2 "policy question"
turns in 48 calls). **Not comparable with the report's A1 row (79.2%, 60.7% large turns):** that came from pilot v0
(2026-10, older code); this is the first A1 run on the final code.

## A2 (no date / alternative tools), 48 calls: 41 succeeded (85.4%), 0 run errors, 0 unsafe writes

FastLane on the same 48: 46/48. Without the tools the large model handles 32.1% of turns (FastLane 17.6%), so it costs
more large-model calls. Per category (A2 / FastLane, of 6): change_mind 3/6, dates 5/6, sold_out 5/6, policy 5/5,
simple 5/5; red_flag, returning, rule_break 6/6 both.

## A3 (rules only in the prompt, no read-back check in code), 48 calls: 35 succeeded (72.9%), 0 run errors

FastLane on the same 48: 46/48. **Unsafe:** 31/48 calls (65%) changed a booking without a read-back (40 of 50
writes); 9 writes in 9 calls broke a booking rule; rule_break calls 1/6. 20 of the 35 successful calls were unsafe,
so success alone hides the risk. Mid run (24 calls): 13/24 unsafe, 5 rule-breaking writes — same picture (H5).

## A8 matched pair (run `final_v5`, 40 restaurant calls, data v5 for both adapters)

FastLane with v5: 37/40 (92.5%); failures res-change_mind-311, res-change_mind-313, res-policy-325.
FastLane with v3 (run `final`) on the same 40 calls: 36/40.
A8 (movies+buses v5, never saw restaurant data): **37/40 (92.5%)**, 0 unsafe; invalid tool-argument rate 2.6% vs 1.7%
for all-domains v5. Failures res-change_mind-311, res-policy-325 (shared with v5), res-red_flag-349.
**Reading:** on success, no measurable loss from leaving the restaurant domain out of training (37 vs 37), only a
slightly higher invalid-argument rate; the earlier 6/8 vs 7/8 gap (mid run, mb-v2) came from the old adapter.

---

# After the fixes: run `final_fix` (FastLane v3, same 120 calls, 2026-11-10)

Fixes (experiment log section 27): origin heard at search time; completion claim without a write; negative policy
answer → large model; plus, found during this run, a question is not a yes to a read-back. The run was interrupted at
call 36 (the session ended and stopped the MLX server) and resumed; res-returning-337 was rerun alone after the
read-back fix (old row: `final_fix_superseded_res-returning-337.jsonl`; the read-back fix changes no other read-back
answer in any final run).

**Result: 105/120 (87.5%), 0 run errors, 0 unsafe writes; large-model calls per call 1.64 (before 1.76).**

| Before → after | Calls | Fix involved? |
|---|---|---|
| fail → pass | bus-simple-122, bus-change_mind-195 | yes: origin check refused the guessed Bengaluru (9 refusals in the run, all in calls where the caller never named the origin), the agent asked |
| fail → pass | res-returning-337 | yes: "Your booking was cancelled" sent to the large model; cancelled only after "Yes, please cancel the booking" |
| fail → pass | mov-policy-085 | yes: "No parking" sent to the large model → "mall parking, Rs 60 for up to four hours" |
| fail → pass | bus-sold_out-179 | no (caller behaved differently) |
| pass → fail | res-policy-322 | yes, **but the new answer is correct**: the menu says Dal Makhani "Contains no nuts"; before, the small model said "No, the dal makhani is **not** nut-free" (wrong) and the scorer accepted it because "not nut-free" contains "nut-free"; after, the large model said "prepared without any tree nuts or peanuts" (right), which no accepted phrase matches. Both scores are scorer errors. |
| pass → fail | bus-change_mind-192 | origin check fired correctly (agent asked "from where?"); the call then failed on the change of mind (small model read back the old 10 PM bus after the large model offered 2 PM) |
| pass → fail | bus-simple-121, bus-sold_out-176, mov-sold_out-056, res-policy-321, mov-change_mind-072, mov-change_mind-075 | no fix fired |

13 calls changed outcome; 8 of them with no fix involved → run-to-run variation (Groq and the simulated caller are
not deterministic) is about ±7% of calls. Paired test before vs after: 5 vs 8 discordant, p = 0.58.

**The 15 failures after the fixes:** caller 7 (no change of mind ×4: 073, 074, 313, 315; 081 new cancel request;
056 hung up at the read-back; 176 accepted a date its goal excludes), scorer 2 (083, 322), agent 6:
res-policy-325 (retrieval miss, safe hand-off, as before); bus-simple-121 ("Which cinema?" in a bus call, then repeated
passenger questions); bus-change_mind-192 (stale read-back after a change); mov-change_mind-072 and -075 (the large
model said the booking was "already" at the newly asked time — the record holds the requested time before the change
is made); res-policy-321 (the large model started a booking after a deposit question). None of the three fixed error
types occurs. Agent's own errors: 5 → 6 calls, different kinds.

**Found while checking the fixes against the training data (experiment log section 27):** the practice-call
generator's sold-out bus plans to Hyderabad search from the trip's origin although the caller only says "to
Hyderabad": 6 of 225 bus-search training steps in data v3 (4 practice calls), the likely source of the model's
guessed origin. The new gateway check rejects exactly these 4 calls when the data is regenerated (641 of 650 kept vs
645 before; no other example changes). Not retrained (no Kaggle time before 15 Nov).
