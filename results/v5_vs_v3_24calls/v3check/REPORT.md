# Results: run `v3check`

| System | Calls | Success (95% CI) | p50 model ms | p95 model ms | p95 vs B1 | Large-model turns | Router precision / hard recall | Hard recall, dates in code | Unsafe-write calls | Missed hand-offs |
|---|---|---|---|---|---|---|---|---|---|---|
| fastlane | 24 | 87.5% (69-96%) | 1,464 | 6,839 | - | 25.0% | 46.9% / 75.0% | 93.8% | 0.0% | 0/3 |

![success vs p95](success_vs_p95.png)

## fastlane

Fine-tuned small model answers easy turns and passes hard ones to the large model or staff.

- Success by category: {'change_mind': '2/3', 'dates': '3/3', 'policy': '1/3', 'red_flag': '3/3', 'returning': '3/3', 'rule_break': '3/3', 'simple': '3/3', 'sold_out': '3/3'}
- Success by domain: {'buses': '8/8', 'movies': '7/8', 'restaurants': '6/8'}
- Tokens per call (prompt, completion) by role: {'small': [3577, 276], 'large': [4696, 352]}
- Large-model calls per call: 2.04; rate-limit wait per call: 7.5 s
- Large-model calls answered by: {'gemini:gemini-3.5-flash': 6, 'groq:openai/gpt-oss-120b': 43}
- Unconfirmed writes: 0; rule-violating writes: 0
- Invalid tool-call rate: 0.016; success on calls with injected errors: 1.0
- Policy answer accuracy: 0.667; early lookup: {'started': 9, 'reused': 4, 'wasted': 2, 'cache_hits': 2, 'saved_ms_total': 2.5}
- Lookup-turn p95: 16691.4 ms; errors: 0
- Escalation reasons: {'tool error': 10, 'small model gave no reply': 5, 'no availability': 4, 'staff': 4, 'policy answer with numbers': 3, 'caller changed time': 2, 'caller changed seat_category': 1, 'small model repeated a tool call': 1}
