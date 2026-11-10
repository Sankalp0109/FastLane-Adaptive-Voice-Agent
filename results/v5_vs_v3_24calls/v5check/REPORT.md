# Results: run `v5check`

| System | Calls | Success (95% CI) | p50 model ms | p95 model ms | p95 vs B1 | Large-model turns | Router precision / hard recall | Hard recall, dates in code | Unsafe-write calls | Missed hand-offs |
|---|---|---|---|---|---|---|---|---|---|---|
| fastlane | 24 | 87.5% (69-96%) | 1,574 | 9,545 | - | 21.0% | 37.9% / 61.1% | 78.6% | 0.0% | 0/3 |
| A8 | 8 | 75.0% (41-93%) | 1,267 | 6,777 | - | 19.5% | 55.6% / 62.5% | 80.0% | 0.0% | 0/1 |

![success vs p95](success_vs_p95.png)

## fastlane

Fine-tuned small model answers easy turns and passes hard ones to the large model or staff.

- Success by category: {'change_mind': '1/3', 'dates': '3/3', 'policy': '2/3', 'red_flag': '3/3', 'returning': '3/3', 'rule_break': '3/3', 'simple': '3/3', 'sold_out': '3/3'}
- Success by domain: {'buses': '8/8', 'movies': '6/8', 'restaurants': '7/8'}
- Tokens per call (prompt, completion) by role: {'small': [3914, 316], 'large': [4412, 297]}
- Large-model calls per call: 1.83; rate-limit wait per call: 1.2 s
- Large-model calls answered by: {'gemini:gemini-3.5-flash': 9, 'groq:openai/gpt-oss-120b': 35}
- Unconfirmed writes: 0; rule-violating writes: 0
- Invalid tool-call rate: 0.05; success on calls with injected errors: 1.0
- Policy answer accuracy: 0.667; early lookup: {'started': 5, 'reused': 1, 'wasted': 2, 'cache_hits': 0, 'saved_ms_total': 0.2}
- Lookup-turn p95: 10343.5 ms; errors: 0
- Escalation reasons: {'tool error': 15, 'no availability': 4, 'staff': 3, 'small model': 3, 'policy answer with numbers': 2, 'caller changed seat_category': 1, 'caller changed time': 1}

## A8

Adapter trained on movies and buses only; tested on restaurants.

- Success by category: {'change_mind': '0/1', 'dates': '1/1', 'policy': '0/1', 'red_flag': '1/1', 'returning': '1/1', 'rule_break': '1/1', 'simple': '1/1', 'sold_out': '1/1'}
- Success by domain: {'restaurants': '6/8'}
- Tokens per call (prompt, completion) by role: {'small': [3896, 297], 'large': [2424, 212]}
- Large-model calls per call: 1.12; rate-limit wait per call: 0.0 s
- Large-model calls answered by: {'gemini:gemini-3.5-flash': 1, 'groq:openai/gpt-oss-120b': 8}
- Unconfirmed writes: 0; rule-violating writes: 0
- Invalid tool-call rate: 0.029; success on calls with injected errors: None
- Policy answer accuracy: 1.0; early lookup: {'started': 0, 'reused': 0, 'wasted': 0, 'cache_hits': 0, 'saved_ms_total': 0.0}
- Lookup-turn p95: 20533.3 ms; errors: 0
- Escalation reasons: {'tool error': 4, 'small model gave no reply': 2, 'no availability': 1, 'caller changed time': 1, 'staff': 1}
