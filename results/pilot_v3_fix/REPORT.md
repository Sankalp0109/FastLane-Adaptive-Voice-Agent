# Results: run `pilot_v3_fix`

| System | Calls | Success (95% CI) | p50 model ms | p95 model ms | p95 vs B1 | Large-model turns | Router precision / hard recall | Unsafe-write calls | Missed hand-offs |
|---|---|---|---|---|---|---|---|---|---|
| fastlane | 3 | 100.0% (44-100%) | 3,246 | 16,173 | - | 50.0% | 50.0% / 66.7% | 0.0% | 0/0 |

![success vs p95](success_vs_p95.png)

## fastlane

Fine-tuned small model answers easy turns and passes hard ones to the large model or staff.

- Success by category: {'dates': '1/1', 'policy': '1/1', 'rule_break': '1/1'}
- Success by domain: {'buses': '3/3'}
- Tokens per call (prompt, completion) by role: {'small': [5141, 453], 'large': [10364, 856]}
- Large-model calls per call: 4.67; rate-limit wait per call: 3.5 s
- Large-model calls answered by: {'gemini:gemini-3.5-flash': 2, 'groq:openai/gpt-oss-120b': 12}
- Unconfirmed writes: 0; rule-violating writes: 0
- Invalid tool-call rate: 0.05; success on calls with injected errors: None
- Policy answer accuracy: 1.0; early lookup: {'started': 2, 'reused': 0, 'wasted': 1, 'cache_hits': 0, 'saved_ms_total': 0.0}
- Lookup-turn p95: 5259.3 ms; errors: 0
- Escalation reasons: {'tool error': 3, 'small model': 2, 'policy answer with numbers': 1, 'small model gave no reply': 1, 'unconfident policy match': 1}
