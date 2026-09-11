# Results on golden set (n=200)

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | judge groundedness | judge resolution_fit | judge brand_voice | judge safety_scope |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trivial | 0.227 (0.168–0.292) | 0.046 | 0.0 | 0.0 | 0.0 | 0.0 | 2.55 | 2.11 | 3.88 | 2.73 |
| simple | 0.741 (0.676–0.805) | 0.742 | 0.92 | 0.303 | 0.455 | 0.125 | 2.77 | 2.67 | 3.96 | 4.35 |
| system | 0.73 (0.665–0.789) | 0.744 | 0.818 | 0.474 | 0.6 | 0.22 | 4.62 | 4.28 | 4.94 | 5.0 |

Escalation threshold sweep (system):

| threshold | precision | recall | rate |
|---|---|---|---|
| 0.35 | 0.73 | 0.711 | 0.37 |
| 0.45 | 0.818 | 0.474 | 0.22 |
| 0.55 | 0.829 | 0.447 | 0.205 |

By sample strategy (intent accuracy / escalation F1):

- trivial: random_by_time: 0.256 / 0.0; stratified: 0.2 / 0.0
- simple: random_by_time: 0.733 / 0.449; stratified: 0.747 / 0.462
- system: random_by_time: 0.722 / 0.586; stratified: 0.737 / 0.613

Automatic groundedness (links only): {'n': 199, 'grounded_rate': 1.0}
Reply modes: {'policy': 78, 'diagnostic': 72, 'fix': 46, 'none': 4}
Reply cache misses: 1

Gold escalation rate: 0.38
llm cache: 0 misses
Wall time: 21.2s
