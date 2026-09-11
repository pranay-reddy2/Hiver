# fresh_v1: golden set n=100, system version v1

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | 
|---|---|---|---|---|---|---|
| trivial | 0.204 (0.118–0.29) | 0.042 | 0.0 | 0.0 | 0.0 | 0.0 |
| simple | 0.753 (0.667–0.839) | 0.703 | 0.867 | 0.325 | 0.473 | 0.15 |
| system | 0.817 (0.742–0.882) | 0.812 | 0.842 | 0.4 | 0.542 | 0.19 |

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.065 | [-0.011, +0.149] | 0.085 |
| escalation_f1 | +0.070 | [-0.070, +0.214] | 0.174 |

Escalation threshold sweep (system, re-scored from stored signals):

| threshold | precision | recall | rate |
|---|---|---|---|
| 0.35 | 0.7 | 0.7 | 0.4 |
| 0.45 | 0.842 | 0.4 | 0.19 |
| 0.55 | 0.842 | 0.4 | 0.19 |

Escalation recall by gold intent (system):

| gold intent | n | recall |
|---|---|---|
| account_access | 9 | 0.22 |
| billing_subscription | 14 | 0.79 |
| content_availability | 1 | 0.0 |
| family_plan | 6 | 0.0 |
| other | 3 | 0.0 |
| unhandleable | 7 | 0.43 |

Signal ablation (system, drop one signal):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| full | 0.842 | 0.4 | 0.542 | 0.19 |
| without_sensitive_intent | 0.8 | 0.1 | 0.178 | 0.05 |
| without_money_or_security_incident | 0.778 | 0.175 | 0.286 | 0.09 |
| without_low_confidence | 0.889 | 0.4 | 0.552 | 0.18 |
| without_weak_retrieval | 0.933 | 0.35 | 0.509 | 0.15 |
| without_hostile_or_urgent | 0.833 | 0.375 | 0.517 | 0.18 |
| without_needs_private_data | 0.842 | 0.4 | 0.542 | 0.19 |
| without_repeat_contact | 0.842 | 0.4 | 0.542 | 0.19 |

Intent confusion (system; rows = gold, cols = predicted):

| gold \ pred | billing_ | family_p | account_ | playback | offline_ | content_ | feature_ | other |
|---|---|---|---|---|---|---|---|---|
| billing_subscription | 15 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| family_plan | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| account_access | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |
| playback_app_bug | 1 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| offline_downloads | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| content_availability | 0 | 0 | 0 | 2 | 0 | 12 | 0 | 0 |
| feature_request_feedback | 1 | 0 | 1 | 4 | 0 | 0 | 12 | 1 |
| other | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 3 |

Unhandleable detection: gold n=7, recall=0.143, false positives=0

By sample strategy (intent accuracy / escalation F1):

- trivial: random_uniform: 0.2 / 0.0; stratified: 0.208 / 0.0
- simple: random_uniform: 0.733 / 0.545; stratified: 0.771 / 0.364
- system: random_uniform: 0.844 / 0.571; stratified: 0.792 / 0.5

Gold escalation rate: 0.4
llm cache: 0 misses
Wall time: 15.4s