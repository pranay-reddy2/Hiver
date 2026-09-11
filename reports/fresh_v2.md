# fresh_v2: golden set n=100, system version v2

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | 
|---|---|---|---|---|---|---|
| trivial | 0.204 (0.118–0.29) | 0.042 | 0.0 | 0.0 | 0.0 | 0.0 |
| simple | 0.753 (0.667–0.839) | 0.703 | 0.867 | 0.325 | 0.473 | 0.15 |
| system | 0.806 (0.731–0.882) | 0.717 | 0.81 | 0.425 | 0.557 | 0.21 |

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.054 | [-0.032, +0.140] | 0.14 |
| escalation_f1 | +0.085 | [-0.059, +0.234] | 0.133 |

Escalation threshold sweep (system, re-scored from stored signals):

| threshold | precision | recall | rate |
|---|---|---|---|
| 0.35 | 0.69 | 0.725 | 0.42 |
| 0.45 | 0.81 | 0.425 | 0.21 |
| 0.55 | 0.81 | 0.425 | 0.21 |

Escalation recall by gold intent (system):

| gold intent | n | recall |
|---|---|---|
| account_access | 9 | 0.22 |
| billing_subscription | 14 | 0.79 |
| content_availability | 1 | 0.0 |
| family_plan | 6 | 0.0 |
| other | 3 | 0.0 |
| unhandleable | 7 | 0.57 |

Signal ablation (system, drop one signal):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| full | 0.81 | 0.425 | 0.557 | 0.21 |
| without_sensitive_intent | 0.714 | 0.125 | 0.213 | 0.07 |
| without_money_or_security_incident | 0.727 | 0.2 | 0.314 | 0.11 |
| without_low_confidence | 0.85 | 0.425 | 0.567 | 0.2 |
| without_weak_retrieval | 0.882 | 0.375 | 0.526 | 0.17 |
| without_hostile_or_urgent | 0.8 | 0.4 | 0.533 | 0.2 |
| without_needs_private_data | 0.81 | 0.425 | 0.557 | 0.21 |
| without_repeat_contact | 0.81 | 0.425 | 0.557 | 0.21 |

Intent confusion (system; rows = gold, cols = predicted):

| gold \ pred | billing_ | family_p | account_ | playback | offline_ | content_ | feature_ | other |
|---|---|---|---|---|---|---|---|---|
| billing_subscription | 15 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| family_plan | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| account_access | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |
| playback_app_bug | 1 | 0 | 0 | 9 | 0 | 0 | 0 | 0 |
| offline_downloads | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| content_availability | 0 | 0 | 0 | 2 | 0 | 11 | 0 | 0 |
| feature_request_feedback | 1 | 0 | 1 | 4 | 0 | 0 | 12 | 1 |
| other | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 3 |

Unhandleable detection: gold n=7, recall=0.286, false positives=1

By sample strategy (intent accuracy / escalation F1):

- trivial: random_uniform: 0.2 / 0.0; stratified: 0.208 / 0.0
- simple: random_uniform: 0.733 / 0.545; stratified: 0.771 / 0.364
- system: random_uniform: 0.844 / 0.611; stratified: 0.771 / 0.48

Gold escalation rate: 0.4
llm cache: 0 misses
Wall time: 14.9s