# fresh_v1: golden set n=100, system version v1

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | 
|---|---|---|---|---|---|---|
| trivial | 0.198 (0.125–0.281) | 0.041 | 0.0 | 0.0 | 0.0 | 0.0 |
| simple | 0.771 (0.688–0.854) | 0.729 | 0.8 | 0.324 | 0.462 | 0.15 |
| system | 0.844 (0.771–0.907) | 0.834 | 0.651 | 0.757 | 0.7 | 0.43 |

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.073 | [-0.011, +0.156] | 0.065 |
| escalation_f1 | +0.238 | [+0.066, +0.413] | 0.002 |

Escalation threshold sweep (system, re-scored from stored signals):

| threshold | precision | recall | rate |
|---|---|---|---|
| 0.2 | 0.517 | 0.838 | 0.6 |
| 0.3 | 0.651 | 0.757 | 0.43 |
| 0.4 | 0.667 | 0.649 | 0.36 |

Escalation recall by gold intent (system):

| gold intent | n | recall |
|---|---|---|
| account_access | 9 | 1.0 |
| billing_subscription | 14 | 1.0 |
| content_availability | 1 | 0.0 |
| family_plan | 6 | 0.17 |
| other | 3 | 1.0 |
| unhandleable | 4 | 0.25 |

Signal ablation (system, drop one signal):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| full | 0.651 | 0.757 | 0.7 | 0.43 |
| without_sensitive_intent | 0.708 | 0.459 | 0.557 | 0.24 |
| without_money_or_security_incident | 0.651 | 0.757 | 0.7 | 0.43 |
| without_low_confidence | 0.7 | 0.757 | 0.727 | 0.4 |
| without_weak_retrieval | 0.658 | 0.676 | 0.667 | 0.38 |
| without_hostile_or_urgent | 0.643 | 0.73 | 0.684 | 0.42 |
| without_needs_private_data | 0.65 | 0.703 | 0.675 | 0.4 |
| without_repeat_contact | 0.651 | 0.757 | 0.7 | 0.43 |

Intent confusion (system; rows = gold, cols = predicted):

| gold \ pred | billing_ | family_p | account_ | playback | offline_ | content_ | feature_ | other |
|---|---|---|---|---|---|---|---|---|
| billing_subscription | 16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| family_plan | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| account_access | 0 | 0 | 14 | 0 | 0 | 0 | 0 | 0 |
| playback_app_bug | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 |
| offline_downloads | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| content_availability | 0 | 0 | 0 | 2 | 0 | 13 | 0 | 0 |
| feature_request_feedback | 1 | 0 | 2 | 3 | 0 | 0 | 13 | 0 |
| other | 1 | 1 | 1 | 1 | 0 | 0 | 2 | 3 |

Unhandleable detection: gold n=4, recall=0.25, false positives=0

By sample strategy (intent accuracy / escalation F1):

- trivial: random_uniform: 0.191 / 0.0; stratified: 0.204 / 0.0
- simple: random_uniform: 0.766 / 0.581; stratified: 0.776 / 0.286
- system: random_uniform: 0.851 / 0.773; stratified: 0.837 / 0.611

Gold escalation rate: 0.37
llm cache: 0 misses
Wall time: 15.0s