# results_v2: golden set n=200, system version v2

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | judge groundedness | judge resolution_fit | judge brand_voice | judge safety_scope |
|---|---|---|---|---|---|---|---|---|---|---|
| trivial | 0.233 (0.176–0.295) | 0.047 | 0.0 | 0.0 | 0.0 | 0.0 | 4.96 | 4.04 | 4.99 | 4.86 |
| simple | 0.731 (0.668–0.788) | 0.724 | 0.885 | 0.324 | 0.474 | 0.13 | 4.8 | 4.03 | 4.88 | 4.96 |
| system | 0.746 (0.684–0.808) | 0.671 | 0.694 | 0.831 | 0.756 | 0.425 | 4.78 | 4.64 | 4.99 | 5.0 |

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.016 | [-0.041, +0.074] | 0.323 |
| escalation_f1 | +0.282 | [+0.161, +0.418] | 0.0 |
| judge_groundedness | -0.340 | [-0.580, -0.120] | 0.999 |
| judge_resolution_fit | +0.300 | [-0.000, +0.590] | 0.029 |
| judge_brand_voice | -0.230 | [-0.410, -0.060] | 0.999 |
| judge_safety_scope | -0.300 | [-0.460, -0.140] | 1.0 |

Escalation threshold sweep (system, re-scored from stored signals):

| threshold | precision | recall | rate |
|---|---|---|---|
| 0.2 | 0.538 | 0.887 | 0.585 |
| 0.3 | 0.694 | 0.831 | 0.425 |
| 0.4 | 0.716 | 0.746 | 0.37 |

Escalation recall by gold intent (system):

| gold intent | n | recall |
|---|---|---|
| account_access | 23 | 0.91 |
| billing_subscription | 26 | 1.0 |
| family_plan | 8 | 0.5 |
| other | 4 | 0.5 |
| playback_app_bug | 3 | 0.0 |
| unhandleable | 7 | 0.86 |

Signal ablation (system, drop one signal):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| full | 0.694 | 0.831 | 0.756 | 0.425 |
| without_sensitive_intent | 0.774 | 0.577 | 0.661 | 0.265 |
| without_money_or_security_incident | 0.69 | 0.817 | 0.748 | 0.42 |
| without_low_confidence | 0.701 | 0.761 | 0.73 | 0.385 |
| without_weak_retrieval | 0.714 | 0.775 | 0.743 | 0.385 |
| without_hostile_or_urgent | 0.695 | 0.803 | 0.745 | 0.41 |
| without_needs_private_data | 0.688 | 0.746 | 0.716 | 0.385 |
| without_repeat_contact | 0.694 | 0.831 | 0.756 | 0.425 |

Intent confusion (system; rows = gold, cols = predicted):

| gold \ pred | billing_ | family_p | account_ | playback | offline_ | content_ | feature_ | other |
|---|---|---|---|---|---|---|---|---|
| billing_subscription | 28 | 1 | 2 | 0 | 0 | 0 | 0 | 1 |
| family_plan | 0 | 10 | 0 | 1 | 0 | 0 | 0 | 0 |
| account_access | 3 | 0 | 22 | 2 | 0 | 0 | 0 | 1 |
| playback_app_bug | 0 | 0 | 2 | 22 | 1 | 0 | 0 | 1 |
| offline_downloads | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| content_availability | 0 | 0 | 0 | 1 | 0 | 18 | 0 | 3 |
| feature_request_feedback | 3 | 0 | 2 | 6 | 0 | 6 | 25 | 3 |
| other | 0 | 0 | 1 | 3 | 2 | 2 | 1 | 8 |

Unhandleable detection: gold n=7, recall=0.571, false positives=1

By sample strategy (intent accuracy / escalation F1):

- trivial: random_uniform: 0.274 / 0.0; stratified: 0.194 / 0.0
- simple: random_uniform: 0.684 / 0.5; stratified: 0.776 / 0.453
- system: random_uniform: 0.737 / 0.767; stratified: 0.755 / 0.747

Automatic groundedness (links only): {'n': 195, 'grounded_rate': 0.995}
Format checks: {'under_280': 0.995, 'at_most_one_emoji': 1.0, 'asks_for_given_info': 0.0, 'mean_chars': 121.0}
Draft cost / latency: {'n_live_calls': 195, 'mean_input_tokens': 774.0, 'mean_output_tokens': 108.0, 'mean_thinking_tokens': 1077.0, 'mean_seconds': 6.58, 'p90_seconds': 11.75}
Reply modes: {'policy': 72, 'diagnostic': 64, 'fix': 44, 'none': 20}
Reply cache misses: 0

Gold escalation rate: 0.355
llm cache: 0 misses
Wall time: 18.4s