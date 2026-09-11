# results: golden set n=200, system version v1

Label provenance: claude-draft (review each row): 200  **(model-drafted labels; every number below is agreement with a model's reading of the guide)**
Rows changed from the model draft: 0 of 200 (0.0); intent 0, escalate 0

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | judge groundedness | judge resolution_fit | judge brand_voice | judge safety_scope |
|---|---|---|---|---|---|---|---|---|---|---|
| trivial | 0.233 (0.176–0.295) | 0.047 | 0.0 | 0.0 | 0.0 | 0.0 | 4.96 | 4.04 | 4.99 | 4.86 |
| simple | 0.731 (0.668–0.788) | 0.724 | 0.885 | 0.324 | 0.474 | 0.13 | 4.8 | 4.03 | 4.88 | 4.96 |
| system | 0.751 (0.689–0.813) | 0.758 | 0.683 | 0.789 | 0.732 | 0.41 | 4.78 | 4.64 | 4.99 | 5.0 |

Reference similarity (MiniLM cosine between the reply and the reply the brand actually sent; judge-free):

| system | n with reply | mean | mean, missing reply = 0 |
|---|---|---|---|
| trivial | 200 | 0.463 | 0.463 |
| simple | 200 | 0.472 | 0.472 |
| system | 188 | 0.453 | 0.426 |

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.021 | [-0.032, +0.078] | 0.261 |
| escalation_f1 | +0.258 | [+0.130, +0.395] | 0.0 |
| reference_similarity | -0.046 | [-0.088, -0.005] | 0.984 |
| judge_groundedness | -0.250 | [-0.480, -0.030] | 0.993 |
| judge_resolution_fit | +0.390 | [+0.100, +0.680] | 0.003 |
| judge_brand_voice | -0.130 | [-0.290, +0.020] | 0.96 |
| judge_safety_scope | -0.200 | [-0.341, -0.060] | 0.998 |

Escalation threshold sweep (system, re-scored from stored signals):

| threshold | precision | recall | rate |
|---|---|---|---|
| 0.2 | 0.536 | 0.845 | 0.56 |
| 0.3 | 0.683 | 0.789 | 0.41 |
| 0.4 | 0.704 | 0.704 | 0.355 |

Escalation recall by gold intent (system):

| gold intent | n | recall |
|---|---|---|
| account_access | 23 | 0.91 |
| billing_subscription | 26 | 0.96 |
| family_plan | 8 | 0.5 |
| other | 4 | 0.5 |
| playback_app_bug | 3 | 0.0 |
| unhandleable | 7 | 0.57 |

Signal ablation (system, drop one signal; `sensitive_intent_only` is the one-rule baseline):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| sensitive_intent_only | 0.697 | 0.648 | 0.672 | 0.33 |
| full | 0.683 | 0.789 | 0.732 | 0.41 |
| without_sensitive_intent | 0.766 | 0.507 | 0.61 | 0.235 |
| without_money_or_security_incident | 0.683 | 0.789 | 0.732 | 0.41 |
| without_low_confidence | 0.689 | 0.718 | 0.703 | 0.37 |
| without_weak_retrieval | 0.699 | 0.718 | 0.708 | 0.365 |
| without_hostile_or_urgent | 0.684 | 0.761 | 0.72 | 0.395 |
| without_needs_private_data | 0.676 | 0.704 | 0.69 | 0.37 |
| without_repeat_contact | 0.683 | 0.789 | 0.732 | 0.41 |

Intent confusion (system; rows = gold, cols = predicted):

| gold \ pred | billing_ | family_p | account_ | playback | offline_ | content_ | feature_ | other |
|---|---|---|---|---|---|---|---|---|
| billing_subscription | 28 | 1 | 2 | 0 | 0 | 0 | 0 | 1 |
| family_plan | 0 | 10 | 0 | 1 | 0 | 0 | 0 | 0 |
| account_access | 3 | 0 | 23 | 2 | 0 | 0 | 0 | 1 |
| playback_app_bug | 0 | 0 | 2 | 22 | 1 | 0 | 0 | 1 |
| offline_downloads | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 |
| content_availability | 0 | 0 | 0 | 1 | 0 | 18 | 0 | 3 |
| feature_request_feedback | 3 | 0 | 2 | 6 | 0 | 6 | 25 | 3 |
| other | 0 | 0 | 1 | 3 | 2 | 2 | 1 | 8 |

Unhandleable detection by the classifier rule: gold n=7, recall=0.0, false positives=0 (escalation recall on those rows is in the per-intent table above; other signals can still escalate them)

By sample strategy (intent accuracy / escalation F1):

- trivial: random_uniform: 0.274 / 0.0; stratified: 0.194 / 0.0
- simple: random_uniform: 0.684 / 0.5; stratified: 0.776 / 0.453
- system: random_uniform: 0.737 / 0.75; stratified: 0.765 / 0.716

Automatic groundedness (links only): {'n': 200, 'grounded_rate': 0.995}
Format checks: {'under_280': 0.995, 'at_most_one_emoji': 1.0, 'asks_for_given_info': 0.0, 'mean_chars': 121.1}
Draft cost / latency: {'n_live_calls': 200, 'mean_input_tokens': 774.0, 'mean_output_tokens': 108.0, 'mean_thinking_tokens': 1074.0, 'mean_seconds': 6.54, 'p90_seconds': 11.33}
Reply modes: {'policy': 75, 'diagnostic': 65, 'fix': 45, 'none': 15}
Reply cache misses: 0

Gold escalation rate: 0.355
llm cache: 0 misses
Wall time: 23.1s