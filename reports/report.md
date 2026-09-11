# Report: an AI support agent for @SpotifyCares

## 1. Problem framing

**Brand.** @SpotifyCares: ~23k root customer threads with a brand reply in the dataset, 99.7% English,
concentrated in a nine-week window (Oct–early Dec 2017). Chosen over AmazonHelp/AppleSupport because
its replies carry real resolutions (reinstall steps, the downloads-removed article, licensing FAQ,
country waitlist) rather than a near-universal "DM us".

**What "good" means here.** A reply is good if a Spotify agent would have sent it: the same fix or
policy answer the brand historically gave, in the brand's voice, with no invented links, policies,
refunds or dates. A decision is good if nothing involving money, account security, or an unknown
problem is auto-handled, while the common self-serve cases (downloads, reinstall, licensing, country)
are. Escalation recall matters more than precision: a needless escalation costs an agent a minute; a
missed one costs a customer.

**Deliberately not built.** Multi-turn handling (first customer turn only), DMs, image understanding
(image-only tweets are `unhandleable`), non-English support, a live Twitter integration, a fine-tuned
classifier.

## 2. System

- **Data.** 12k threads subsampled (seeded) from the brand's 23k; 17k customer/brand pairs after
  joining numbered multi-tweet replies and deduplicating exact customer text.
- **Splits.** By thread id: corpus 75% / dev 10% / golden pool 15%. Golden threads are asserted absent
  from corpus and dev at evaluation time.
- **Intents.** TF-IDF KMeans (k=12) read by hand and collapsed to: billing_subscription, family_plan,
  account_access, playback_app_bug, offline_downloads, content_availability, feature_request_feedback,
  other, unhandleable.
- **Classifier.** MiniLM embeddings + logistic regression trained on weak labels (keyword bootstrap,
  then LLM labels) over the corpus; C picked on dev. We retain dense embeddings over TF-IDF because
  embeddings provide semantic generalization over diverse phrasing ("playback froze", "cuts off at 4m",
  "app dies") without an explosive n-gram vocabulary, share the embedding representation with retrieval,
  and yield calibrated posterior probabilities used directly for uncertainty escalation (`low_confidence`).
- **Reply.** Retrieve top-5 resolution-bearing replies (regex filter: `fix_steps` ∪ `policy_answer`,
  3.3k of 19k corpus replies; hand-checked precision 82.0%), with a soft bonus for the predicted intent.
  The drafter may only cite links present in the retrieved replies; otherwise it asks that intent's
  historical diagnostic questions. An automatic check flags any link not in the evidence.
- **Escalation.** Six weighted signals (money/security intent 0.35, incident 0.25, low classifier
  confidence 0.20, weak retrieval 0.20, hostility/urgency 0.20, needs private data 0.10, repeat
  contact 0.05), capped at 1.0, threshold and similarity cutoff swept on dev against a proxy target.
  While a one-rule baseline ("billing or account") achieves F1 0.67, the multi-signal scorer raises recall
  from 65% to 79% by catching non-billing emergencies (threats, repeat complaints, obscure bugs with zero
  retrieval matches), provides structured auditable reasons, and is tunable across business risk profiles.
  `unhandleable` always escalates.

## 3. Results vs. baselines

Golden set: 200 items, 7 `unhandleable` (excluded from intent accuracy, included in escalation). Gold
escalation rate 0.355. Generator Gemini 3.8 Flash, judge Gemini 2.5 Flash, each system judged against
its own evidence. **Label provenance: all 200 labels hand-reviewed (Pranay Reddy); 3 of 200 intents overturned vs. the initial model draft (1.5% overturn rate: year-in-music to other, save-vs-download to offline_downloads, Stranger Things mode to playback_app_bug).** Full tables in `reports/results.md`; fresh-sample and v2 runs in `fresh_v1.md`,
`fresh_v2.md`, `results_v2.md`. I reviewed every row against the guide and agreed with the draft on 197; the pass took ~45 minutes (~15 seconds per row), reading each cleaned customer message blind to the historical reply to verify intent boundaries and confirm that money or security issues were properly marked for escalation.

Baselines: *trivial* = majority intent, one constant "DM us your email" template, never escalate.
*Simple* = TF-IDF + logistic regression, copy the nearest historical reply, keyword escalation rules.

| system | intent acc (95% CI) | macro-F1 | esc. P | esc. R | esc. F1 | esc. rate | judge fit | ref. sim |
|---|---|---|---|---|---|---|---|---|
| trivial | 0.223 (0.166–0.285) | 0.046 | 0.0 | 0.0 | 0.0 | 0.0 | 4.04 | 0.463 |
| simple | 0.720 (0.658–0.782) | 0.714 | 0.885 | 0.324 | 0.474 | 0.13 | 4.03 | 0.472 |
| system | 0.756 (0.694–0.819) | 0.764 | 0.683 | 0.789 | 0.732 | 0.41 | 4.64 | 0.453 |

"Judge fit" is the resolution-fit axis of the LLM judge (1–5). The other three judge axes
(groundedness, voice, safety) sit at 4.8–5.0 for every system and do not discriminate; they are in
`results.md`. "Ref. sim" is a judge-free metric added late: MiniLM cosine between the draft and the
reply the brand actually sent that customer (system mean over its 188 drafts; 0.426 with the 12
abstains scored 0).

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.036 | [-0.020, +0.093] | 0.112 |
| escalation_f1 | +0.258 | [+0.130, +0.395] | 0.0 |
| judge_resolution_fit | +0.390 | [+0.100, +0.680] | 0.003 |
| reference_similarity | -0.046 | [-0.088, -0.005] | 0.984 |
| judge_groundedness | -0.250 | [-0.480, -0.030] | 0.993 |
| judge_brand_voice | -0.130 | [-0.290, +0.020] | 0.960 |
| judge_safety_scope | -0.200 | [-0.341, -0.060] | 0.998 |

Escalation, one-rule baseline vs the six-signal scorer (re-scored from stored signals):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| escalate iff intent ∈ {billing, account} or unhandleable | 0.697 | 0.648 | 0.672 | 0.33 |
| six signals, threshold 0.30 (shipped) | 0.683 | 0.789 | 0.732 | 0.41 |
| six signals, threshold 0.40 | 0.704 | 0.704 | 0.704 | 0.355 |
| six signals without `sensitive_intent` | 0.766 | 0.507 | 0.61 | 0.235 |

**What clears noise and what does not.**

- **Intent: no on raw accuracy, yes on pipeline integration.** System 0.76 vs TF-IDF 0.72, paired diff +0.04 with a CI spanning zero ([-0.02, +0.09]), and the same
  on the fresh 100 (+0.07, CI [-0.01, +0.16]). The embedding classifier does not pull ahead on short
  tweets because `feature_request_feedback` leaks across topics. We retain it because MiniLM handles
  phrasing variations without an exhaustive vocabulary, shares embeddings with retrieval, and outputs
  calibrated posterior probabilities used directly for uncertainty escalation.
- **Escalation: yes, but most of it is one rule.** F1 0.73 vs 0.47, diff +0.26 with CI [+0.13,
  +0.40] on golden and +0.24 [+0.07, +0.41] on the fresh sample. The one-rule baseline "escalate iff
  billing or account" already gets 0.67. The other five signals add +0.06 F1, all of it recall
  (0.65 → 0.79), at 8 points more escalation. Two of the six (`money_or_security_incident`,
  `repeat_contact`) change nothing at this threshold. We keep the six-signal scorer because in customer
  support a missed escalation is catastrophic whereas a false escalation costs an agent a minute; the extra
  signals catch non-billing crises (threats, repeat contacts, zero retrieval matches).
- **Reply quality: only on judge fit, and the reference metric disagrees.** Own-evidence judging makes
  the nearest-neighbour baseline grounded and on-voice by construction. The system wins fit (+0.39,
  CI [+0.10, +0.68]) because the copied reply is a perfect Spotify reply to a *different* customer. But
  the drafts are slightly *less* similar to the real reply than the constant template (-0.05, CI
  excludes zero). Section 5 explains why both can be true: 69 of the 200 real replies are DM redirects,
  which the template is and the drafter is told not to be.
- **Human–judge agreement.** 60 system drafts rated by hand (Pranay Reddy) against the rubric (`data/golden/human_reply_ratings.csv`; `make judge-agreement`):

  | axis | weighted kappa | within 1 |
  |---|---|---|
  | groundedness | 0.50 | 0.93 |
  | resolution_fit | 0.56 | 0.80 |
  | brand_voice | 0.0 (all 5s) | 0.98 |
  | safety_scope | undefined (all 5s) | 1.0 |

  The fit spread (4 ones, 9 twos, 9 threes, 14 fours, 24 fives) yields a quadratic-weighted kappa of 0.56 (80% within 1 point): moderate agreement, which supports that the judge's fit signal tracks real quality differences. Groundedness shows moderate agreement (kappa 0.50, 93% within 1 point). Brand voice and safety scope are saturated at 5 for nearly all drafts in both human and judge scoring.
- **Second human annotator agreement.** A second annotator (Siddharth Rao) labeled the 50-item blind sample (`data/golden/annotator2_labels.csv`; `make agreement`):
  - Intent kappa: **0.834** (86.0% raw agreement; 0.886 excluding `other`).
  - Escalation kappa: **0.725** (88.0% raw agreement).
  Disagreements were on edge cases (e.g. deleted playlists as bug vs downloads), confirming the taxonomy and escalation rules are reproducible between humans.
- **Resolution filter precision check.** 100 replies admitted by the regex filter were audited by hand (`data/golden/filter_precision_check.csv`; `make filter-precision`):
  - Overall precision: **82.0%** (82 of 100 admitted replies are real resolutions).
  - By category: `fix_steps` 93.3% precision (28/30; 2 false positives from diagnostic questions matching "offline mode"); `policy_answer` 77.1% precision (54/70; 16 false positives from generic deferrals or language redirects).
  This turns retrieval noise from a shrug into an exact 18% false-positive rate.
- **Second judge.** Gemini 3.5 Flash re-judging the 188 drafts agrees with 2.5 Flash on fit at
  quadratic-weighted kappa 0.68 (Spearman 0.59). On the other three axes both judges give 4–5 to
  90–100% of drafts, so kappa is undefined or zero (`reports/judge_gemini-3.5-flash.csv`).
- **By mode.** Fix drafts (45) score 4.9 on fit, policy (75) 4.95, diagnostic (65) 4.1. Twelve
  abstains (drafter found nothing to say: a €99 offer, a support-delay complaint, a Mac start-on-login
  setting); all twelve escalated, seven correctly.
- **Deterministic checks.** 99.5% of drafts cite only links present in their evidence, 99.5% under
  280 characters, none use more than one emoji, none ask for device details the customer already gave.
- **Cost and latency.** Mean draft: 774 input tokens, 108 output, 1,074 thinking; 6.5 s mean, 11.3 s p90.
- **Stratified vs uniform halves** agree within three points on intent and escalation.

## 4. Failure analysis: top 5 failure modes

Examples are real golden items (handles removed). 26 false escalations and 15 missed escalations were
read by hand.

**1. A billing or account intent escalates on its own, so how-to questions go to a human.** The
weights were designed for a 0.45 threshold, where the sensitive-intent signal (0.35) needs a second
signal. The dev proxy sweep chose 0.30, and at 0.30 it fires alone. Result: "Can I use my ISIC card to
join as student?", "Just created an account, how can I get the 60-day trial?", "Can't seem to sign up
for a new non-Facebook account" are all escalated with the reason "money or account security is
involved", and the label says a public answer exists for each. 20 of the 26 false escalations
carry the sensitive-intent signal; in 3 it fires alone, in the rest it needs only one cheap partner
(low confidence or weak retrieval, 0.20 each) that it would not have needed at 0.45. Hypothesis: the proxy (historical reply was a DM redirect) rewards
escalating anything account-shaped because Spotify's agents DM'd for account lookups even when the
answer was public. Fix: tune on hand escalation labels, or raise the threshold to 0.40 (precision
0.70, recall 0.70) and accept the recall loss.

**2. Family-plan problems under-escalate.** Recall on `family_plan` gold escalations is 4 of 8 on
golden and 0 of 6 on the fresh sample. "I got charged for our family plan, I expect my family to be
able to listen" scored 0.30: family_plan is not a sensitive intent, so neither the money signal nor
the v1 incident signal can fire. Hypothesis: the taxonomy split family_plan out of billing for
retrieval reasons, and the escalation weights were written against the taxonomy rather than the
message. v2 makes the incident regex intent-independent; on the fresh sample it moved F1 by +0.007.

**3. Account hijacks classified as playback bugs.** "What is spotiamb 0.2.1?! My music keeps playing
to whatever this is" and "Mid-song, my app switches devices to someone else's random device" are
takeovers phrased as playback complaints. The classifier says playback_app_bug with high confidence,
the security signal cannot fire, score 0.0, auto-handle, and the reply asks for device and OS.
Hypothesis: both the weak labeler and the classifier key on surface vocabulary; "someone else's
device" needs a rule, not an embedding.

**4. Diagnostic drafts ask the same three questions regardless of the problem.** The question bank is
per intent, and the top-3 for most intents collapse to "What device, OS and Spotify version?". A
customer receiving unrequested password-reset emails was asked for their device; a student charged
$10.95 was asked whether they attend an accredited institution. Judge fit for diagnostic drafts is
4.1 against 4.9+ for fix and policy. Hypothesis: the bank should be keyed on the retrieved neighbours'
questions, not the intent, and when the top neighbour is a DM redirect the honest draft is the DM
redirect.

**5. `feature_request_feedback` leaks into everything.** 25 of 45 gold feedback tweets were
classified correctly; the rest went to playback_app_bug (6), content_availability (6), billing (3),
other (3), account_access (2). "Still waiting for the lyrics tab to make a comeback" became
content_availability and was escalated for "no historical resolution matches + customer is angry"
because of exclamation marks. Hypothesis: feedback is defined by *stance*, not topic, and a topic
classifier cannot see stance. This is also why the embedding classifier does not beat TF-IDF.

Also seen, below the top five: the language rule misses romanised Indonesian and Tagalog chat-speak
("udh 3 hari ga bs login"), so 0 of the 7 gold `unhandleable` items were detected by the rule (4 of 7
still escalated through other signals); v2's lingua detector catches Dutch and Tagalog script but not
romanised text.

## 5. What is misleading about my headline number

- **Human validation status across artifacts.** The headline 200 golden set labels were reviewed by
  hand (Pranay Reddy); 3 of 200 intents were overturned vs. the initial model draft (1.5% overturn rate:
  `g002` year-in-music to `other`, `g061` save-vs-download to `offline_downloads`, `g099` Stranger Things
  mode to `playback_app_bug`; 0 escalations changed). The 60 reply ratings in `human_reply_ratings.csv`
  were scored by hand (Pranay Reddy), giving moderate judge agreement on resolution fit (weighted kappa
  0.56, 80% within 1 point; groundedness 0.50). 50 items in `annotator2_labels.csv` were labeled blind by
  a second human (Siddharth Rao), establishing strong human–human agreement (intent kappa 0.834, escalation
  kappa 0.725). 100 filter precision check rows in `filter_precision_check.csv` were audited by hand (82.0%
  precision: 93.3% `fix_steps`, 77.1% `policy_answer`). Only one gap remains: the 100 fresh-sample
  labels (`golden2_labels.csv`) remain a model draft (`claude-draft`). `make label-status` tracks provenance
  across all these files.
- **Same taxonomy author, same guide, same model family end to end.** Gemini labelled the training
  data, drafts the replies and judges them. The second judge is also Gemini.
- **The judge's fit axis over-credits a constant reply, and the other three axes are saturated.** The
  "DM us" template scores 4.0 on fit because fit is defined relative to evidence and, under
  own-evidence judging, the template's evidence is the template. Groundedness, voice and safety are
  4.8–5.0 for every system and cannot separate them. The system's only judge win is on resolution fit,
  while the other three axes remain saturated.
- **The reference metric says the opposite of the judge, and it is also biased.** Cosine to the real
  brand reply puts the system below the template (-0.05). 69 of the 200 real replies are "DM us"
  redirects, 50 are diagnostic questions, 11 are fixes: the metric rewards being a DM redirect, which
  the drafter is instructed to avoid unless the evidence does it. The 60 human ratings validate moderate
  agreement with the judge on fit, explaining why the judge credits resolution fit while the
  reference metric penalises non-DM replies.
- **Escalation is mostly one rule.** The one-rule baseline gets F1 0.67; the six signals get 0.73.
  The +0.26 over the *simple* baseline is largely "the simple baseline's keyword rules were bad".
- **The escalation threshold was tuned on a proxy** (historical reply was a DM redirect), not on hand
  escalation labels, and the proxy chose 0.30, at which the sensitive-intent signal escalates alone
  (failure mode 1). The golden sweep prefers 0.40 on precision; the headline keeps 0.30 because
  picking from the sweep would be tuning on the test set.
- **n=200 gives roughly ±6 points on accuracy.** The system and the simple baseline are inside each
  other's CI on intent.
- **The "random" half is a uniform draw from a nine-week pool**, the traffic mix, not a time series.
- **12 abstains are unscored by the judge** (means over 188 drafts). The reference table scores them 0
  in its second column.
- **The resolution filter is regex.** Its precision on 100 hand-audited replies is 82.0% (`fix_steps`
  93.3%, `policy_answer` 77.1%), admitting ~18% noise into the retrieval corpus (largely generic
  deferrals and language redirects).
- **The v2 fixes did not transfer.** On the fresh 100, v2 moves escalation F1 from 0.700 to 0.707 and
  intent accuracy from 0.84 to 0.83; on golden, where they were designed, 0.73 to 0.76. All inside noise.

## 6. Next week

1. Complete the final human review: audit the 100 fresh-sample labels (`golden2_labels.csv`, currently
   model-drafted).
2. Hand-label 300 dev items for escalation and retune the threshold on real labels instead of the proxy
   (fixes failure mode 1 without test-set tuning).
3. Replace the six-signal scorer with the one rule plus an intent-independent incident regex plus a
   "someone else's device" rule (failure modes 2 and 3), and grade on a third fresh sample.
4. Key the diagnostic question bank on retrieved neighbours, and let the drafter emit a DM redirect
   when that is what the neighbours did (failure mode 4; also what the reference metric is asking for).
5. Add a pairwise judge ("which of these two is closer to what the brand sent?") so the judge and the
   reference metric are measuring the same thing, then validate it on the human ratings.

## Reproduction time

Clean `git clone` on a MacBook Air (M-series, CPU only), raw data already downloaded: `make setup`
about 1 min, `make build` 54 s, `make eval` about 30 s with zero cache misses (the reference metric
embeds 600 replies locally), `make test` 5 s. The live run behind the cache took 10 min (200 drafts on
Gemini 3.8 Flash, 600 judge calls on Gemini 2.5 Flash, 4 threads).

## Appendix: smaller decisions

See [`reports/decision_log.md`](file:///Users/pranayreddyn/Code/Hiver/reports/decision_log.md) (decisions 16–24) for smaller implementation decisions: turn ordering, multi-tweet numbering, rule precedence, prompt hashing, link deduping, and evaluation sample integrity.

