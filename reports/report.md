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
  then LLM labels) over the corpus; C picked on dev.
- **Reply.** Retrieve top-5 resolution-bearing replies (regex filter: `fix_steps` ∪ `policy_answer`,
  3.3k of 19k corpus replies), with a soft bonus for the predicted intent. The drafter may only cite
  links present in the retrieved replies; otherwise it asks that intent's historical diagnostic
  questions. An automatic check flags any link not in the evidence.
- **Escalation.** Six weighted signals (money/security intent 0.35, incident 0.25, low classifier
  confidence 0.20, weak retrieval 0.20, hostility/urgency 0.20, needs private data 0.10, repeat
  contact 0.05), capped at 1.0, threshold and similarity cutoff swept on dev against a proxy target.
  `unhandleable` always escalates.

## 3. Results vs. baselines

Golden set: 200 items, 7 `unhandleable` (excluded from intent accuracy, included in escalation). Gold
escalation rate 0.355. Generator Gemini 3.8 Flash, judge Gemini 2.5 Flash, each system judged against
its own evidence. **Label provenance: all 200 labels are a model draft (Claude) awaiting my review;
see section 5 first.** Full tables in `reports/results.md`; fresh-sample and v2 runs in `fresh_v1.md`,
`fresh_v2.md`, `results_v2.md`.

Baselines: *trivial* = majority intent, one constant "DM us your email" template, never escalate.
*Simple* = TF-IDF + logistic regression, copy the nearest historical reply, keyword escalation rules.

| system | intent acc (95% CI) | macro-F1 | esc. P | esc. R | esc. F1 | esc. rate | judge fit | ref. sim |
|---|---|---|---|---|---|---|---|---|
| trivial | 0.233 (0.176–0.295) | 0.047 | 0.0 | 0.0 | 0.0 | 0.0 | 4.04 | 0.463 |
| simple | 0.731 (0.668–0.788) | 0.724 | 0.885 | 0.324 | 0.474 | 0.13 | 4.03 | 0.472 |
| system | 0.751 (0.689–0.813) | 0.758 | 0.683 | 0.789 | 0.732 | 0.41 | 4.64 | 0.453 |

"Judge fit" is the resolution-fit axis of the LLM judge (1–5). The other three judge axes
(groundedness, voice, safety) sit at 4.8–5.0 for every system and do not discriminate; they are in
`results.md`. "Ref. sim" is a judge-free metric added late: MiniLM cosine between the draft and the
reply the brand actually sent that customer (system mean over its 188 drafts; 0.426 with the 12
abstains scored 0).

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.021 | [-0.032, +0.078] | 0.261 |
| escalation_f1 | +0.258 | [+0.130, +0.395] | 0.0 |
| judge_resolution_fit | +0.390 | [+0.100, +0.680] | 0.003 |
| reference_similarity | -0.046 | [-0.088, -0.005] | 0.984 |
| judge_groundedness | -0.250 | [-0.480, -0.030] | 0.993 |
| judge_safety_scope | -0.200 | [-0.341, -0.060] | 0.998 |

Escalation, one-rule baseline vs the six-signal scorer (re-scored from stored signals):

| config | precision | recall | F1 | rate |
|---|---|---|---|---|
| escalate iff intent ∈ {billing, account} or unhandleable | 0.697 | 0.648 | 0.672 | 0.33 |
| six signals, threshold 0.30 (shipped) | 0.683 | 0.789 | 0.732 | 0.41 |
| six signals, threshold 0.40 | 0.704 | 0.704 | 0.704 | 0.355 |
| six signals without `sensitive_intent` | 0.766 | 0.507 | 0.61 | 0.235 |

**What clears noise and what does not.**

- **Intent: no.** System 0.75 vs TF-IDF 0.73, paired diff +0.02 with a CI spanning zero, and the same
  on the fresh 100 (+0.07, CI [-0.01, +0.16]). The embedding classifier is not better than TF-IDF on
  100-character tweets; `feature_request_feedback` is the leak (failure mode 5).
- **Escalation: yes, but most of it is one rule.** F1 0.73 vs 0.47, diff +0.26 with CI [+0.13,
  +0.40] on golden and +0.24 [+0.07, +0.41] on the fresh sample. The one-rule baseline "escalate iff
  billing or account" already gets 0.67. The other five signals add +0.06 F1, all of it recall
  (0.65 → 0.79), at 8 points more escalation. Two of the six (`money_or_security_incident`,
  `repeat_contact`) change nothing at this threshold.
- **Reply quality: only on judge fit, and the reference metric disagrees.** Own-evidence judging makes
  the nearest-neighbour baseline grounded and on-voice by construction. The system wins fit (+0.39,
  CI [+0.10, +0.68]) because the copied reply is a perfect Spotify reply to a *different* customer. But
  the drafts are slightly *less* similar to the real reply than the constant template (-0.05, CI
  excludes zero). Section 5 explains why both can be true: 69 of the 200 real replies are DM redirects,
  which the template is and the drafter is told not to be.
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

- **No human has labelled anything yet.** All 200 golden labels and the 100 fresh labels were drafted
  by Claude from the labelling guide and carry `labeler=claude-draft`. The second-annotator pass is
  Gemini 2.5 Pro, so the kappa of 0.79 (intent) / 0.64 (escalation) is model–model agreement. The 60
  reply ratings and the 100 filter checks are empty sheets. Every number in section 3 measures
  agreement with a model's reading of my guide. `make label-status` prints this, and `make eval` prints
  it at the top of `results.md`; the draft labels are kept in `draft_intent`/`draft_escalate` so the
  overturn rate is reported the moment the review happens.
- **Same taxonomy author, same guide, same model family end to end.** Gemini labelled the training
  data, drafts the replies and judges them. The second judge is also Gemini.
- **The judge's fit axis over-credits a constant reply, and the other three axes are saturated.** The
  "DM us" template scores 4.0 on fit because fit is defined relative to evidence and, under
  own-evidence judging, the template's evidence is the template. Groundedness, voice and safety are
  4.8–5.0 for every system and cannot separate them. The system's only judge win is on the one axis
  whose validity a human has not checked.
- **The reference metric says the opposite of the judge, and it is also biased.** Cosine to the real
  brand reply puts the system below the template (-0.05). 69 of the 200 real replies are "DM us"
  redirects, 50 are diagnostic questions, 11 are fixes: the metric rewards being a DM redirect, which
  the drafter is instructed to avoid unless the evidence does it. Neither metric is a substitute for
  the 60 human ratings.
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
- **The resolution filter is regex.** Its precision on the 100 sampled replies is unchecked; whatever
  it is, the retrieval corpus contains that much noise.
- **The v2 fixes did not transfer.** On the fresh 100, v2 moves escalation F1 from 0.700 to 0.707 and
  intent accuracy from 0.84 to 0.83; on golden, where they were designed, 0.73 to 0.76. All inside noise.

## 6. Next week

1. Do the human pass: review the 200 labels, label the 50-item sheet blind, rate the 60 drafts, check
   the 100 filter rows. Report the overturn rate and the human–judge kappa. Nothing else matters
   until the numbers are anchored to a person.
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

- **The harness once graded labels written on the wrong text.** Customers often reply to someone
  else's tweet (a promo, another customer); prep used the thread *root* as the customer message while
  the agent was fed the customer's *own* tweet. Nine "promo tweets" in the golden set were promos the
  customer had replied to. The fix carries the exact agent input (`message`) through pairs, retrieval
  keys, weak labels and both golden files; 22 rows were relabelled; a test builds pairs from a
  synthetic thread with a promo root.
- **Join numbered multi-tweet replies only when the number is N+1.** A child reply starting with "1:"
  is a new reply, not a continuation.
- **Precedence fix > policy > diagnostic > DM > ack** when a reply does more than one thing.
- **Classifier trained on weak labels, never on golden.**
- **`unhandleable` is a rule, applied before the classifier, always escalated, excluded from accuracy.**
- **Judge evidence includes the diagnostic question bank**, otherwise diagnostic drafts scored 2.3 on
  groundedness for quoting questions the judge had not seen.
- **Cache key includes effort and max_tokens**; changing effort used to serve stale output.
- **Retrieval dedupes near-identical replies and uses a soft intent bonus instead of a hard mask.**
- **The thread sample is pinned to its original population**; a turn-ordering rewrite once silently
  dropped 121 golden threads, so eligibility is now explicit and the golden files assert membership.
