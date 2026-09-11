# Report: an AI support agent for @SpotifyCares

> Status: full pipeline, harness and headline numbers are in. Still pending from a human: review of
> the draft golden labels, a human second-annotator pass on the 50-item sheet (currently a model),
> the 100-reply filter precision check, and the 60-draft human ratings behind the judge-agreement
> number. Each is a CSV under `data/golden/` with a Make target.

## 1. Problem framing

**Brand.** @SpotifyCares: ~23k root customer threads with a brand reply in the dataset, 99.7% English,
concentrated in a nine-week window (Oct–early Dec 2017). Chosen over AmazonHelp/AppleSupport because
its replies carry real resolutions (reinstall steps, the downloads-removed article, licensing FAQ,
country waitlist) rather than a near-universal "DM us".

**What "good" means here.** A reply is good if a Spotify agent would have sent it: same fix or policy
answer the brand historically gave, in the brand's voice, with no invented links, policies, refunds
or dates. A decision is good if nothing involving money, account security, or an unknown problem is
auto-handled, while the common self-serve cases (downloads, reinstall, licensing, country) are.
Escalation recall matters more than precision: a needless escalation costs an agent a minute; a
missed one costs a customer.

**Deliberately not built.** Multi-turn conversation handling (first customer turn only), DM handling,
image understanding (image-only tweets are `unhandleable`), non-English support, a live Twitter
integration, and a fine-tuned classifier.

## 2. System

- **Data.** 12k threads subsampled (seeded) from the brand's 23k; 17k customer/brand pairs after
  joining numbered multi-tweet replies and deduplicating exact customer text.
- **Splits.** By thread id: corpus 75% / dev 10% / golden pool 15%. Golden threads are asserted absent
  from corpus and dev at evaluation time.
- **Intents.** TF-IDF KMeans (k=12) read by hand and collapsed to: billing_subscription,
  family_plan, account_access, playback_app_bug, offline_downloads, content_availability,
  feature_request_feedback, other, unhandleable.
- **Classifier.** MiniLM embeddings + logistic regression trained on weak labels
  (keyword bootstrap, then LLM labels) over the corpus; C picked on dev.
- **Reply.** Retrieve top-5 resolution-bearing replies (filter: `fix_steps` ∪ `policy_answer`,
  3.3k of 19k corpus replies), restricted to the predicted intent when possible. The drafter may only
  cite links present in the retrieved replies; otherwise it asks that intent's historical diagnostic
  questions. An automatic check flags any link not in the evidence.
- **Escalation.** Six weighted signals (money/security intent 0.35, low classifier confidence 0.20,
  weak retrieval 0.20, hostility/urgency 0.20, needs private data 0.10, repeat contact 0.05);
  threshold and similarity cutoff swept on dev. `unhandleable` always escalates.

## 3. Results vs. baselines

Golden set: 200 items, 7 of them `unhandleable` (excluded from intent accuracy, included in escalation).
Gold escalation rate 0.355. Generator Gemini 3.8 Flash, judge Gemini 2.5 Flash, each system judged
against its own evidence. Full tables (confusion matrix, per-intent recall, signal ablation) in
`reports/results.md`; the fresh-sample and v2 runs in `fresh_v1.md`, `fresh_v2.md`, `results_v2.md`.

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | judge groundedness | judge resolution_fit | judge brand_voice | judge safety_scope |
|---|---|---|---|---|---|---|---|---|---|---|
| trivial | 0.233 (0.176–0.295) | 0.047 | 0.0 | 0.0 | 0.0 | 0.0 | 4.96 | 4.04 | 4.99 | 4.86 |
| simple | 0.731 (0.668–0.788) | 0.724 | 0.885 | 0.324 | 0.474 | 0.13 | 4.8 | 4.03 | 4.88 | 4.96 |
| system | 0.751 (0.689–0.813) | 0.758 | 0.683 | 0.789 | 0.732 | 0.41 | 4.78 | 4.64 | 4.99 | 5.0 |

System minus simple baseline, paired bootstrap (2000 resamples):

| metric | diff | 95% CI | P(diff ≤ 0) |
|---|---|---|---|
| intent_accuracy | +0.021 | [-0.032, +0.078] | 0.261 |
| escalation_f1 | +0.258 | [+0.130, +0.395] | 0.0 |
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

**What clears noise and what does not.**

- **Intent: no.** System 0.75 vs TF-IDF 0.73, paired difference +0.02 with a CI spanning zero, and the
  same on the fresh 100 (+0.07, CI [-0.01, +0.16]). The embedding classifier is not better than TF-IDF
  on 100-character tweets; `feature_request_feedback` is the leak (25 of 45 correct; see failure mode 5).
- **Escalation: yes.** F1 0.73 vs 0.47, paired difference +0.26 with CI [+0.13, +0.40] on golden and
  +0.24 with CI [+0.07, +0.41] on the fresh sample. The win is recall (0.79 vs 0.32) bought with
  precision (0.68 vs 0.89) at a 41% escalation rate against a 36% gold rate. The ablation shows
  `sensitive_intent` carries it (F1 drops to 0.61 without it); `money_or_security_incident` and
  `repeat_contact` change nothing at this threshold and `unhandleable` detection is 0 of 7 in v1.
- **Reply quality: only on resolution fit.** Judging each system against its own evidence makes the
  nearest-neighbour baseline grounded and on-voice by construction: it copies a real reply. So the
  system loses on groundedness (-0.25) and safety (-0.20), ties on voice, and wins on resolution fit
  (+0.39, CI [+0.10, +0.68]). That is the honest value of drafting: the baseline's reply is a perfect
  Spotify reply to a *different* customer.
- **By mode.** Fix drafts (45) score 4.9 on fit, policy (75) 4.95, diagnostic (65) 4.1. Twelve drafts
  are abstains: the drafter returned no reply because neither the evidence nor the question bank fit
  the issue (a €99 offer, a support-delay complaint, a Mac start-on-login setting); all twelve were
  escalated, seven of them correctly.
- **Second judge.** Re-judging the 188 scored drafts with Gemini 3.5 Flash gives quadratic-weighted
  kappa 0.68 (Spearman 0.59) with the 2.5 Flash judge on resolution fit, the one axis that carries the
  result. Groundedness, voice and safety are saturated near 5 for both judges, so agreement there is
  90 to 100% exact but kappa is undefined or zero; those axes do not discriminate on this data.
  Per-item scores in `reports/judge_gemini-3.5-flash.csv`.
- **Deterministic checks.** 99.5% of drafts cite only links present in their evidence, 99.5% are under
  280 characters, none use more than one emoji, none ask for device details the customer already gave.
- **Cost and latency.** Mean draft: 774 input tokens, 108 output, 1,074 thinking; 6.5 s mean, 11.3 s p90.
- **Stratified vs uniform halves** agree within three points on intent and escalation.

## 4. Failure analysis: top 5 failure modes

Examples are real golden items (handles and names removed). 40 escalation misses and 8 false escalations
were reviewed by hand.

**1. The harness was grading labels written on the wrong text.** The first failure analysis reported
15 non-actionable golden tweets, mostly Spotify promo copy, that the `unhandleable` rule missed. Tracing
them showed the real bug: customers often reply to someone else's tweet (a promo, another customer), and
prep used the thread *root* as the customer message while the agent was fed the customer's *own* tweet.
Nine "promo tweets" in the golden set were promos the customer had replied to; the actual messages
("I just paid for the premium and the money came out of my account but premium isn't working") were
real requests, and my labels on them were wrong. 203 of 12,000 threads in the sample have a root by a
different author; 494 are the customer continuing their own thread. The fix carries the exact agent
input (`message_text`) through pairs, retrieval keys, weak labels and both golden files, and 22 golden
rows were relabelled. Hypothesis for why it survived so long: the `turn_index == 0` filter looked like
"first customer message" and nothing ever compared the labelled column to the evaluated column. A
test now builds pairs from a synthetic thread with a promo root. After the fix the genuine
non-actionable set is smaller (non-English, image-only, under three words) and the language detector
(lingua, v2) catches Dutch and Tagalog but not romanised Indonesian chat-speak ("udh 3 hari ga bs login").

**2. Family-plan problems under-escalate.** Recall on `family_plan` gold escalations is 4 of 8 on golden
and 0 of 6 on the fresh sample.
"I got charged for our family plan, I expect my family to be able to listen" scored 0.30: family_plan is
not a sensitive intent, so neither the money signal nor the incident signal can fire. Hypothesis: the
taxonomy split family_plan out of billing for retrieval reasons, and the escalation weights were written
against the taxonomy rather than against what the message says. Fix: make the incident regex
intent-independent.

**3. Account hijacks classified as playback bugs.** "What is spotiamb 0.2.1?! My music keeps playing
to whatever this is" and "Mid-song, my app switches devices to someone else's random device" are account
takeovers phrased as playback complaints. The classifier says playback_app_bug (confidence high), the
security signal cannot fire, score 0.0, auto-handle, and the reply asks for device and OS. Hypothesis:
the weak labeler and the classifier both key on surface vocabulary; "someone else's device" needs a
rule, not an embedding.

**4. Diagnostic-mode drafts ask the same three questions regardless of the problem.** The question bank
is per intent, and the top-3 questions for most intents collapse to "What device, operating system and
Spotify version are you using?". A customer receiving unrequested password-reset emails was asked for
their device and OS; a student charged $10.95 was asked whether they attend an accredited institution.
Judge fit for diagnostic drafts is 4.1, against 4.9+ for fix and policy drafts. Hypothesis: the bank
should be keyed on the retrieved neighbours' questions, not the intent; and when the top neighbour is a
DM redirect, the honest draft is the DM redirect.

**5. `feature_request_feedback` leaks into everything.** 25 of 45 gold feedback tweets were classified
correctly; the rest went to playback_app_bug (6), content_availability (6), billing (3), other (3),
account_access (2).
"Still waiting for the lyrics tab to make a comeback" became content_availability and was escalated for
"no historical resolution matches + customer is angry" because of the exclamation marks. Hypothesis:
feedback is defined by *stance* (a wish, an opinion) rather than *topic*, and a topic classifier cannot
see stance. This is also why the embedding classifier does not beat TF-IDF.

## 5. What is misleading about my headline number

- **The golden labels are a model draft awaiting human review.** They were drafted by an AI assistant
  (Claude) from the labelling guide and are marked `claude-draft` in the file; the second-annotator pass
  is another model (Gemini 3.8 Flash), so the kappa of 0.79 (intent) / 0.64 (escalation) is model-model
  agreement, not human agreement. Until a human reviews the 200 and labels the 50, every number in
  section 3 measures agreement with a model's reading of the guide.
- **Same taxonomy author, same labelling guide, same model family as the judge.** The intents fit one
  reading of the data, and Gemini both labelled the training data and judges the drafts.
- **n=200 gives roughly ±6 points on accuracy.** The system and the simple baseline are inside each
  other's CI on intent; the reply-quality gap is the only result that clears the noise comfortably.
- **The "random" half is a uniform draw from a nine-week pool.** It was first called "random by
  time", which overstated it; it is the traffic mix, not a time series.
- **The escalation threshold was tuned on a proxy** (historical reply was a DM redirect), not on
  hand escalation labels. The proxy chose 0.30, which escalates 41% of traffic; 0.40 trades recall 0.79
  for 0.70 at 36%. The headline uses the proxy's choice because picking from the sweep would be tuning
  on the test set.
- **Weak labels are LLM labels.** The classifier learns the LLM's reading of my intent definitions;
  the human golden set is the only check on that.
- **The judge (Gemini 2.5 Flash) is a smaller model from the same family as the generator (Gemini 3.8 Flash).** Absolute anchored scoring
  and the human-agreement numbers limit, but do not remove, self-preference.
- **Groundedness and voice are not where the system wins, and they cannot be.** A copied historical
  reply is perfectly grounded and perfectly on-voice by definition; own-evidence judging makes that
  visible. The only judge axis the system wins is resolution fit. The earlier draft of this report had
  the system ahead on every axis because the baseline was judged against evidence it never saw.
- **The judge over-credits a constant reply, and the rubric lets it.** The template baseline, one
  fixed "DM us your email or username" sent to every customer, scores 4.0 on resolution fit and 4.9 on
  safety. Safety is defensible: the template asks for the email *by DM*, which is what Spotify's agents
  did about ten thousand times in the corpus. Fit is not: a reply that addresses no customer's issue
  should not average 4. It scores that high because fit is defined relative to the evidence and, under
  own-evidence judging, the template's evidence is the template. Whether that is a judge failure or a
  rubric failure only the 60 human ratings can say. A second Gemini judge agrees with the first on fit
  (kappa 0.68), which rules out one model's quirk but not a shared reading of the rubric; until the
  human ratings exist the judge columns are unvalidated on exactly the axis where the system claims
  its win.
- **Two escalation signals are dead weight at this threshold.** The ablation shows dropping
  `money_or_security_incident` or `repeat_contact` changes nothing; `sensitive_intent` carries the
  result (F1 0.73 to 0.61 without it). The six-signal design is mostly a one-signal rule plus noise.
- **12 abstains are unscored.** The judge means are over 188 drafts; the 12 items where the drafter
  declined would likely score low on fit had it been forced to answer.
- **The resolution filter is regex.** Its hand-checked precision on 100 admitted replies is pending;
  whatever it is, the retrieval corpus contains that much noise.
- **The v2 fixes did not transfer.** On the fresh 100, v2 moves escalation F1 from 0.700 to 0.707 and
  intent accuracy from 0.84 to 0.83; on golden, where they were designed, 0.73 to 0.76. Both inside
  noise. The language detector catches Dutch and Tagalog and misses romanised Indonesian chat-speak.

## 6. Next week

1. Human pass on the golden labels and the 50-item sheet; human ratings on the 60 drafts. Nothing else
   matters until the numbers are anchored to a person.
2. Fix failure modes 1 and 2 (language/"not a request" detector; intent-independent incident regex),
   re-run, and report the delta against this baseline, on a fresh golden sample so the fix is not
   graded on the examples that motivated it.
3. Hand-label 300 dev items for escalation and retune the threshold on real labels instead of the proxy.
4. Key the diagnostic question bank on retrieved neighbours, and let the drafter emit a DM redirect when
   that is what the neighbours did.
5. Replace the regex resolution filter with an LLM pass over the 19k replies, once the 100-reply
   precision check says how bad the regex is.

## Reproduction time

Measured on a clean `git clone` on a MacBook Air (M-series, CPU only), raw data already downloaded:
`make setup` about 1 min, `make build` 54 s, `make eval` 18 s with zero cache misses and the exact
headline table, `make test` 5 s. The live run behind the cache took 10 min (200 drafts on Gemini 3.8
Flash and 600 judge calls on Gemini 2.5 Flash, 4 threads).

## Appendix: smaller decisions

- **Join numbered multi-tweet replies only when the number is N+1.** A child reply starting with "1:" is a new reply, not a continuation; the first pass got this wrong and left signatures mid-text.
- **Precedence fix > policy > diagnostic > DM > ack.** A reply that both asks a question and gives a fix is a fix.
- **Rules first, LLM second, for weak labels.** Keyword rules run with no API and seed the classifier; LLM labels override them when cached. The rule/LLM agreement is printed so the upgrade is visible.
- **Classifier trained on weak labels, never on golden.** Golden is 200 hand labels; using them for training would leave nothing to report.
- **`unhandleable` is a rule, applied before the classifier, always escalated, excluded from accuracy.** One rule for both annotators removes an ambiguous class from kappa.
- **Judge sees the system's retrieved evidence for every system, including baselines.** Otherwise baselines could not be scored for groundedness at all.
- **Golden set is half stratified, half random-by-time, reported separately.** Stratified shows per-class behaviour; random shows the traffic mix the agent would actually face.
- **Subsample 12k threads, not all 23k.** Embedding and indexing stay under three minutes on a laptop CPU and the retrieval corpus still has 3.3k resolutions.
- **Gemini 3.1 Pro generates, Gemini 2.5 Flash labels and judges.** `gemini-2.5-pro` is retired for new keys; the 404 was discovered on the first annotator call. Flash has the larger daily quota, which is why it does the 1,000+ bulk calls.
- **Add a "money moved / account compromised" signal after the demo, before looking at golden.** "Charged me twice" scored 0.35 and would have auto-handled; the fix and its test were written before the first eval run.
- **Judge evidence includes the diagnostic question bank.** The first judge pass scored diagnostic drafts 2.3 on groundedness because the questions they quote were never shown to the judge. That is a harness bug, not a system change, so it was fixed and re-judged.
- **Promo copy, status-page text and brand-authored tweets are `unhandleable`.** Rule added to the guide after the second annotator split on five promo tweets; the system does not yet implement it, and that is failure mode 1.
- **Each system is judged against its own evidence.** The first pass gave every system the agent's retrieved evidence, which penalised the nearest-neighbour baseline for citing a reply the judge could not see. Now the baseline's evidence is its own five nearest replies and the template baseline's evidence is the template.
- **Cache key now includes effort and max_tokens.** Changing effort used to serve stale output. The change invalidated the old cache, so the weak labels were re-run (Gemini at temperature 0 is close to but not exactly deterministic; the label distribution is reported both times).
- **Retrieval dedupes near-identical replies and uses a soft intent bonus instead of a hard mask.** SpotifyCares reuses the same sentences thousands of times, so top-5 was often one resolution five times; and a wrong intent no longer wipes out the evidence.
- **The thread sample is pinned to its original population.** Rewriting turn ordering changed which threads were eligible for the seeded subsample and silently dropped 121 golden threads; eligibility is now stated explicitly (brand answered the root or a one-hop reply) and the golden files assert every thread is in the golden pool.
