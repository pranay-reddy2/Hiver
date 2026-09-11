# Golden evaluation set

**Source.** `golden_candidates.csv` holds 200 root customer tweets to @SpotifyCares drawn from the
`golden_pool` split (thread-id split, disjoint from the retrieval corpus and the dev set; the
harness asserts this). 100 are stratified over a keyword pre-label so rare intents show up; 100 are
drawn uniformly at random from the pool. The two halves are reported separately.

**How to label.** Copy `golden_candidates.csv` to `golden_labels.csv` and fill in:

| column | values |
|---|---|
| `intent` | one of the intents in `configs/intents.yaml` |
| `escalate` | `y` if a human should handle this, `n` if the agent may reply publicly |
| `escalate_reason` | short free text, e.g. "double charge, needs account lookup" |
| `notes` | anything odd (sarcasm, two issues, image-only) |

Rules fixed before labelling:
- Bare mention, image-only, non-English, or under three real words → `unhandleable`, `escalate=y`.
- Not a customer request at all (Spotify promo copy, status-page text, a brand-authored tweet with a `/XX` signature) → `unhandleable`, `escalate=y`. Added after the second-annotator pass split on five promo tweets.
- Two intents → pick the one that decides what the agent does next.
- Money changed hands, or account security is in question → `escalate=y`.
- A known self-serve fix exists in Spotify's history (reinstall, log out/in, downloads article, licensing FAQ) → `escalate=n`.
- The keyword pre-label is deliberately not shown; do not look at `historical_reply` before choosing the intent.

**Second annotator.** `annotator2_candidates.csv` has 50 of the 200. A second person labels it
blind into `annotator2_labels.csv`; `make agreement` reports Cohen's kappa overall, excluding
`other`, and per class.

**Reply ratings.** `human_reply_ratings.csv` holds human 1–5 scores on the four rubric axes for 60
system drafts (`make judge-sheet` produces the sheet). `make judge-agreement` compares them to the LLM judge.
