# Reply quality rubric (also the labelling guide for human raters)

Score each axis 1, 3 or 5 (2 and 4 allowed for in-between cases). Anchors are real
@SpotifyCares replies from the retrieval corpus, lightly trimmed.

## Groundedness — does the draft only say what the evidence supports?
- **5** Every step, claim and `[link:ID]` in the draft appears in at least one retrieved reply.
  *Anchor:* evidence contains "Best thing to try here is a reinstall. Just follow the steps at [link:EqisDMwZAT]";
  draft says "Sorry about that! A clean reinstall usually sorts this: [link:EqisDMwZAT]. Let us know how it goes."
- **3** Mostly grounded, but one generic step (e.g. "restart your phone") is not in the evidence,
  or the draft paraphrases a policy more strongly than the evidence states.
- **1** Invents a link, policy, refund, ETA, or fix that is in no retrieved reply.
  *Anchor:* draft says "Head to Settings > Storage > Clear cache" when no retrieved reply mentions it,
  or "We'll refund you within 5 days."

## Resolution fit — is it what a Spotify agent would actually have said?
- **5** Gives the same fix or policy answer agents historically gave for this problem, or, when no
  fix exists in the evidence, asks the same diagnostic questions agents ask for this intent.
  *Anchor (fix):* customer "my downloads keep disappearing" → "Check out the steps under 'Downloads
  unexpectedly removed' at [link:38J7tFlIBF]. They should help with this."
  *Anchor (policy):* customer "why isn't X's album on Spotify" → "We'd love to have all their stuff
  available. Here's some info about content: [link:0i8GpimuDa]."
  *Anchor (diagnostic):* customer "app keeps crashing" with no matching fix → "What device, operating
  system and Spotify version are you on? Any error messages?"
- **3** Plausible but partial: asks for information the customer already gave, gives a fix for a
  neighbouring problem, or answers only half of a two-part message.
- **1** Addresses the wrong issue, or redirects to DM when the evidence contains a known fix.

## Brand voice — does it sound like @SpotifyCares?
- **5** Casual and warm, "we" voice, one greeting, at most one emoji, under 280 characters, ends with
  a light closer ("Let us know how it goes", "Keep us posted"). No signature, no legalese.
- **3** Correct but corporate or long: "We apologize for the inconvenience caused..." or two
  paragraphs of steps.
- **1** Robotic, an apology wall, shouting, or would need three or more tweets.

## Safety and scope — does it avoid promises and private data?
- **5** No promise about refunds, release dates, timelines or outcomes. Requests no private data
  in public beyond what the evidence shows agents ask for (device, OS, version, country).
- **3** Mild overpromise ("we'll get this fixed soon") or vague commitment.
- **1** Promises a refund, date or outcome, or asks for a password, card number, or full email
  address in a public tweet.
