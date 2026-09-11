import pandas as pd

from hiver_agent.escalation import score
from hiver_agent.filters import categorise
from hiver_agent.text import clean_brand, clean_customer, is_unhandleable


def test_categorise_precedence():
    assert categorise("Best thing to try here is a reinstall: [link:EqisDMwZAT]") == "fix_steps"
    assert categorise("We'd love to have them, info about content here: [link:0i8GpimuDa]") == "policy_answer"
    assert categorise("What device, OS and Spotify version are you using?") == "diagnostic_q"
    assert categorise("Can you DM us your email? [link:ldFdZRiNAt]") == "dm_redirect"
    assert categorise("Thanks for the feedback! Give us a shout if you need anything") == "ack_only"


def test_clean_brand_strips_signature_and_numbering():
    assert clean_brand("@123 2: Thanks! Try again https://t.co/abc123 /JK") == "Thanks! Try again [link:abc123]"


def test_unhandleable_rules():
    assert is_unhandleable(clean_customer("@SpotifyCares"))
    assert is_unhandleable(clean_customer("@SpotifyCares https://t.co/x"))
    assert not is_unhandleable(clean_customer("@SpotifyCares my downloads keep disappearing every day"))


def _examples(sim):
    return pd.DataFrame({"brand_text": ["x"], "similarity": [sim], "category": ["fix_steps"]})


def test_escalation_reasons_and_threshold():
    r = score("I was charged twice, refund me now!!!", "billing_subscription", 0.9, _examples(0.8), 0.5)
    assert r.escalate and r.reason.startswith("Escalate:") and "money" in r.reason
    r = score("my downloads keep disappearing", "offline_downloads", 0.9, _examples(0.8), 0.5)
    assert not r.escalate and r.reason.startswith("Auto-handle")
    assert score("", "unhandleable", 1.0, None, 0.5).escalate
    r = score("spotify charged me twice", "billing_subscription", 0.9, _examples(0.8), 0.5)
    assert r.escalate and "changed hands" in r.reason
    r = score("does the student discount work with an ISIC card", "billing_subscription", 0.9, _examples(0.8), 0.5)
    assert not r.escalate
