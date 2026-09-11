import io

import numpy as np
import pandas as pd

from hiver_agent import prep
from hiver_agent.reply import groundedness_check, reply_checks
from hiver_agent.retrieval import Retriever

CSV = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,100,True,Tue Oct 31 22:10:47 +0000 2017,@SpotifyCares my downloads vanished https://t.co/img,2,
2,SpotifyCares,False,Tue Oct 31 22:15:47 +0000 2017,@100 1: Sorry! Check the steps at https://t.co/38J7tFlIBF /AB,3,1
3,SpotifyCares,False,Tue Oct 31 22:16:47 +0000 2017,@100 2: And let us know how it goes /AB,4,2
4,100,True,Tue Oct 31 22:30:47 +0000 2017,@SpotifyCares that worked thanks,5,3
5,SpotifyCares,False,Tue Oct 31 22:35:47 +0000 2017,@100 Awesome! Give us a shout if you need anything /CD,,4
6,101,True,Tue Oct 31 23:00:00 +0000 2017,@SpotifyCares charged twice,7,
7,SpotifyCares,False,Tue Oct 31 23:05:00 +0000 2017,@101 1: Hi there! Can you DM us your email? /EF,8,6
8,SpotifyCares,False,Tue Oct 31 23:06:00 +0000 2017,@101 1: Also which country are you in? /EF,,7
"""


def _frame():
    df = pd.read_csv(io.StringIO(CSV), dtype={"tweet_id": "int64"})
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    return df


def test_continuation_joining_and_root_finding():
    pairs = prep.build_pairs(_frame())
    first = pairs[pairs.customer_tweet_id == 1].iloc[0]
    assert first.n_parts == 2
    assert first.brand_text == "Sorry! Check the steps at [link:38J7tFlIBF] And let us know how it goes"  # signatures and numbering gone
    later = pairs[pairs.customer_tweet_id == 4].iloc[0]
    assert later.thread_id == 1 and later.turn_index == 1 and later.query_text.startswith("my downloads vanished ||")
    assert first.message_text == "my downloads vanished"  # turn 0, own root: message is the tweet itself
    # a child starting with "1:" is a new reply, not a continuation of another "1:"
    second = pairs[pairs.customer_tweet_id == 6].iloc[0]
    assert second.n_parts == 1 and "country" not in second.brand_text


def test_load_brand_frame_keeps_parent_chain(tmp_path):
    p = tmp_path / "twcs.csv"
    p.write_text(CSV)
    kept = prep.load_brand_frame(p)
    assert set(kept.tweet_id) == {1, 2, 3, 4, 5, 6, 7, 8}


def _retriever():
    frame = pd.DataFrame({
        "thread_id": [1, 2, 3, 4], "query_text": ["a", "b", "c", "d"], "root_text": ["a", "b", "c", "d"],
        "brand_text": ["Hey Sam! Try a reinstall [link:EqisDMwZAT]", "Hi there! Try a reinstall [link:EqisDMwZAT]", "Check the downloads article [link:38J7tFlIBF]", "Info on licensing [link:0i8GpimuDa]"],
        "category": ["fix_steps"] * 3 + ["policy_answer"], "weak_intent": ["playback_app_bug", "playback_app_bug", "offline_downloads", "content_availability"], "turn_index": [0] * 4,
    })
    vec = np.eye(4, dtype=np.float32)
    r = Retriever(frame, vec, {"_global": ["What device are you on?"]}, frame, vec)
    return r


def test_retrieval_dedupes_and_soft_intent(monkeypatch):
    r = _retriever()
    monkeypatch.setattr("hiver_agent.retrieval.embed", lambda texts: np.array([[0.9, 0.85, 0.5, 0.1]], dtype=np.float32))
    out = r.search("x", k=3, intent="content_availability")
    texts = out.brand_text.tolist()
    assert sum("reinstall" in t for t in texts) == 1  # near-duplicate collapsed
    assert "Info on licensing [link:0i8GpimuDa]" in texts  # soft bonus keeps the predicted intent in play, not exclusive
    assert len(out) == 3


def test_groundedness_and_format_checks():
    ex = pd.DataFrame({"brand_text": ["Try [link:EqisDMwZAT]"]})
    assert groundedness_check("Reinstall via [link:EqisDMwZAT]", ex)["grounded"]
    assert not groundedness_check("See [link:NOTREAL]", ex)["grounded"]
    assert not groundedness_check("See https://example.com", ex)["grounded"]
    c = reply_checks("Hey! What device and version are you on? 🙂", "my iPhone on iOS 11 keeps crashing")
    assert c["asks_for_given_info"] and c["at_most_one_emoji"] and c["under_280"]
