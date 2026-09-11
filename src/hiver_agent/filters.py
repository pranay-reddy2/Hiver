"""Reply-category filter. Decides which historical replies count as 'resolutions'."""
from __future__ import annotations

import re

from .text import links_in

FIX_STEPS = re.compile(
    r"\b(reinstall|clean (re)?install|log(ging)? ?out|log(ging)? back in|restart(ing)?|"
    r"clear(ing)? (the |your )?(browser'?s? )?cache|offline mode|toggle|turn (it |that )?(off|on)|"
    r"steps (at|under|here|below)|check out the steps|follow the steps|update (the|your|to the latest) app|"
    r"latest version|firewall|different (device|network|browser)|remove (the|your) (device|account) from)\b",
    re.I,
)
FIX_LINKS = {"38J7tFlIBF", "EqisDMwZAT", "PDA2aKzC6Q"}
POLICY = re.compile(
    r"\b(licens|not (yet )?available|available in|launch(ing)? in|new countries|vote for|the idea|"
    r"distributor|only the songs you|will be available|as soon as|once (the )?licensing|"
    r"student discount|hulu|family plan|premium (for )?family|(don't|do not) have (any )?(info|plans)|"
    r"pass (on|along).*(feedback|thoughts)|(feature|option) (isn't|is not) (available|possible)|"
    r"at the moment|right now we|(isn't|is not|aren't|are not) (yet )?(available|possible)|(it's|its|it is) not possible|"
    r"have (some )?info(rmation)? (about|on)|info about .{0,40}here|aware of (this|the) issue|investigating|known issue|requires an active)\b",
    re.I,
)
POLICY_LINKS = {"0i8GpimuDa", "X8DVSX9QgM", "VMrgnGpB4G", "MEjmIRL2eB", "DzfTSK1S9k", "Z9OErWpTNz", "9x0iGXxLD1", "SEK4E4gi22", "ZgU70TbP8M"}
DIAG_Q = re.compile(
    r"\b(which|what|are you|can you (let us know|tell us|send)|could you (let us know|tell us)|do you|does (this|it|that)|"
    r"is (this|it|that)|have you|when did|where (are|is)|how (many|long|about))\b",
    re.I,
)
DM = re.compile(r"\bDM\b|direct message|dm us|slide into|private message", re.I)
DM_LINKS = {"ldFdZRiNAt", "ldFdZR1cbT"}
ACK = re.compile(
    r"\b(pass(ed)? (it |this |that )?(on|along)|thanks for (the|your) (feedback|thoughts|info)|"
    r"give us a shout|congrats|awesome|glad|that's what we like to hear|enjoy|you're welcome|"
    r"we hear you|rest assured|stay tuned|keep us posted|fingers crossed|you know where to find us|ever need us again|"
    r"always a tweet away|shoot us a message|have a (nice|great|good) (day|one|weekend)|made us blush|appreciate)\b",
    re.I,
)
CLOSING_LINKS = {"m4HWSbgHVZ", "FQulrDtFky"}

CATEGORIES = ["fix_steps", "policy_answer", "diagnostic_q", "dm_redirect", "ack_only", "other"]
RESOLUTION_CATEGORIES = {"fix_steps", "policy_answer"}


def categorise(brand_text: str) -> str:
    links = set(links_in(brand_text))
    if FIX_STEPS.search(brand_text) or links & FIX_LINKS:
        return "fix_steps"
    if POLICY.search(brand_text) or links & POLICY_LINKS:
        return "policy_answer"
    if "?" in brand_text and DIAG_Q.search(brand_text):
        return "diagnostic_q"
    if DM.search(brand_text) or links & DM_LINKS:
        return "dm_redirect"
    if ACK.search(brand_text) and not (links - CLOSING_LINKS):
        return "ack_only"
    return "other"


def is_resolution(brand_text: str) -> bool:
    return categorise(brand_text) in RESOLUTION_CATEGORIES
