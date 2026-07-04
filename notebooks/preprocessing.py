"""Text preprocessing and script profiling for Hinglish hate-speech data.

`script_profile` is kept identical in spirit to the function used in
01_data_exploration.ipynb so the "script" column here means the same thing
it did during data exploration.
"""

import re

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[a-zA-Z]")
URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|in|org|net)/\S*|t\.co/\S+")
MENTION = re.compile(r"@\w+")
HASHTAG_HASH = re.compile(r"#(\w+)")
WHITESPACE = re.compile(r"\s+")
REPEAT = re.compile(r"(.)\1{2,}")  # 3+ repeats -> 2


def script_profile(text: str) -> str:
    """Classify a post by script: Devanagari vs romanised (Latin) vs mixed.

    Thresholds match the exploration notebook: >0.6 Devanagari share is
    'mostly_devanagari', <0.1 is 'mostly_latin', anything between is
    'mixed_script'.
    """
    if not isinstance(text, str) or len(text) == 0:
        return "empty"
    devanagari = len(DEVANAGARI.findall(text))
    latin = len(LATIN.findall(text))
    total = devanagari + latin
    if total == 0:
        return "other"
    dev_ratio = devanagari / total
    if dev_ratio > 0.6:
        return "mostly_devanagari"
    elif dev_ratio < 0.1:
        return "mostly_latin"
    else:
        return "mixed_script"


def clean_text(text: str) -> str:
    """Light normalisation for the statistical baseline.

    Deliberately conservative: strips URLs and @mentions (noise that leaks
    thread structure), unwraps the '#' from hashtags so the word survives,
    collapses character elongation ('bhaaaai' -> 'bhaai') and whitespace,
    and lowercases. No stemming or stopword removal - romanised Hinglish has
    no reliable stopword list, and that is a deliberate later experiment.
    """
    if not isinstance(text, str):
        return ""
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = HASHTAG_HASH.sub(r"\1", text)
    text = REPEAT.sub(r"\1\1", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip().lower()
