"""Create tiny fixtures that mirror the exact on-disk schema of each source.
Text is synthetic but shaped like the real romanised/Devanagari mix so the
loaders, the romanised filter, and the baseline all get exercised.
"""
import json
import random
from pathlib import Path

random.seed(0)
ROOT = Path(__file__).parent / "fixtures"

# vocab pools chosen so tokens repeat (min_df=2 needs overlap) and the two
# classes are separable enough for a sane baseline F1.
HATE = ["nafrat", "chutiya", "gaddar", "bakwas", "bhadwa", "randi", "kutta"]
NEUT = ["accha", "shukriya", "namaste", "khana", "cricket", "movie", "dost"]
FILLER = ["hai", "ka", "me", "tha", "ye", "wo", "bhai", "yaar", "sab", "log"]


def _sent(pool):
    words = [random.choice(pool) for _ in range(3)] + \
            [random.choice(FILLER) for _ in range(4)]
    random.shuffle(words)
    return " ".join(words)


def make_bohra(n=80):
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "hate_speech.tsv"
    lines = []
    for i in range(n):
        if i % 3 == 0:
            lines.append(f"{_sent(HATE)}\tyes  ")
        else:
            lines.append(f"{_sent(NEUT)}\tno  ")
    p.write_text("\n".join(lines), encoding="utf-8")


def make_hasoc2022(n=80):
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "final.csv"
    rows = [",tweet_id,text,label"]
    for i in range(n):
        if i % 5 == 0:                      # some pure Devanagari (should be filtered out)
            text, label = "\u092d\u0915\u094d\u0924 \u0917\u0926\u094d\u0926\u093e\u0930 \u0939\u0948", "HOF"
        elif i % 3 == 0:
            text, label = _sent(HATE), "HOF"
        else:
            text, label = _sent(NEUT), "NOT"
        # quote the text field like the real csv
        rows.append(f'{i},{1000+i},"{text}",{label}')
    p.write_text("\n".join(rows), encoding="utf-8")


def make_hasoc2021():
    # one topic, two thread folders, each with data.json + labels.json
    base = ROOT / "hasoc2021" / "train" / "sample_topic"
    for t, root_hate in enumerate([True, False]):
        d = base / f"thread_{t}"
        d.mkdir(parents=True, exist_ok=True)
        ids, labels = [], {}

        def node(depth, hate):
            tid = str(random.randint(10**17, 10**18))
            ids.append(tid)
            if random.random() < 0.4:       # some Devanagari nodes to be filtered
                tweet = "\u0928\u092b\u0930\u0924 \u0915\u0940 \u092c\u093e\u0924"
            else:
                tweet = _sent(HATE if hate else NEUT)
            labels[tid] = "HOF" if hate else "NONE"
            obj = {"tweet_id": tid, "tweet": tweet}
            if depth > 0:
                key = "comments" if depth == 2 else "replies"
                obj[key] = [node(depth - 1, hate and random.random() < 0.7)
                            for _ in range(3)]
            return obj

        data = node(2, root_hate)
        (d / "data.json").write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        (d / "labels.json").write_text(json.dumps(labels, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    make_bohra()
    make_hasoc2022()
    make_hasoc2021()
    print("fixtures written to", ROOT)
