"""Adversarial attacks on Hinglish hate-speech classifiers (Objective 4).

`variants.py` perturbs a single term. This module perturbs a whole post, which is
what an evader actually does.

Two attacks, so the novel one has something to be measured against:

1. `romanisation_attack` - the contribution. Rewrites tokens into plausible
   romanisation variants using the rules in `variants.py`. Every output is a
   spelling a Hindi speaker could produce naturally and read without difficulty.

2. `character_attack` - the comparison baseline, in the spirit of AdvCodeMix.
   Random character swaps, deletions and repetitions. Not linguistically
   motivated, so it is the control that shows whether the romanisation rules add
   anything beyond generic noise.

The point of the pairing: if both attacks degrade a model equally, the
romanisation rules are doing nothing special. If the romanisation attack
degrades it more *at the same edit rate*, the linguistic structure matters.

Usage
-----
    from hinglish_hate.attack import romanisation_attack, attack_corpus

    romanisation_attack("tum log sab chutiye ho", rate=0.5, seed=42)
    attacked = attack_corpus(df, romanisation_attack, rate=0.5)
"""

from __future__ import annotations

import random
import re
from typing import Callable

import pandas as pd

from .variants import generate_variants

# Words never worth perturbing: too short to vary meaningfully, or structural.
_MIN_LEN = 3
_SKIP = re.compile(r"^(?:https?://|@|#|\d+$)")

# Keyboard-adjacency for the character baseline. Rough QWERTY neighbours.
_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _tokenise(text: str) -> list[str]:
    """Split on whitespace, keeping tokens intact so they can be rejoined."""
    return str(text).split()


def _is_perturbable(token: str) -> bool:
    return len(token) >= _MIN_LEN and not _SKIP.match(token) and token.isascii()


def romanisation_attack(text: str, rate: float = 0.5, seed: int | None = None,
                        depth: int = 1) -> str:
    """Rewrite a proportion of tokens into romanisation variants.

    Parameters
    ----------
    text : the post to perturb
    rate : proportion of eligible tokens to rewrite, 0.0 to 1.0
    seed : per-call RNG seed, so an experiment is reproducible
    depth : passed through to generate_variants; 1 applies one alternation

    Returns the perturbed post. Tokens with no available variant are left alone,
    so the realised edit rate can be below `rate`; `attack_corpus` records it.
    """
    rng = random.Random(seed)
    tokens = _tokenise(text)
    idx = [i for i, t in enumerate(tokens) if _is_perturbable(t)]
    if not idx:
        return text

    n = max(1, int(round(rate * len(idx))))
    chosen = rng.sample(idx, min(n, len(idx)))

    for i in chosen:
        base = tokens[i].lower()
        variants = [v for v in generate_variants(base, depth=depth) if v != base]
        if variants:
            tokens[i] = rng.choice(variants)
    return " ".join(tokens)


def character_attack(text: str, rate: float = 0.5, seed: int | None = None) -> str:
    """AdvCodeMix-style baseline: random character-level noise.

    One edit per selected token, chosen from swap / neighbour-substitute /
    delete / repeat. Deliberately *not* linguistically motivated - that is the
    point of it as a control.
    """
    rng = random.Random(seed)
    tokens = _tokenise(text)
    idx = [i for i, t in enumerate(tokens) if _is_perturbable(t)]
    if not idx:
        return text

    n = max(1, int(round(rate * len(idx))))
    for i in rng.sample(idx, min(n, len(idx))):
        w = list(tokens[i])
        op = rng.choice(["swap", "sub", "delete", "repeat"])
        p = rng.randrange(len(w))

        if op == "swap" and len(w) > 2:
            q = min(p, len(w) - 2)
            w[q], w[q + 1] = w[q + 1], w[q]
        elif op == "sub":
            nb = _NEIGHBOURS.get(w[p].lower())
            if nb:
                w[p] = rng.choice(nb)
        elif op == "delete" and len(w) > _MIN_LEN:
            del w[p]
        elif op == "repeat":
            w.insert(p, w[p])
        tokens[i] = "".join(w)
    return " ".join(tokens)


def attack_corpus(df: pd.DataFrame, attack_fn: Callable[..., str],
                  rate: float = 0.5, seed: int = 42,
                  text_col: str = "text") -> pd.DataFrame:
    """Apply an attack to every row, with a per-row seed for reproducibility.

    Returns a copy with the original text kept in `text_clean`, plus
    `changed` (was the row modified at all) and `token_change_rate` (the
    realised proportion of tokens that differ). Report the realised rate, not
    the requested one - they are not the same, and the difference matters when
    comparing two attacks at "the same" edit rate.
    """
    out = df.copy()
    out["text_clean"] = out[text_col]

    attacked, realised = [], []
    for i, original in enumerate(out[text_col]):
        adv = attack_fn(original, rate=rate, seed=seed + i)
        attacked.append(adv)

        a, b = str(original).split(), str(adv).split()
        if not a:
            realised.append(0.0)
            continue
        diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        realised.append(diff / len(a))

    out[text_col] = attacked
    out["changed"] = out["text_clean"] != out[text_col]
    out["token_change_rate"] = realised
    return out


def attack_report(clean_score: float, attacked_score: float, label: str = "") -> dict:
    """Absolute and relative degradation under attack.

    Relative drop is the more comparable figure across models: a model starting
    at 0.66 and one starting at 0.51 losing the same 0.05 have lost very
    different proportions of what they had.
    """
    drop = clean_score - attacked_score
    return {
        "attack": label,
        "clean": round(clean_score, 4),
        "attacked": round(attacked_score, 4),
        "absolute_drop": round(drop, 4),
        "relative_drop_pct": round(100 * drop / clean_score, 2) if clean_score else None,
    }
