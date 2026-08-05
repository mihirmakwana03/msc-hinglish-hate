"""Romanisation variant generation for Hinglish surface forms.

Romanised Hindi has no fixed orthography, so one spoken word appears in many
written forms (kutta / kutte / kuttaa / kuta). This module encodes the regular
transliteration alternations as rewrite rules and expands a base term into its
plausible written variants.

Two uses:
  1. Day 9 lexicon: expand curated base terms into surface forms to match on.
  2. Objective 4 attack: perturb an input post into romanisation variants that
     preserve meaning for a human reader but may break a classifier.

The rules are deliberately conservative and reversible in spirit: each captures
an alternation that genuinely occurs in romanised Hindi, rather than arbitrary
character noise.
"""

from __future__ import annotations

import itertools
import re

# Each rule is (pattern, replacements). Applied to the lowercased base form.
# Ordered roughly from most to least common in observed romanisation.
VARIANT_RULES: list[tuple[str, tuple[str, ...]]] = [
    # long/short vowel ambiguity: the single biggest source of variation
    ("aa", ("a",)),
    ("a", ("aa",)),
    ("ee", ("i", "ii")),
    ("i", ("ee",)),
    ("oo", ("u", "uu")),
    ("u", ("oo",)),
    # aspiration written with or without h
    ("bh", ("b",)),
    ("dh", ("d",)),
    ("gh", ("g",)),
    ("kh", ("k",)),
    ("th", ("t",)),
    ("ph", ("f",)),
    ("f", ("ph",)),
    # retroflex/dental collapse
    ("d", ("dd",)),
    ("t", ("tt",)),
    # k/c/q alternation
    ("k", ("c", "q")),
    ("c", ("k",)),
    # sibilants
    ("sh", ("s",)),
    ("s", ("sh",)),
    # y/i glide
    ("y", ("i",)),
    # gemination (doubling) of medial consonants
    ("tt", ("t",)),
    ("dd", ("d",)),
    ("ll", ("l",)),
    ("nn", ("n",)),
]

# common inflectional endings in Hindi nouns/adjectives (kutta/kutte/kutti)
ENDING_SWAPS: dict[str, tuple[str, ...]] = {
    "a": ("e", "i", "ay"),
    "aa": ("e", "i"),
    "e": ("a", "i"),
    "i": ("a", "e"),
    "o": ("e", "a"),
}


def _apply_single_rules(term: str) -> set[str]:
    """Apply each rewrite rule once, independently, to the term."""
    out = set()
    for pattern, repls in VARIANT_RULES:
        if pattern not in term:
            continue
        for repl in repls:
            # replace each occurrence position independently
            for m in re.finditer(re.escape(pattern), term):
                cand = term[: m.start()] + repl + term[m.end():]
                if cand != term:
                    out.add(cand)
    return out


def _apply_ending_swaps(term: str) -> set[str]:
    out = set()
    for ending, repls in ENDING_SWAPS.items():
        if term.endswith(ending):
            stem = term[: -len(ending)]
            for r in repls:
                cand = stem + r
                if cand != term:
                    out.add(cand)
    return out


def _elongate(term: str) -> set[str]:
    """Vowel elongation for emphasis: bhai -> bhaai, kya -> kyaa."""
    out = set()
    for i, ch in enumerate(term):
        if ch in "aeiou":
            out.add(term[: i + 1] + ch + term[i + 1:])
    return out


def generate_variants(term: str, depth: int = 1, max_variants: int = 40) -> list[str]:
    """Expand a base term into romanisation variants.

    depth=1 applies one alternation at a time (safe, high precision).
    depth=2 composes two alternations (wider net, more noise).
    The base term is always included first.
    """
    term = str(term).strip().lower()
    if not term:
        return []
    seen = {term}
    frontier = {term}
    for _ in range(max(1, depth)):
        nxt = set()
        for t in frontier:
            nxt |= _apply_single_rules(t)
            nxt |= _apply_ending_swaps(t)
            nxt |= _elongate(t)
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if len(seen) >= max_variants:
            break
    ordered = [term] + sorted(v for v in seen if v != term)
    return ordered[:max_variants]


def expand_lexicon(terms, depth: int = 1, max_variants: int = 40) -> dict[str, list[str]]:
    """Map each base term to its generated variant list."""
    return {t: generate_variants(t, depth=depth, max_variants=max_variants) for t in terms}


def variant_column(terms, depth: int = 1, max_variants: int = 40) -> list[str]:
    """Pipe-separated variant strings, ready for the lexicon CSV `variants` column.

    Excludes the base term itself (it already lives in the `term` column).
    """
    rows = []
    for t in terms:
        vs = [v for v in generate_variants(t, depth=depth, max_variants=max_variants) if v != str(t).strip().lower()]
        rows.append("|".join(vs))
    return rows
