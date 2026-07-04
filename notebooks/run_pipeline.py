"""Exercise the whole pipeline on fixtures and assert each stage works.
This proves the code runs end-to-end and produces a valid F1; only the real
numbers wait on the real data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_fixtures
from hinglish_hate import (
    load_bohra, load_hasoc2021, load_hasoc2022,
    build_corpus, filter_romanised, evaluate_within, evaluate_cross, summarise,
)

make_fixtures.make_bohra()
make_fixtures.make_hasoc2022()
make_fixtures.make_hasoc2021()
FX = Path(__file__).parent / "fixtures"

# 1. loaders ----------------------------------------------------------------
bohra = load_bohra(FX / "hate_speech.tsv")
h22 = load_hasoc2022(FX / "final.csv")
h21 = load_hasoc2021(FX / "hasoc2021" / "train")

for name, df in [("bohra2018", bohra), ("hasoc2021", h21), ("hasoc2022", h22)]:
    assert list(df.columns) == ["text", "label", "source", "script"], df.columns
    assert set(df["label"].unique()) <= {0, 1}
    assert (df["source"] == name).all()
    print(f"{name:11s} rows={len(df):4d}  hate%={df['label'].mean():.2f}  "
          f"scripts={dict(df['script'].value_counts())}")

# 2. unify + romanised filter ----------------------------------------------
corpus = build_corpus([bohra, h21, h22])
roman = filter_romanised(corpus, include_mixed=True)
print(f"\ncorpus={len(corpus)}  romanised={len(roman)}")
assert (roman["script"] != "mostly_devanagari").all(), "Devanagari leaked through filter"
assert len(roman) < len(corpus), "filter removed nothing - check script column"

# 3. baseline: within-Bohra (safety net) + cross (Bohra -> HASOC22) ---------
bohra_r = filter_romanised(bohra)
h22_r = filter_romanised(h22, include_mixed=True)

within = evaluate_within(bohra_r)
cross = evaluate_cross(bohra_r, h22_r)
print()
print(summarise("within Bohra", within))
print(summarise("Bohra -> HASOC22", cross))

for r in (within, cross):
    assert 0.0 <= r["macro_f1"] <= 1.0

print("\nOK - pipeline runs end to end and produces F1 on all stages.")
