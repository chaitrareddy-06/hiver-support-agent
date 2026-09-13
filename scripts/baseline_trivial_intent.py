"""
Trivial baseline #1: majority-class intent predictor.

Predicts the single most frequent intent for every message, regardless of
content. This is the floor any real classifier must beat -- if our LLM
classifier can't meaningfully outscore "always guess Delivery Delay," it
isn't adding value.

Note on scope: this covers the INTENT classification baseline only. A
trivial baseline for the auto-handle/escalate decision (e.g. "always
auto-handle") is deliberately deferred to Phase 4, once we have actual
ground-truth escalate/auto-handle labels to score against. Right now the
golden set only has `requests_escalation` (a phrase-match feature), not a
labeled correct decision -- scoring a baseline against a feature instead of
a real label would be misleading, so we're not doing that yet.

Evaluated against: data/golden_set_labeled.csv (the only place we have real
human-confirmed intent ground truth).
"""

import pandas as pd

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    n = len(df)

    intent_counts = df["intent_confirmed"].value_counts()
    majority_intent = intent_counts.idxmax()
    majority_count = intent_counts.max()

    print(f"Golden set size: {n}")
    print("\nIntent distribution (ground truth):")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} ({count / n:.1%})")

    print(f"\nMajority-class baseline predicts: '{majority_intent}' for every row")

    predictions = pd.Series([majority_intent] * n, index=df.index)
    correct = (predictions == df["intent_confirmed"])
    overall_accuracy = correct.mean()

    print(f"\nOverall accuracy: {overall_accuracy:.1%} ({correct.sum()}/{n})")

    print("\nPer-class accuracy (recall) under this baseline:")
    for intent in intent_counts.index:
        mask = df["intent_confirmed"] == intent
        class_correct = (predictions[mask] == df.loc[mask, "intent_confirmed"]).sum()
        class_total = mask.sum()
        print(f"  {intent}: {class_correct}/{class_total} ({class_correct / class_total:.1%})")


if __name__ == "__main__":
    main()