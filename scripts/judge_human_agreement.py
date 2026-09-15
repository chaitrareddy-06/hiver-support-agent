"""
Judge-vs-human agreement check.

Compares the LLM judge's scores on the ORIGINAL historical AmazonHelp
replies (orig_overall_quality in judge_scores.csv) against the human
labels already collected for those same replies during golden-set
labeling (overall_quality in golden_set_labeled.csv).

Same replies, two raters (one human, one LLM) -- this is what lets us
report how much to trust the judge before using it to compare AI-drafted
vs human reply quality.

Usage:
    python scripts/judge_human_agreement.py
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
JUDGE_SCORES_PATH = "data/judge_scores.csv"


def main():
    golden_df = pd.read_csv(GOLDEN_SET_PATH)
    judge_df = pd.read_csv(JUDGE_SCORES_PATH)

    merged = golden_df.merge(
        judge_df[["customer_tweet_id", "orig_overall_quality"]],
        on="customer_tweet_id",
        how="inner",
    )

    human_labels = merged["overall_quality"]
    judge_labels = merged["orig_overall_quality"]

    valid = human_labels.notna() & judge_labels.notna()
    human_labels = human_labels[valid]
    judge_labels = judge_labels[valid]

    n = len(human_labels)
    exact_match = (human_labels == judge_labels).sum()
    accuracy = accuracy_score(human_labels, judge_labels)
    kappa = cohen_kappa_score(human_labels, judge_labels)

    print(f"Rows compared: {n}")
    print(f"Exact agreement: {exact_match}/{n} ({accuracy:.1%})")
    print(f"Cohen's kappa: {kappa:.3f}")
    print()
    print("Kappa interpretation guide: <0=Poor, 0.01-0.20=Slight, 0.21-0.40=Fair,")
    print("0.41-0.60=Moderate, 0.61-0.80=Substantial, 0.81-1.00=Almost Perfect")
    print()

    labels_order = ["Good", "Acceptable", "Poor"]
    cm = confusion_matrix(human_labels, judge_labels, labels=labels_order)
    print("Confusion matrix (rows=human label, cols=judge label):")
    print(f"{'':12}" + "".join(f"{l:>12}" for l in labels_order))
    for i, row_label in enumerate(labels_order):
        print(f"{row_label:12}" + "".join(f"{cm[i][j]:>12}" for j in range(len(labels_order))))


if __name__ == "__main__":
    main()