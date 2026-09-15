"""
Escalation policy: decides AUTO-HANDLE vs ESCALATE for each message, with a
stated reason.

Combines two signals, in priority order:
1. requests_escalation (phrase-based signal from Phase 1/2) -- if the
   customer explicitly asked for a human, we never override that.
2. top1_similarity_score (from Phase 4c's retrieval step) -- if even our
   best historical match is weak, the grounded reply is likely unreliable,
   so we escalate as a safety net rather than auto-handling with low
   confidence.

Threshold (SIMILARITY_THRESHOLD) is set at the 20th percentile of
top1_similarity_score across the golden set (~0.65) -- i.e. we escalate
the bottom ~20% of messages by grounding strength. This is a data-driven
choice, not an arbitrary round number.

If neither signal fires, the message is auto-handled.

Usage:
    python scripts/escalation_policy.py
"""

import pandas as pd

GROUNDED_REPLIES_PATH = "data/grounded_replies.csv"
GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
OUTPUT_PATH = "data/escalation_decisions.csv"

SIMILARITY_THRESHOLD = 0.65


def decide(row):
    if row.get("requests_escalation") == True or row.get("requests_escalation") == "True":
        return "ESCALATE", "Customer explicitly requested escalation."
    if pd.notna(row.get("top1_similarity_score")) and row["top1_similarity_score"] < SIMILARITY_THRESHOLD:
        return "ESCALATE", f"Weak historical grounding (top1_similarity_score={row['top1_similarity_score']:.3f} < {SIMILARITY_THRESHOLD})."
    return "AUTO-HANDLE", "Clear intent match with reasonably strong historical precedent."


def main():
    golden_df = pd.read_csv(GOLDEN_SET_PATH)
    replies_df = pd.read_csv(GROUNDED_REPLIES_PATH)

    merged = golden_df.merge(
        replies_df[["customer_tweet_id", "top1_similarity_score"]],
        on="customer_tweet_id",
        how="left",
    )

    decisions = merged.apply(decide, axis=1)
    merged["decision"] = [d[0] for d in decisions]
    merged["decision_reason"] = [d[1] for d in decisions]

    output_cols = [
        "customer_tweet_id", "customer_text", "rough_intent",
        "requests_escalation", "top1_similarity_score",
        "decision", "decision_reason",
    ]
    merged[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(merged)}")
    print(merged["decision"].value_counts())
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()