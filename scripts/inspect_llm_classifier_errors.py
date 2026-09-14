"""
Inspect exactly which golden-set rows the LLM classifier got wrong, and look
for patterns -- which intents get confused with which, and what the
messages themselves look like. This is raw material for the report's
"top 5 failure modes with real examples" section.
"""

import pandas as pd

PREDICTIONS_PATH = "data/llm_classifier_predictions.csv"

pd.set_option("display.max_colwidth", 80)


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    wrong = df[~df["correct"]].copy()

    print(f"Total wrong: {len(wrong)} / {len(df)} ({len(wrong) / len(df):.1%})\n")

    print("=== Confusion pairs (true -> predicted), most common first ===")
    confusion_pairs = (
        wrong.groupby(["intent_confirmed", "predicted_intent"])
        .size()
        .sort_values(ascending=False)
    )
    for (true_intent, pred_intent), count in confusion_pairs.items():
        print(f"  {count:3d}x  {true_intent}  -->  {pred_intent}")

    print("\n=== All wrong rows (true intent | predicted intent | message) ===")
    for _, row in wrong.iterrows():
        print(f"\n[{row['customer_tweet_id']}]")
        print(f"  TRUE:      {row['intent_confirmed']}")
        print(f"  PREDICTED: {row['predicted_intent']}")
        print(f"  MESSAGE:   {row['customer_text']}")


if __name__ == "__main__":
    main()