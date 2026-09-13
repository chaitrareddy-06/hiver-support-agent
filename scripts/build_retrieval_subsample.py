"""
Build the working/retrieval subsample used by the classifier and retrieval
system in later phases.

Source: data/amazon_threads.csv (123,569 cleaned English customer<->AmazonHelp
        reply pairs, built in Phase 1)
Output: data/retrieval_subsample.csv (~4,000 rows)

Why exclude golden set rows:
The golden set (data/golden_set_labeled.csv) is our fixed evaluation "answer
key." If those same threads also lived in the retrieval corpus, the system
could retrieve the *exact* original reply for a golden-set question and get
credit for "grounding" that isn't real generalization -- it would just be
looking up the answer key. Excluding them keeps the eval honest.

Why 4,000 rows:
Big enough to give reasonable retrieval coverage across all 7 intents
(including rarer ones like Marketplace/Seller), small enough to keep
retrieval fast and the pipeline runnable in under 15 minutes on a laptop,
per the assignment's "we will not run your code on the full dataset" rule.
"""

import pandas as pd

RANDOM_STATE = 42
SUBSAMPLE_SIZE = 4000

FULL_DATA_PATH = "data/amazon_threads.csv"
GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
OUTPUT_PATH = "data/retrieval_subsample.csv"


def main():
    print(f"Loading full dataset from {FULL_DATA_PATH} ...")
    full_df = pd.read_csv(FULL_DATA_PATH)
    print(f"  {len(full_df):,} total cleaned pairs")

    print(f"Loading golden set from {GOLDEN_SET_PATH} ...")
    golden_df = pd.read_csv(GOLDEN_SET_PATH)
    golden_ids = set(golden_df["customer_tweet_id"])
    print(f"  {len(golden_ids):,} golden set ids to exclude")

    # Exclude any row that's part of the golden set to avoid retrieval leakage
    before = len(full_df)
    eligible_df = full_df[~full_df["customer_tweet_id"].isin(golden_ids)].copy()
    excluded = before - len(eligible_df)
    print(f"  Excluded {excluded} rows that overlap with the golden set")
    print(f"  {len(eligible_df):,} rows eligible for sampling")

    if len(eligible_df) < SUBSAMPLE_SIZE:
        raise ValueError(
            f"Not enough eligible rows ({len(eligible_df)}) to sample "
            f"{SUBSAMPLE_SIZE}. Reduce SUBSAMPLE_SIZE or check inputs."
        )

    subsample_df = eligible_df.sample(n=SUBSAMPLE_SIZE, random_state=RANDOM_STATE)
    subsample_df = subsample_df.reset_index(drop=True)

    subsample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(subsample_df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()