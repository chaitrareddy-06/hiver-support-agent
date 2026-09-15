import pandas as pd

df = pd.read_csv("data/grounded_replies.csv")
missing = df[df["drafted_reply"].isna()]

print(f"Missing rows: {len(missing)}")
print()
print("customer_tweet_id, intent_confirmed, top1_similarity_score:")
print(missing[["customer_tweet_id", "intent_confirmed", "top1_similarity_score"]].to_string(index=False))
print()
print("Missing count by intent:")
print(missing["intent_confirmed"].value_counts())