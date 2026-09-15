import pandas as pd

df = pd.read_csv("data/grounded_replies.csv")

print("top1_similarity_score distribution:")
print(df["top1_similarity_score"].describe())
print()
print("Percentiles:")
for p in [10, 20, 25, 30, 40, 50]:
    print(f"  {p}th percentile: {df['top1_similarity_score'].quantile(p/100):.3f}")