import pandas as pd

df = pd.read_csv("data/amazon_threads.csv")
print(f"Total pairs: {len(df)}")

# Random sample for manual inspection
sample = df.sample(n=100, random_state=42)

for i, row in sample.iterrows():
    print(f"\n--- Row {i} ---")
    print(f"Customer: {row['customer_text']}")
    print(f"Amazon:   {row['amazon_reply_text']}")