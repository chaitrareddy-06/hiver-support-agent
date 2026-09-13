import pandas as pd
import re
import sys
sys.path.append("scripts")
from build_golden_set import auto_tag  # reuse the same tagger

df = pd.read_csv("data/amazon_threads.csv")
df["rough_intent"] = df["customer_text"].apply(auto_tag)

general = df[df["rough_intent"] == "General Inquiry / Feedback"]
print(f"Total in General Inquiry bucket: {len(general)}")

sample = general.sample(n=25, random_state=7)
for i, row in sample.iterrows():
    print(f"\n[Row {i}]")
    print(f"Customer: {row['customer_text']}")