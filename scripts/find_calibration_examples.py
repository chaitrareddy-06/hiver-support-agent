import pandas as pd

df = pd.read_csv("data/golden_set_labeled.csv")

# Filter out rows flagged as pipeline artifacts (multi-part fragments, mismatched pairs)
artifact_keywords = ["multi-part", "mismatch", "pairing"]

def is_artifact(notes):
    if pd.isna(notes):
        return False
    notes_lower = str(notes).lower()
    return any(kw in notes_lower for kw in artifact_keywords)

df["is_artifact"] = df["notes"].apply(is_artifact)

genuine_poor = df[(df["overall_quality"] == "Poor") & (~df["is_artifact"])]
print(f"Genuine (non-artifact) Poor examples: {len(genuine_poor)}")
print()
for _, row in genuine_poor.head(3).iterrows():
    print(f"Customer: {row['customer_text']}")
    print(f"Reply: {row['amazon_reply_text']}")
    print(f"Notes: {row.get('notes', '')}")
    print("---")