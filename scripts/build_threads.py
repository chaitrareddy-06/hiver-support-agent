import pandas as pd
from langdetect import detect, LangDetectException

# Load the full dataset
print("Loading dataset...")
df = pd.read_csv("data/twcs.csv")
print(f"Total rows: {len(df)}")

# Split into inbound (customer) and outbound (brand) tweets
inbound = df[df["inbound"] == True].copy()
outbound = df[df["inbound"] == False].copy()

# Filter outbound tweets to AmazonHelp only
amazon_replies = outbound[outbound["author_id"] == "AmazonHelp"].copy()
print(f"AmazonHelp replies (all languages): {len(amazon_replies)}")

# Real language detection (replaces ASCII-ratio heuristic, which only
# caught non-Latin scripts and missed German/French/Spanish etc.)
def detect_lang(text):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

print("Detecting language of AmazonHelp replies (this takes a few minutes)...")
amazon_replies["detected_lang"] = amazon_replies["text"].apply(detect_lang)

lang_counts = amazon_replies["detected_lang"].value_counts()
print("\nTop detected languages in AmazonHelp replies:")
print(lang_counts.head(10))

amazon_replies_en = amazon_replies[amazon_replies["detected_lang"] == "en"].copy()
print(f"\nAmazonHelp replies (detected English): {len(amazon_replies_en)}")

# Join each English AmazonHelp reply back to the customer tweet it replied to
merged = amazon_replies_en.merge(
    inbound,
    left_on="in_response_to_tweet_id",
    right_on="tweet_id",
    suffixes=("_amazon", "_customer")
)
print(f"Matched pairs before customer-language check: {len(merged)}")

# Also check the customer tweet itself is English (a German customer
# tweet can still get an English boilerplate reply, which we don't want
# in an English-only training/eval set)
print("Detecting language of matched customer tweets...")
merged["customer_lang"] = merged["text_customer"].apply(detect_lang)
merged_en = merged[merged["customer_lang"] == "en"].copy()
print(f"Matched pairs (both sides detected English): {len(merged_en)}")

# Build clean paired dataset
threads = pd.DataFrame({
    "customer_tweet_id": merged_en["tweet_id_customer"],
    "customer_text": merged_en["text_customer"],
    "amazon_reply_id": merged_en["tweet_id_amazon"],
    "amazon_reply_text": merged_en["text_amazon"],
    "created_at": merged_en["created_at_amazon"]
})

# Save to CSV
output_path = "data/amazon_threads.csv"
threads.to_csv(output_path, index=False)
print(f"\nSaved {len(threads)} matched pairs to {output_path}")

# Show a few sample rows
print("\n--- Sample rows ---")
for i, row in threads.head(5).iterrows():
    print(f"\n[Pair {i}]")
    print(f"Customer: {row['customer_text'][:200]}")
    print(f"Amazon:   {row['amazon_reply_text'][:200]}")