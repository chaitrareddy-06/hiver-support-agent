import pandas as pd

# Load the full dataset
df = pd.read_csv("data/twcs.csv")

print("Total rows:", len(df))
print("\nColumns:", df.columns.tolist())

# Top accounts by tweet volume (to confirm AmazonHelp is well represented)
print("\nTop 20 authors by volume:")
print(df["author_id"].value_counts().head(20))

# Basic look at AmazonHelp specifically
amazon = df[df["author_id"] == "AmazonHelp"]
print("\nAmazonHelp tweet count:", len(amazon))
print("\nSample AmazonHelp rows:")
print(amazon.head(5)[["author_id", "text", "inbound", "in_response_to_tweet_id"]])