"""
Build a semantic embedding index over the retrieval/working subsample
(data/retrieval_subsample.csv), so we can later find the historical
customer messages most similar in MEANING (not just keywords) to a new
incoming message, and use their real Amazon replies as grounding when
drafting a new reply.

Model: sentence-transformers 'all-MiniLM-L6-v2' -- small, fast, well-suited
to short social-media-length text, and runs entirely on CPU in a
reasonable time for 4,000 rows. No API calls in this step, so no
rate-limit concerns -- this is pure local computation.

Output:
- data/retrieval_embeddings.npy -- one embedding vector per row, in the
  exact same row order as data/retrieval_subsample.csv, so row i's
  embedding corresponds to row i of that CSV.

Note: this .npy file is NOT committed to git (same reasoning as
retrieval_subsample.csv itself -- it's fully reproducible by rerunning
this script, so there's no need to version a derived binary artifact).
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/retrieval_subsample.csv"
EMBEDDINGS_OUTPUT_PATH = "data/retrieval_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df):,} rows from {INPUT_PATH}")

    print(f"Loading embedding model: {MODEL_NAME} (first run downloads it, ~80MB)...")
    model = SentenceTransformer(MODEL_NAME)

    texts = df["customer_text"].astype(str).tolist()
    print(f"Embedding {len(texts):,} customer messages (this may take a minute or two on CPU)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {EMBEDDINGS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()