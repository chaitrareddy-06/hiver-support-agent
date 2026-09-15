"""
Retrieval-grounded reply drafting.

For each customer message, this script:
1. Embeds the message (same model as the retrieval index).
2. Finds the TOP_K most semantically similar historical customer messages
   in the retrieval subsample (data/retrieval_subsample.csv + its
   embeddings), using cosine similarity.
3. Pulls those historical messages' REAL Amazon replies as grounding
   examples -- concrete evidence of how AmazonHelp actually resolved
   similar issues before.
4. Asks the LLM (Groq) to draft a new reply for the current message,
   informed by those real historical resolutions -- not generic
   "I'm sorry to hear that, please DM us" boilerplate.

This is what makes the reply "grounded" rather than pure LLM invention:
the model is shown real precedent and asked to follow the same pattern
(tone, level of specificity, what info AmazonHelp actually asks for/gives),
not just told to "be helpful."

The top-1 similarity score is saved alongside each drafted reply -- low
scores flag rows where the "grounding" was only loosely related to the
actual message, which matters for honest reporting later.

Evaluated against: golden set customer messages (the same 200 rows used
throughout), so the resulting replies can be scored by the eval harness
(Phase 5) using the same rubric as the human labels.

Usage:
    python scripts/generate_grounded_replies.py        # all 200 rows
    python scripts/generate_grounded_replies.py 10      # test on first 10 rows
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-120b"

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
RETRIEVAL_CORPUS_PATH = "data/retrieval_subsample.csv"
RETRIEVAL_EMBEDDINGS_PATH = "data/retrieval_embeddings.npy"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_PATH = "data/grounded_replies.csv"

TOP_K = 3
BATCH_SIZE = 3
MAX_RETRIES = 4
SLEEP_BETWEEN_BATCHES = 1


def cosine_similarity_matrix(query_vecs, corpus_vecs):
    query_norm = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    corpus_norm = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    return query_norm @ corpus_norm.T


def build_batch_prompt(batch_rows, retrieved_examples_per_row):
    system_prompt = (
        "You are drafting customer support replies for AmazonHelp on behalf of Amazon. "
        "For each customer message below, you are shown up to 3 real historical examples "
        "of similar customer messages and how AmazonHelp actually replied to them. "
        "Use these as grounding: match AmazonHelp's real tone and, "
        "CRITICAL: match the SPECIFICITY of the grounding examples, not just their tone. "
        "If a grounding example gives a concrete action (e.g. \"follow the prompts for a "
        "return label,\" \"share your order number,\" \"check your spam folder\"), your "
        "reply must give an equally concrete action for the new message -- not a vaguer, "
        "generic version of it. Do NOT default to generic filler like \"please contact our "
        "support team\" or \"reach out to us\" if the grounding examples show AmazonHelp "
        "giving more specific guidance than that. Only be as vague as the real examples "
        "actually are -- never vaguer.\n\n"
        "Respond with ONLY a JSON array, one object per message, in the same order as "
        "given, with this exact shape:\n"
        '[{"index": 0, "reply": "drafted reply text"}, ...]\n'
        "No extra commentary, no markdown fences -- JSON only."
    )

    user_parts = []
    for i, (row, examples) in enumerate(zip(batch_rows, retrieved_examples_per_row)):
        example_text = "\n".join(
            f"  Example {j+1}:\n"
            f"    Customer: {ex_customer}\n"
            f"    AmazonHelp reply: {ex_reply}"
            for j, (ex_customer, ex_reply) in enumerate(examples)
        )
        user_parts.append(
            f"Message {i}:\n"
            f"  New customer message: {row['customer_text']}\n"
            f"  Grounding examples (real past AmazonHelp resolutions for similar issues):\n"
            f"{example_text}\n"
        )

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def generate_batch(client, batch_rows, retrieved_examples_per_row):
    system_prompt, user_prompt = build_batch_prompt(batch_rows, retrieved_examples_per_row)
    expected_count = len(batch_rows)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=3000,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)

            result_map = {}
            for item in parsed:
                idx = item.get("index")
                reply = item.get("reply")
                if idx is not None and reply:
                    result_map[idx] = reply

            if len(result_map) == expected_count:
                return result_map
            else:
                print(
                    f"    Attempt {attempt}: got {len(result_map)}/{expected_count} "
                    f"replies, expected all. Retrying..."
                )

        except Exception as e:
            print(f"    Attempt {attempt} failed: {e}")

        time.sleep(2)

    print(f"    All {MAX_RETRIES} attempts failed for this batch. Leaving these rows blank.")
    return {}


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)

    print("Loading golden set...")
    golden_df = pd.read_csv(GOLDEN_SET_PATH)
    if limit:
        golden_df = golden_df.head(limit).copy()
    golden_df = golden_df.reset_index(drop=True)
    n = len(golden_df)

    # Resume support: load any already-completed replies from a previous run
    # so we don't re-call the LLM on rows that already succeeded.
    existing_replies = {}
    if os.path.exists(OUTPUT_PATH):
        prev_df = pd.read_csv(OUTPUT_PATH)
        for _, row in prev_df.iterrows():
            if pd.notna(row.get("drafted_reply")):
                existing_replies[row["customer_tweet_id"]] = row["drafted_reply"]
        print(f"Found {len(existing_replies)} already-completed replies from a previous run. Skipping those.")

    print("Loading retrieval corpus and embeddings...")
    corpus_df = pd.read_csv(RETRIEVAL_CORPUS_PATH)
    corpus_embeddings = np.load(RETRIEVAL_EMBEDDINGS_PATH)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding {n} golden-set customer messages...")
    query_texts = golden_df["customer_text"].astype(str).tolist()
    query_embeddings = embed_model.encode(query_texts, show_progress_bar=True, batch_size=64)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)

    print(f"Finding top-{TOP_K} similar historical threads for each message...")
    sims = cosine_similarity_matrix(query_embeddings, corpus_embeddings)
    top_k_indices = np.argsort(-sims, axis=1)[:, :TOP_K]

    retrieved_examples_all = []
    retrieved_top1_scores = []
    for i in range(n):
        examples = []
        for corpus_idx in top_k_indices[i]:
            ex_customer = corpus_df.iloc[corpus_idx]["customer_text"]
            ex_reply = corpus_df.iloc[corpus_idx]["amazon_reply_text"]
            examples.append((ex_customer, ex_reply))
        retrieved_examples_all.append(examples)
        retrieved_top1_scores.append(float(sims[i, top_k_indices[i][0]]))

    num_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nGenerating grounded replies for {n} rows in {num_batches} batches of up to {BATCH_SIZE}...")

    all_replies = [
        existing_replies.get(golden_df.iloc[i]["customer_tweet_id"])
        for i in range(n)
    ]
    output_cols = [
        "customer_tweet_id", "customer_text", "amazon_reply_text",
        "intent_confirmed", "drafted_reply", "top1_similarity_score",
    ]

    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch_rows = [golden_df.iloc[i] for i in range(start, end)]
        batch_examples = retrieved_examples_all[start:end]

        already_done = all(all_replies[start + i] is not None for i in range(end - start))
        if already_done:
            print(f"\nBatch {batch_num + 1}/{num_batches} (rows {start}-{end - 1}) -- all rows already done, skipping.")
        else:
            print(f"\nBatch {batch_num + 1}/{num_batches} (rows {start}-{end - 1})...")
            result_map = generate_batch(client, batch_rows, batch_examples)

            for local_idx, reply in result_map.items():
                all_replies[start + local_idx] = reply

        golden_df["drafted_reply"] = all_replies
        golden_df["top1_similarity_score"] = retrieved_top1_scores
        golden_df[output_cols].to_csv(OUTPUT_PATH, index=False)

        if batch_num < num_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    missing = sum(1 for r in all_replies if r is None)
    print(f"\nDone. {n - missing}/{n} replies generated successfully.")
    if missing:
        print(f"WARNING: {missing} rows have no drafted reply (batch(es) failed after all retries).")
        print("These rows are blank in the output CSV -- rerun just those rows if needed.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()