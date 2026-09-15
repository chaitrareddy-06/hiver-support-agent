"""
LLM-as-judge: scores reply quality using the SAME rubric as the human-labeled
golden set (Relevant / Concrete-Actionable / Resolves-Forward, each Yes/No,
deriving Overall Quality: all Yes=Good, one No=Acceptable, 2-3 No=Poor).

Scores BOTH:
1. The original historical AmazonHelp reply (amazon_reply_text) -- all 200
   rows. This is what lets us measure judge-vs-human agreement: we already
   have human labels for these exact replies in golden_set_labeled.csv.
2. The AI-drafted grounded reply (drafted_reply) -- however many rows have
   one so far (currently 153/200, rest pending Groq quota reset).

This gives a direct, apples-to-apples comparison: how does our AI's reply
quality compare to the real human agent's, on the exact same messages,
scored by the exact same rubric?

Usage:
    python scripts/llm_judge.py        # all rows
    python scripts/llm_judge.py 10     # test on first 10 rows
"""

import os
import sys
import time
import json

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-120b"

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
GROUNDED_REPLIES_PATH = "data/grounded_replies.csv"
OUTPUT_PATH = "data/judge_scores.csv"

BATCH_SIZE = 3
MAX_RETRIES = 4
SLEEP_BETWEEN_BATCHES = 1

RUBRIC_INSTRUCTIONS = (
    "You are judging the quality of a customer support reply using this exact rubric:\n"
    "- Relevant: does the EVERY part of the reply address what the customer actually asked/"
    "reported? Read the full reply carefully, sentence by sentence. If ANY part of the reply "
    "is irrelevant to, contradicts, or appears to be leftover/copy-pasted boilerplate from a "
    "different, unrelated conversation (e.g. a sentence about something the customer never "
    "mentioned, like privacy/personal-information policy in a wrong-item-shipped case), mark "
    "this as No -- even if the rest of the reply is otherwise on-topic. Do not give credit for "
    "a reply that is only partially relevant. (Yes/No)\n"
    "- Concrete/Actionable: does it give a specific action, link, or next step -- not vague "
    "filler like 'please contact support'? (Yes/No)\n"
    "- Resolves/Forward: does it either resolve the issue or clearly move it forward "
    "(e.g. asks for specific info needed to proceed)? (Yes/No)\n\n"
    "Respond with ONLY a JSON array, one object per item, in the same order given, with "
    "this exact shape:\n"
    '[{"index": 0, "relevant": "Yes", "concrete_actionable": "No", "resolves_forward": "Yes"}, ...]\n'
    "No extra commentary, no markdown fences -- JSON only."
)


def derive_overall(relevant, concrete, resolves):
    no_count = sum(1 for v in [relevant, concrete, resolves] if v == "No")
    if no_count == 0:
        return "Good"
    elif no_count == 1:
        return "Acceptable"
    else:
        return "Poor"


def build_batch_prompt(items):
    parts = []
    for i, (customer_text, reply_text) in enumerate(items):
        parts.append(
            f"Item {i}:\n"
            f"  Customer message: {customer_text}\n"
            f"  Reply to judge: {reply_text}\n"
        )
    return RUBRIC_INSTRUCTIONS, "\n".join(parts)


def judge_batch(client, items):
    system_prompt, user_prompt = build_batch_prompt(items)
    expected_count = len(items)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1500,
                temperature=0.0,
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
                if idx is not None:
                    result_map[idx] = {
                        "relevant": item.get("relevant"),
                        "concrete_actionable": item.get("concrete_actionable"),
                        "resolves_forward": item.get("resolves_forward"),
                    }

            if len(result_map) == expected_count:
                return result_map
            else:
                print(f"    Attempt {attempt}: got {len(result_map)}/{expected_count}, retrying...")

        except Exception as e:
            print(f"    Attempt {attempt} failed: {e}")

        time.sleep(2)

    print(f"    All {MAX_RETRIES} attempts failed for this batch. Leaving these rows blank.")
    return {}


def score_column(client, df, text_col, prefix):
    """Judges df[text_col] for every row, with resume support, batched, checkpointed."""
    n = len(df)

    existing = {}
    if os.path.exists(OUTPUT_PATH):
        prev = pd.read_csv(OUTPUT_PATH)
        for _, row in prev.iterrows():
            key = row["customer_tweet_id"]
            if pd.notna(row.get(f"{prefix}_overall_quality")):
                existing[key] = {
                    "relevant": row.get(f"{prefix}_relevant"),
                    "concrete_actionable": row.get(f"{prefix}_concrete_actionable"),
                    "resolves_forward": row.get(f"{prefix}_resolves_forward"),
                    "overall_quality": row.get(f"{prefix}_overall_quality"),
                }

    results = [existing.get(df.iloc[i]["customer_tweet_id"]) for i in range(n)]

    rows_to_judge = [i for i in range(n) if results[i] is None and pd.notna(df.iloc[i][text_col])]
    print(f"[{prefix}] {n - len(rows_to_judge)} already judged, {len(rows_to_judge)} to judge.")

    num_batches = (len(rows_to_judge) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(num_batches):
        batch_idx = rows_to_judge[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        items = [
            (df.iloc[i]["customer_text"], df.iloc[i][text_col])
            for i in batch_idx
        ]
        print(f"[{prefix}] Batch {b + 1}/{num_batches}...")
        batch_result = judge_batch(client, items)

        for local_idx, global_idx in enumerate(batch_idx):
            if local_idx in batch_result:
                r = batch_result[local_idx]
                overall = derive_overall(r["relevant"], r["concrete_actionable"], r["resolves_forward"])
                results[global_idx] = {**r, "overall_quality": overall}

        yield results  # allow caller to save progress after every batch

        if b < num_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)

    print("Loading golden set and grounded replies...")
    golden_df = pd.read_csv(GOLDEN_SET_PATH)
    replies_df = pd.read_csv(GROUNDED_REPLIES_PATH)

    merged = golden_df.merge(
        replies_df[["customer_tweet_id", "drafted_reply"]],
        on="customer_tweet_id",
        how="left",
    )
    if limit:
        merged = merged.head(limit).copy()
    merged = merged.reset_index(drop=True)

    def save(orig_results, draft_results):
        out = merged[["customer_tweet_id", "customer_text", "amazon_reply_text", "drafted_reply"]].copy()
        for prefix, results in [("orig", orig_results), ("draft", draft_results)]:
            out[f"{prefix}_relevant"] = [r["relevant"] if r else None for r in results]
            out[f"{prefix}_concrete_actionable"] = [r["concrete_actionable"] if r else None for r in results]
            out[f"{prefix}_resolves_forward"] = [r["resolves_forward"] if r else None for r in results]
            out[f"{prefix}_overall_quality"] = [r["overall_quality"] if r else None for r in results]
        out.to_csv(OUTPUT_PATH, index=False)
        return out

    orig_results = [None] * len(merged)
    draft_results = [None] * len(merged)

    print("\n--- Judging original historical replies ---")
    for progress in score_column(client, merged, "amazon_reply_text", "orig"):
        orig_results = progress
        save(orig_results, draft_results)

    print("\n--- Judging AI-drafted grounded replies ---")
    for progress in score_column(client, merged, "drafted_reply", "draft"):
        draft_results = progress
        save(orig_results, draft_results)

    final = save(orig_results, draft_results)

    print("\n=== SUMMARY ===")
    print("\nOriginal (human) reply quality:")
    print(final["orig_overall_quality"].value_counts())
    print("\nAI-drafted reply quality:")
    print(final["draft_overall_quality"].value_counts())
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
