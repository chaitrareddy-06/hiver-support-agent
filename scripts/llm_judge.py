"""
LLM-as-judge: scores reply quality using the SAME rubric as the human-labeled
golden set (Relevant / Concrete-Actionable / Resolves-Forward, each Yes/No,
deriving Overall Quality: all Yes=Good, one No=Acceptable, 2-3 No=Poor).

Scores BOTH:
1. The original historical AmazonHelp reply (amazon_reply_text) -- all 200
   rows. This is what lets us measure judge-vs-human agreement: we already
   have human labels for these exact replies in golden_set_labeled.csv.
2. The AI-drafted grounded reply (drafted_reply) -- all 200 rows.

This gives a direct, apples-to-apples comparison: how does our AI's reply
quality compare to the real human agent's, on the exact same messages,
scored by the exact same rubric?

Judge model deliberately different from the drafting model (drafting used
Groq's gpt-oss-20b/120b; judge uses qwen/qwen3.8-27b) to avoid self-grading
bias -- a recognized good practice in LLM-as-judge setups, and also a
practical necessity given daily token quota limits hit on the other models.

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
MODEL_NAME = "qwen/qwen3.8-27b"

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
GROUNDED_REPLIES_PATH = "data/grounded_replies.csv"
OUTPUT_PATH = "data/judge_scores.csv"

BATCH_SIZE = 3
MAX_RETRIES = 4
SLEEP_BETWEEN_BATCHES = 1

RUBRIC_INSTRUCTIONS = (
    "You are judging the quality of a customer support reply using this exact rubric. "
    "Judge each of the three criteria INDEPENDENTLY and STRICTLY -- do not let your overall "
    "impression of the reply's tone or politeness influence any individual criterion. A "
    "reply can sound polite and well-written while still failing on Relevant, Concrete, "
    "or Resolves -- score what the reply actually DOES, not how it sounds.\n\n"
    "- Relevant: does EVERY part of the reply address what the customer actually asked/"
    "reported? Read the full reply sentence by sentence. If ANY part is irrelevant to, "
    "contradicts, or appears to be leftover/copy-pasted boilerplate from a different, "
    "unrelated conversation, mark this as No -- even if the rest is otherwise on-topic. "
    "(Yes/No)\n"
    "- Concrete/Actionable: does it give a SPECIFIC action, link, order number reference, "
    "or next step? Generic phrases like 'please contact our support team', 'we'll look "
    "into it', 'reach out to us here', or 'I'd like to help' with no specific instruction "
    "attached do NOT count as concrete -- mark No for these even if a link is included, "
    "unless the link is paired with a specific instruction of what to do with it. (Yes/No)\n"
    "- Resolves/Forward: does it either fully resolve the issue, or ask for a SPECIFIC "
    "piece of information needed to proceed (e.g. 'share your order number')? A vague "
    "acknowledgment like 'we're looking into it' or 'thanks for reaching out' without a "
    "concrete next step does NOT count -- mark No. (Yes/No)\n\n"
    "IMPORTANT: Be strict. Most real customer support replies are mediocre, not excellent. "
    "If you are judging most replies as fully Yes/Yes/Yes ('Good'), you are being too "
    "lenient -- go back and check each criterion again against the literal definitions "
    "above, not your overall impression.\n\n"
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


def load_existing(df, prefix):
    """Loads already-judged results for one column (orig/draft) from disk, if present."""
    n = len(df)
    results = [None] * n

    if not os.path.exists(OUTPUT_PATH):
        return results

    prev = pd.read_csv(OUTPUT_PATH)
    lookup = {}
    for _, row in prev.iterrows():
        key = row["customer_tweet_id"]
        if pd.notna(row.get(f"{prefix}_overall_quality")):
            lookup[key] = {
                "relevant": row.get(f"{prefix}_relevant"),
                "concrete_actionable": row.get(f"{prefix}_concrete_actionable"),
                "resolves_forward": row.get(f"{prefix}_resolves_forward"),
                "overall_quality": row.get(f"{prefix}_overall_quality"),
            }

    for i in range(n):
        key = df.iloc[i]["customer_tweet_id"]
        if key in lookup:
            results[i] = lookup[key]

    return results


def score_column(client, df, text_col, prefix, results):
    """
    Judges df[text_col] for every row not already present in `results`,
    batched, checkpointed. Mutates and yields `results` in place so the
    caller always has an accurate, up-to-date list to save, even for rows
    this call didn't touch.
    """
    n = len(df)
    rows_to_judge = [i for i in range(n) if results[i] is None and pd.notna(df.iloc[i][text_col])]
    print(f"[{prefix}] {n - len(rows_to_judge)} already judged, {len(rows_to_judge)} to judge.")

    if not rows_to_judge:
        yield results
        return

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
    client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL, timeout=30.0)

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

    # Load BOTH columns' existing progress up front, before any judging
    # starts, so saving progress on one column never overwrites the other
    # with an empty placeholder.
    orig_results = load_existing(merged, "orig")
    draft_results = load_existing(merged, "draft")

    def save():
        out = merged[["customer_tweet_id", "customer_text", "amazon_reply_text", "drafted_reply"]].copy()
        for prefix, results in [("orig", orig_results), ("draft", draft_results)]:
            out[f"{prefix}_relevant"] = [r["relevant"] if r else None for r in results]
            out[f"{prefix}_concrete_actionable"] = [r["concrete_actionable"] if r else None for r in results]
            out[f"{prefix}_resolves_forward"] = [r["resolves_forward"] if r else None for r in results]
            out[f"{prefix}_overall_quality"] = [r["overall_quality"] if r else None for r in results]
        out.to_csv(OUTPUT_PATH, index=False)
        return out

    print("\n--- Judging original historical replies ---")
    for _ in score_column(client, merged, "amazon_reply_text", "orig", orig_results):
        save()

    print("\n--- Judging AI-drafted grounded replies ---")
    for _ in score_column(client, merged, "drafted_reply", "draft", draft_results):
        save()

    final = save()

    print("\n=== SUMMARY ===")
    print("\nOriginal (human) reply quality:")
    print(final["orig_overall_quality"].value_counts())
    print("\nAI-drafted reply quality:")
    print(final["draft_overall_quality"].value_counts())
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()