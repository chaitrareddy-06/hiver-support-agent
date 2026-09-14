"""
LLM-based intent classifier -- now using Groq (Llama 3.3 70B) instead of
Gemini.

Why the switch: Gemini's free tier was discovered mid-build to have been
recently cut to 20 requests/day across all models (confirmed via a live
429 RESOURCE_EXHAUSTED error and corroborated by other developers hitting
the same wall). That's unworkable for a project needing hundreds of calls
across classification, reply generation, and LLM-as-judge scoring. Groq's
free tier (no card required) gives ~30 requests/minute and ~1,000
requests/day on llama-3.3-70b-versatile -- 50x the daily headroom.

Still batching multiple messages per call (not strictly required at Groq's
higher limits, but keeps total calls low and fast, and this is what will
also be used for reply generation and judging later).

Same zero-shot design and same evaluation target (golden set) as before --
only the backend changed. Groq exposes an OpenAI-compatible API, so we use
the standard `openai` Python package pointed at Groq's base_url instead of
Google's SDK.

Usage:
    python scripts/llm_intent_classifier.py        # runs all 200 rows
    python scripts/llm_intent_classifier.py 20      # test on first 20 rows only (1 batch)
"""

import os
import sys
import json
import time
import pandas as pd
from openai import OpenAI

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
OUTPUT_PATH = "data/llm_classifier_predictions.csv"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 3  # seconds -- Groq's free tier is generous, light pacing is enough
MAX_RETRIES = 4

INTENTS = [
    "Delivery Delay / Non-Delivery",
    "Wrong / Missing / Damaged Item",
    "Refund / Billing / Charge Dispute",
    "Account Access Issue",
    "App / Digital Content / Technical Issue",
    "Marketplace / Seller / Pricing Issue",
    "General Inquiry / Feedback",
]

INTENT_DESCRIPTIONS = """
1. Delivery Delay / Non-Delivery -- late packages, stuck "out for delivery," wrong location, missed pickup, missed guaranteed delivery dates.
2. Wrong / Missing / Damaged Item -- wrong product/size/variant received, damaged item, missing part of an order.
3. Refund / Billing / Charge Dispute -- disputed charges, refund requests, unexpected/recurring charges, promised refunds not arriving.
4. Account Access Issue -- login/password problems, account closed/suspended/hacked, missing order history.
5. App / Digital Content / Technical Issue -- Amazon app, Prime Video/Music, Alexa/Echo, checkout errors, website bugs.
6. Marketplace / Seller / Pricing Issue -- third-party seller complaints, pricing changes, confusion over who's responsible (Amazon vs seller).
7. General Inquiry / Feedback -- vague complaints with no clear actionable request, general questions, positive feedback/gratitude not tied to a specific transaction.
""".strip()

SYSTEM_PROMPT = f"""You are classifying customer support messages sent to @AmazonHelp on Twitter into exactly one of 7 intent categories.

Categories:
{INTENT_DESCRIPTIONS}

You will be given a numbered list of customer messages. For EACH message, choose the single best-fitting category from the exact list above (use the exact category text, no changes).

Respond with ONLY a JSON object of this exact shape, nothing else, no markdown fences:
{{"classifications": [{{"index": 0, "intent": "<exact category text>"}}, {{"index": 1, "intent": "<exact category text>"}}, ...]}}

Every message in the input must appear exactly once in the output, using the same index numbers as given."""


def classify_batch(client, batch_df):
    """Classify a batch of rows. Returns a dict {index: intent}."""
    lines = []
    for local_idx, (_, row) in enumerate(batch_df.iterrows()):
        text = str(row["customer_text"]).replace("\n", " ")
        lines.append(f'{local_idx}. "{text}"')
    prompt = "Messages:\n" + "\n".join(lines)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            results = parsed["classifications"]
            result_map = {}
            for r in results:
                intent = r["intent"]
                if intent not in INTENTS:
                    # Guard against near-miss text (e.g. slightly reworded);
                    # treat as a parse failure and retry the whole batch.
                    raise ValueError(f"Unrecognized intent returned: {intent!r}")
                result_map[r["index"]] = intent

            if len(result_map) != len(batch_df):
                raise ValueError(
                    f"Expected {len(batch_df)} classifications, got {len(result_map)}"
                )
            return result_map
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    Batch error on attempt {attempt + 1}: {e}")
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to classify batch after {MAX_RETRIES} attempts")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    df = pd.read_csv(GOLDEN_SET_PATH)
    if limit:
        df = df.head(limit).copy()
    df = df.reset_index(drop=True)
    n = len(df)

    num_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Classifying {n} rows in {num_batches} batches of up to {BATCH_SIZE} (via Groq)...")

    all_predictions = [None] * n
    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch_df = df.iloc[start:end].reset_index(drop=True)

        print(f"\nBatch {batch_num + 1}/{num_batches} (rows {start}-{end - 1})...")
        result_map = classify_batch(client, batch_df)

        for local_idx, pred in result_map.items():
            all_predictions[start + local_idx] = pred

        batch_correct = sum(
            1 for local_idx, pred in result_map.items()
            if pred == batch_df.iloc[local_idx]["intent_confirmed"]
        )
        print(f"  Batch accuracy: {batch_correct}/{len(batch_df)}")

        if batch_num < num_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    df["predicted_intent"] = all_predictions
    df["correct"] = df["predicted_intent"] == df["intent_confirmed"]

    accuracy = df["correct"].mean()
    print(f"\nOverall accuracy: {accuracy:.1%} ({df['correct'].sum()}/{n})")
    print("(Trivial baseline: 25.0%  |  Simple TF-IDF+LogReg baseline: 58.5%)")

    output_df = df[["customer_tweet_id", "customer_text", "intent_confirmed", "predicted_intent", "correct"]]
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()