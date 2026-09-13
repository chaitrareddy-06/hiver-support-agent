import pandas as pd
import re

df = pd.read_csv("data/amazon_threads.csv")
print(f"Total pairs: {len(df)}")

# --- Rough keyword-based auto-tagger, used ONLY to bucket rows for
# stratified sampling. This is NOT the real intent classifier (Phase 4).
KEYWORD_RULES = {
    "Delivery Delay / Non-Delivery": [
        r"\bdeliver", r"\bshipping\b", r"\bshipped\b", r"\bpackage\b",
        r"\btracking\b", r"\bcourier\b", r"\bcarrier\b", r"\blate\b",
        r"\bout for delivery\b", r"\bcancell?ed\b", r"\bnot arrived\b",
        r"\bhasn'?t arrived\b", r"\bnever (came|arrived)\b"
    ],
    "Wrong / Missing / Damaged Item": [
        r"\bwrong item\b", r"\bmissing item\b", r"\bdamaged\b", r"\bwrong size\b",
        r"\bnot the (one|item)\b", r"\bbroken\b", r"\bdefective\b",
        r"\bdoesn'?t work\b", r"\bdon'?t work\b", r"\bnone of them work\b",
        r"\breplace(ment|d)?\b", r"\bfaulty\b"
    ],
    "Refund / Billing / Charge Dispute": [
        r"\brefund", r"\bcharged\b", r"\bcharge\b", r"\bbilling\b",
        r"\bcashback\b", r"\bmoney back\b", r"\bpayment\b", r"\bcharges?\b",
        r"\bpay(ing|ment)?\b.*\b(£|\$|rs\.?|rupees|extra)\b",
        r"\bbank statement\b", r"\bnot refunded\b", r"\bdidn'?t.{0,15}refund"
    ],
    "Account Access Issue": [
        r"\bpassword\b", r"\blogin\b", r"\bsign in\b",
        r"\baccount (closed|suspended|hacked|locked)\b",
        r"\bcan'?t access\b", r"\baccess (to |)(the |my |)account\b",
        r"\bgained access\b", r"\bhack(ed)?\b"
    ],
    "App / Digital Content / Technical Issue": [
        r"\bapp\b", r"\becho\b", r"\balexa\b", r"\bprime video\b", r"\bstreaming\b",
        r"\bwebsite\b", r"\bcheckout\b", r"\bbug\b", r"\berror\b", r"\blink\b",
        r"\burl\b", r"\bsubtitle"
    ],
    "Marketplace / Seller / Pricing Issue": [
        r"\bseller\b", r"\bmarketplace\b", r"\bprice\b", r"\bpricing\b",
        r"\boverpriced\b", r"\bfulfilled by\b", r"\bsold by\b", r"\bsupplier\b",
        r"\bin stock\b", r"\bnot showing as buyable\b"
    ],
}

def auto_tag(text):
    if not isinstance(text, str):
        return "General Inquiry / Feedback"
    for intent, patterns in KEYWORD_RULES.items():
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return intent
    return "General Inquiry / Feedback"

print("Auto-tagging for stratified sampling (rough, not final classifier)...")
df["rough_intent"] = df["customer_text"].apply(auto_tag)

print("\nRough intent distribution (full dataset):")
print(df["rough_intent"].value_counts())

# --- Stratified sampling with a minimum floor per category
MIN_PER_CATEGORY = 20
TARGET_TOTAL = 200

counts = df["rough_intent"].value_counts()
n_categories = len(counts)

# Give every category at least MIN_PER_CATEGORY, then distribute the rest
# proportionally to category size (so Delivery Delay still ends up largest)
remaining = TARGET_TOTAL - (MIN_PER_CATEGORY * n_categories)
proportional_extra = (counts / counts.sum() * remaining).round().astype(int)

sample_frames = []
for intent, base_count in counts.items():
    n_target = MIN_PER_CATEGORY + proportional_extra[intent]
    n_target = min(n_target, base_count)  # can't sample more than exists
    subset = df[df["rough_intent"] == intent]
    sampled = subset.sample(n=n_target, random_state=42)
    sample_frames.append(sampled)

golden = pd.concat(sample_frames).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nFinal golden set size: {len(golden)}")
print(golden["rough_intent"].value_counts())

# --- Add escalation signal + empty labeling columns
import sys
sys.path.append("scripts")
from escalation_signals import requests_human_escalation

golden["requests_escalation"] = golden["customer_text"].apply(requests_human_escalation)

golden["intent_confirmed"] = ""
golden["relevant"] = ""
golden["concrete_actionable"] = ""
golden["resolves_forward"] = ""
golden["overall_quality"] = ""
golden["notes"] = ""

output_cols = [
    "customer_tweet_id", "customer_text", "amazon_reply_text",
    "rough_intent", "requests_escalation",
    "intent_confirmed", "relevant", "concrete_actionable",
    "resolves_forward", "overall_quality", "notes"
]
golden[output_cols].to_csv("data/golden_set_unlabeled.csv", index=False)
print("\nSaved to data/golden_set_unlabeled.csv")