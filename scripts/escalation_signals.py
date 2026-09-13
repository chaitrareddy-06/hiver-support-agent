import re

# Phrases that signal a customer is explicitly demanding human contact,
# regardless of the underlying issue topic. Used as one input signal to
# the AUTO-HANDLE vs ESCALATE decision in Phase 4 (not a standalone intent).
ESCALATION_PHRASES = [
    r"\bcall me\b",
    r"\bcallback\b",
    r"\bcall back\b",
    r"\bspeak to (a |an |)(human|person|someone|representative|agent|manager)\b",
    r"\btalk to (a |an |)(human|person|someone|representative|agent|manager)\b",
    r"\bphone call\b",
    r"\breal (person|human)\b",
    r"\bescalate\b",
    r"\bspeak (to|with) (a |an |the |)supervisor\b",
    r"\bspeak (to|with) (a |an |the |)manager\b",
    r"\btalk (to|with) (a |an |the |)manager\b",
    r"\bask for (a |the |)manager\b",
    r"\bcustomer care (person|number|line)\b",
    r"\bcall center\b",
    r"\bcall centre\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in ESCALATION_PHRASES]

def requests_human_escalation(text: str) -> bool:
    """
    Returns True if the customer message explicitly asks for human/phone
    contact, based on phrase matching. Deterministic and explainable —
    known limitation: misses paraphrased or implicit escalation requests.
    """
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for pattern in _compiled)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/amazon_threads.csv")
    df["requests_escalation"] = df["customer_text"].apply(requests_human_escalation)

    flagged = df[df["requests_escalation"]]
    print(f"Total pairs: {len(df)}")
    print(f"Flagged as explicit escalation request: {len(flagged)} ({len(flagged)/len(df)*100:.1f}%)")

    print("\n--- Sample flagged rows ---")
    for i, row in flagged.sample(n=min(10, len(flagged)), random_state=42).iterrows():
        print(f"\n[Row {i}]")
        print(f"Customer: {row['customer_text']}")