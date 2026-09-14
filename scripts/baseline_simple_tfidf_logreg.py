"""
Simple baseline #2: TF-IDF + Logistic Regression for intent classification.

Why cross-validation instead of a single train/test split:
We only have 200 labeled examples (the golden set), and some intents are
already thin (Marketplace/Seller: 15 rows). A single train/test split would
leave too few examples of rare classes in the test set to trust the number.
5-fold stratified cross-validation gives every row exactly one out-of-fold
prediction (made by a model that never saw that row during training), using
all 200 rows for both training and evaluation without ever testing a row on
a model that was trained on it.

Compare this baseline's accuracy directly against the trivial majority-class
baseline (25.0%) to see how much signal a simple bag-of-words model captures
before we bring in an LLM classifier.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

GOLDEN_SET_PATH = "data/golden_set_labeled.csv"
RANDOM_STATE = 42
N_FOLDS = 5


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    X = df["customer_text"]
    y = df["intent_confirmed"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Running {N_FOLDS}-fold stratified cross-validation on {len(df)} rows...")
    predictions = cross_val_predict(pipeline, X, y, cv=skf)

    accuracy = accuracy_score(y, predictions)
    print(f"\nOverall accuracy: {accuracy:.1%} ({(predictions == y).sum()}/{len(df)})")
    print("(Trivial majority-class baseline was 25.0% for comparison)")

    print("\nPer-class precision / recall / F1:")
    print(classification_report(y, predictions, zero_division=0))

    print("Confusion matrix (rows = true intent, columns = predicted intent):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y, predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    # Save out-of-fold predictions alongside ground truth for later error inspection
    output_df = df[["customer_tweet_id", "customer_text", "intent_confirmed"]].copy()
    output_df["predicted_intent"] = predictions
    output_df["correct"] = output_df["predicted_intent"] == output_df["intent_confirmed"]
    output_path = "data/baseline_simple_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved per-row predictions to {output_path}")


if __name__ == "__main__":
    main()