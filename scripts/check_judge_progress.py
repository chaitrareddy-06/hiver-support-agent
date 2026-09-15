import pandas as pd

df = pd.read_csv("data/judge_scores.csv")
print(f"orig judged: {df['orig_overall_quality'].notna().sum()}/200")
print(f"draft judged: {df['draft_overall_quality'].notna().sum()}/200")