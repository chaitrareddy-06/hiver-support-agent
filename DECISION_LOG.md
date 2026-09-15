# Decision Log

Non-obvious decisions made during this build, and why.

1. **Brand: AmazonHelp.** Chosen for volume and variety of issue types in the dataset,
   giving enough real examples per intent category to build a meaningful golden set.

2. **English-only scope.** The raw Twitter data is multilingual; non-English volume was
   quantified before excluding it, and the decision is documented rather than silent.
   Tradeoff: a real deployment would need multilingual support.

3. **langdetect over an ASCII heuristic for language filtering.** An initial ASCII-based
   filter was replaced with `langdetect` for more reliable filtering, with before/after
   pair counts logged. Known limitation: langdetect is less reliable on short strings,
   which may cause marginal misclassification at the English/non-English boundary —
   disclosed, not corrected, given the scale of the effect is small.

4. **7-intent taxonomy**, sized deliberately against the fixed 200-row golden set budget
   (roughly ~20-50 examples per intent) rather than picking a taxonomy size first and
   hoping the label budget would stretch to cover it.

5. **"Wrong / Missing / Damaged Item" kept as one category**, not split into three.
   These issues share the same resolution pattern (replacement or refund), so splitting
   them would have fragmented the golden set without changing how the agent should act.

6. **Escalation-request treated as a cross-cutting boolean signal, not an 8th intent.**
   A customer can request escalation regardless of their underlying issue category, so
   modeling it as an independent signal (used later in the escalation policy) is more
   accurate than forcing it into the same taxonomy as topic intents.

7. **Escalation detection uses phrase-matching, not an LLM.** A deliberate engineering
   judgment call: this signal needs to be fast, deterministic, and auditable (why did
   this message get flagged?), which a regex/phrase approach gives for free and an LLM
   call would complicate for little benefit at this signal's scope.

8. **Stratified, not proportional, golden-set sampling** (minimum 20 examples per
   category). This guarantees every intent has enough examples to evaluate meaningfully,
   at the cost of the golden set's aggregate accuracy not directly reflecting real-world
   traffic mix — an explicit, documented tradeoff (see Report, "misleading headline
   number").

9. **Reply-quality rubric uses a derived Overall Quality score**, not an independently
   collected label. Overall Quality is mechanically derived from the three sub-criteria
   (Relevant / Concrete-Actionable / Resolves-Forward), which keeps the rubric internally
   consistent and avoids the possibility of a human rater's overall impression
   contradicting their own sub-scores.

10. **LLM provider switched twice under real constraints, landing on Groq.**
    Anthropic's API required a prepaid card unavailable to the developer. Gemini's free
    tier was tried next but was found, mid-build, to have been silently cut to 20
    requests/day (confirmed via live 429s and corroborated by other developers hitting
    the same new limit). Groq (OpenAI-compatible endpoint, no card required, generous
    free tier) was adopted as the final, consistent provider for classification, reply
    generation, and judging.

11. **Batching multiple messages per LLM call**, originally adopted to survive Gemini's
    20/day cap, and kept after switching to Groq because it remains good practice
    (fewer calls, faster wall-clock time) even without that constraint.

12. **Zero-shot LLM intent classification, not few-shot or fine-tuned.** No labeled
    training set exists beyond the golden 200 rows, and the golden set is deliberately
    kept as a pure evaluation set, never used for training — this also matches exactly
    what the classifier would see on live, unlabeled traffic.

13. **Retrieval corpus explicitly excludes golden-set threads** to prevent retrieval
    leakage — without this exclusion, the grounded-reply system could "cheat" by
    retrieving the exact original answer for a golden-set question, inflating apparent
    reply quality.

14. **Incremental checkpointing added to reply generation after a full-run crash lost
    all progress.** The first full run crashed on batch 13/40 with no partial output
    saved, losing everything. Fixed by saving output after every batch and treating a
    failed batch (after retries) as "leave blank and continue" instead of a hard crash.
    This pattern (save every batch, never let one failure lose prior work) was then
    reused in every subsequent script (grounded replies, LLM judge).

15. **Judge model deliberately different from the drafting model**, to avoid
    self-grading bias (a model rating its own output favorably) — a recognized practice
    in LLM-as-judge setups. This was also a practical necessity: the drafting model's
    daily token quota was exhausted by the time judging began, but the choice is
    defensible independent of that constraint.

16. **Multiple judge models used across the final judge run, tracked per-row**, due to
    sequential daily quota exhaustion across four different Groq models in the same day
    (`gpt-oss-20b`, `gpt-oss-120b`, `qwen3.8-27b`, `groq/compound-mini`, finally
    `allam-2-7b`). Rather than hide this, each row in `judge_scores.csv` records exactly
    which model judged it, so the mix is fully auditable rather than asserted.

17. **Judge rubric iterated twice, then stopped, following the same "iterate once,
    document, move on" philosophy used elsewhere in the project.** First version was
    too lenient (missed contextually irrelevant boilerplate copy-pasted from unrelated
    conversations). A stricter, more explicit version fixed that specific issue but
    overcorrected in the opposite direction (flagging genuinely acceptable replies as
    Poor). Judge-vs-human Cohen's kappa stayed roughly flat (0.281 to 0.246) across the
    two versions, but the *type* of disagreement changed entirely — documented as
    evidence of how sensitive LLM-as-judge calibration is to prompt wording.

18. **A third calibration approach (few-shot anchored examples) was tested on a small
    sample but not scaled to the full run**, given submission time constraints. On a
    20-row sample it showed a substantial jump in judge-human agreement (kappa 0.730 vs
    0.246-0.281 for the instruction-only versions), suggesting concrete anchored
    examples calibrate an LLM judge more effectively than abstract strictness
    instructions — flagged as the top priority for "what I'd do with one more week"
    rather than rushed to completion under deadline pressure.

19. **`client.models.list()` used to verify actual API access instead of trusting
    provider documentation.** Groq's public docs list `llama-3.3-70b-versatile` as
    available, but it was not accessible on this account — caught by querying the API
    directly rather than assuming docs are current, a generalizable lesson repeated
    later when picking fallback models during quota exhaustion.
