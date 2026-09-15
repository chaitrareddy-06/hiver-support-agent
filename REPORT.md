# Report: AI Support Agent for AmazonHelp

## 1. Problem Framing

**Brand chosen:** AmazonHelp, for the volume and variety of issue types in the dataset —
enough real examples per category to build a meaningful golden evaluation set.

**Scope:** English-language messages only. This is a deliberate, documented tradeoff, not
an oversight — the raw dataset is multilingual, and a production system would need to
handle other languages, but doing so well was out of scope for this build's timeline.

**What "good" means for this agent, concretely:**
- **Classification**: the predicted intent should match what a human reviewing the same
  single message, with no other context, would reasonably assign.
- **Reply drafting**: a reply is good if it is *Relevant* (addresses what the customer
  actually said), *Concrete/Actionable* (gives a specific next step, not generic filler),
  and *Resolves/Forward* (either resolves the issue or clearly moves it forward, e.g. by
  asking for a specific piece of information). All three are independently scored
  Yes/No; Overall Quality is derived (all Yes = Good, one No = Acceptable, 2+ No = Poor).
- **Escalation**: a message should escalate if a human would clearly do better than an
  automated reply — either because the customer explicitly asked for a human, or because
  the system has no good precedent to ground a reply in.

**What I chose not to build:**
- Multi-turn/thread-aware classification (each message is classified alone, matching
  what live, unlabeled traffic would look like, but see Failure Mode #1 below for the
  cost of this).
- Fine-tuning or few-shot classification — zero-shot only, since no training-labeled
  data exists beyond the golden 200 rows, which are kept strictly as an eval set.
- A trivial escalation baseline — deferred, since the golden set has no ground-truth
  "correct escalate/auto-handle" label, only the `requests_escalation` phrase signal,
  which is an input feature, not an outcome label. Building a baseline against a feature
  instead of a real label would be misleading.
- Classifier-confidence-based escalation — considered, but deferred in favor of a
  simpler two-signal policy, given time constraints and an API already under quota
  pressure (see Decision Log #15-16).

## 2. System Overview

1. **Intent classification** — Groq (`openai/gpt-oss-20b`/`120b`), zero-shot, using a
   7-category taxonomy defined from the data itself (Delivery Delay, Wrong/Missing/
   Damaged, Refund/Billing, Account Access, App/Digital/Technical, Marketplace/Seller,
   General Inquiry/Feedback), plus a cross-cutting `requests_escalation` phrase signal.
2. **Retrieval-grounded reply drafting** — each incoming message is embedded
   (`sentence-transformers/all-MiniLM-L6-v2`) and matched against a 4,000-message
   historical corpus (excluding golden-set threads, to prevent retrieval leakage). The
   top-3 most similar historical messages and their real AmazonHelp replies are given to
   the LLM as grounding examples, with an explicit instruction to match their
   specificity, not just their tone.
3. **Escalation policy** — AUTO-HANDLE unless (a) the customer explicitly requested
   escalation, or (b) the best retrieved historical match was weak (`top1_similarity_score`
   below the 20th percentile, ~0.65), in which case the system escalates as a safety net
   rather than drafting from thin precedent.

## 3. Results vs. Baselines

**Intent classification:**

| Model | Accuracy | Method |
|---|---|---|
| Trivial (majority class) | 25.0% | Always predicts "Delivery Delay" |
| Simple (TF-IDF + Logistic Regression) | 58.5% | 5-fold stratified CV |
| **LLM classifier (Groq, zero-shot)** | **81.0%** | Zero-shot, 7-category taxonomy |

The LLM classifier clearly beats both baselines. Full error analysis (38/200 wrong)
found most errors trace to context-free follow-up messages or genuine multi-topic
ambiguity, not classifier weakness — see Failure Mode #1.

**Reply quality**, scored by an LLM judge using the identical rubric as the golden set's
human labels (Relevant / Concrete-Actionable / Resolves-Forward → Overall Quality):

| Reply source | Good | Acceptable | Poor |
|---|---|---|---|
| Original human (AmazonHelp) replies | 78 (39%) | 30 (15%) | 92 (46%) |
| AI-drafted grounded replies | 108 (54%) | 31 (15.5%) | 61 (30.5%) |

At face value, the AI-drafted replies score higher than the real historical replies on
the same rubric. **This number should not be taken at face value — see Section 5.**

**Judge-vs-human agreement** (comparing the judge's scores on the *original* replies
against the human labels already collected for the golden set, since those are the same
replies scored by two different raters):

| Judge rubric version | Exact agreement | Cohen's kappa |
|---|---|---|
| V1: instruction-only (lenient) | 61.0% | 0.281 (Fair) |
| V2: instruction-only (strict) | 55.0% | 0.246 (Fair) |
| V3: few-shot anchored examples (20-row sample only) | 85.0% | 0.730 (Substantial) |

V3 was not run on the full 200 rows due to submission time constraints — see Section 6.

**Escalation policy:** 155 auto-handle (77.5%) / 45 escalate (22.5%) on the golden set.
Manual spot-check of both categories found the policy's reasoning held up on real
examples — messages escalated for weak grounding were genuinely ambiguous or emotionally
charged (pricing disputes, price-match questions, frustrated delivery complaints), and
auto-handled messages were routine, well-precedented cases.

## 4. Failure Analysis: Top 5 Failure Modes

**1. Context-free follow-up messages defeat single-message intent classification.**
~10 of 38 classifier errors are messages that reply to an earlier conversation turn with
zero topical signal on their own (e.g. "Earlier today. Does it take a couple of days?").
This is a task-design limitation — without full thread context, no single-message
classifier can resolve these — not a fixable classifier bug.

**2. "General Inquiry/Feedback" is a structural confusion magnet.** Involved in 12/38
classifier errors in both directions, and independently the weakest class (F1=0.37) in
the TF-IDF baseline too — consistent evidence this catch-all category is intrinsically
hard to separate from others, not a fluke of one model.

**3. Multi-part reply fragments and mismatched customer-reply pairs cause artificially
low apparent quality.** Of the golden set's 62 "Poor"-labeled original replies, 19
(30.6%) are data-pipeline artifacts (a reply is genuinely only fragment 2 of 2, or the
pairing script matched the wrong reply to a message) rather than genuinely weak replies.
Real, quantified examples are in the golden set's notes column.

**4. Retrieval-grounded replies genericize even when shown specific grounding
examples.** LLM-drafted replies were consistently observed to be more generic than the
real historical replies they were grounded in, despite an explicit system-prompt
instruction to match the grounding examples' specificity. Example: a real historical
reply gave a specific action ("Select your order and then follow the prompts for a
return label... here: [link]"); the drafted reply for a very similar new case said only
"Please reach our support team here: [link]" — despite being shown the specific example
as grounding.

**5. Retrieved grounding text can bleed into the drafted reply even when contextually
wrong.** One drafted reply for a wrong-item-shipped case included the sentence "we
consider it to be personal information. Our page is visible to the public" — a phrase
that belongs to a completely unrelated privacy-related conversation elsewhere in the
corpus, and makes no sense in context. This suggests the model sometimes reuses
retrieved phrasing without checking whether it actually fits the new message. Notably,
the first version of the LLM judge scored this reply "Good" despite the irrelevant
sentence — the judge rubric had to be tightened specifically to catch this pattern (see
Decision Log #17).

## 5. What Is Misleading About My Headline Number?

1. **Stratified, not proportional, golden-set sampling.** The 81.0% classifier accuracy
   and reply-quality numbers reflect a golden set balanced across intent categories
   (minimum 20 examples each), not real-world traffic proportions. Aggregate accuracy on
   live traffic, where "Delivery Delay" likely dominates, would differ.

2. **~9.5% of the golden set's "Poor" original-reply labels are pipeline artifacts, not
   genuine quality failures.** The corrected estimate of genuinely poor original replies
   is ~21.5% (43/200), not the raw 31% (62/200) — see Failure Mode #3.

3. **81.0% classifier accuracy may already be close to the practical ceiling for a
   single-message classifier on this data.** ~10 of the 38 errors are on messages that
   are inherently unclassifiable without full thread context (Failure Mode #1) — the
   "true" achievable ceiling without that context may be meaningfully below 100%.

4. **The "AI beats human" reply-quality comparison (54% vs 39% Good) should not be
   read as a clean win.** It is scored by an LLM judge whose agreement with human raters
   is only Fair (kappa 0.246-0.281 across two tested versions) on the *exact same*
   original replies. A judge this imperfect may be rewarding fluent, LLM-style phrasing
   over terse-but-effective real replies — the comparison is only as trustworthy as the
   judge itself, and the judge is demonstrably not yet well-calibrated. A promising
   third calibration approach (few-shot examples) showed much stronger agreement (kappa
   0.730) on a small sample, but was not validated at full scale in time for this
   submission.

5. **langdetect's known weakness on short strings** may have caused marginal English/
   non-English misclassification during scope filtering — disclosed, not corrected,
   since the scale of the effect on 123k+ messages is expected to be small.

## 6. What I'd Do With One More Week

1. **Scale the few-shot judge calibration to the full 200 rows** and confirm the kappa
   0.730 result holds at scale — this is the single highest-value next step, since
   almost every other quality claim in this report is downstream of judge reliability.
2. **Add full-thread context to the classifier input** where available, to directly
   test whether Failure Mode #1 (context-free follow-ups) is actually fixable with more
   context, rather than an inherent task-design ceiling.
3. **Build the deferred trivial escalation baseline properly** — this requires first
   collecting a small human-labeled "was escalation actually the right call" set (not
   just the `requests_escalation` phrase feature), then scoring the current
   two-signal policy against it.
4. **Investigate the reply-genericization failure mode (Failure Mode #4) further** —
   one prompt-iteration pass didn't meaningfully help; with more time, a few-shot
   grounding approach (showing the model both a specific-good example and a generic-bad
   example) might work better than instruction-only prompting, mirroring what worked for
   judge calibration.
5. **Add a lightweight relevance filter to the retrieved grounding examples** before
   they reach the reply-drafting prompt, to reduce the "contextually wrong phrase
   bleeds into the new reply" failure mode (Failure Mode #5).
6. **Add classifier-confidence as a third escalation signal**, as originally considered
   but deferred (Decision Log #15) — would require capturing per-prediction confidence,
   not just the label, from the classifier.
7. **Add multilingual support**, scoping and quantifying the non-English volume that
   was excluded in this build, and evaluating whether the same pipeline transfers or
   needs per-language tuning.

## Appendix: A Note on Infrastructure Constraints

This project hit real, cascading free-tier API limits: Anthropic's API required a
prepaid card unavailable to the developer; Google Gemini's free tier was silently cut to
20 requests/day mid-build; and across the final day of work, four different Groq models
(`gpt-oss-20b`, `gpt-oss-120b`, `qwen3.8-27b`, `groq/compound-mini`) hit their daily
quota in sequence during reply generation and judge scoring, requiring a switch to a
fifth (`allam-2-7b`) to complete the final judge rows. Each switch is documented in the
Decision Log and in `judge_scores.csv`'s per-row `judge_model` column, rather than
smoothed over. This is disclosed here explicitly because it materially affected which
model scored which rows, and because working through real infrastructure constraints
under deadline pressure was, in itself, a meaningful part of this build.
