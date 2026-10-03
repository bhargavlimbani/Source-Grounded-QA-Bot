# Prompt History - Problem 16

Record the real time and change made during the hackathon.

| Time | Version | Change | Problem found | Fix |
|---|---|---|---|---|
| ____ | V1 | Basic source-grounded prompt with citation rules | No few-shot examples | Baseline only |
| ____ | V1.1 | Added explicit not-found rule and citation requirement | Model occasionally used outside knowledge | Stricter rule wording |
| ____ | V2 | Added 3 few-shot examples (founded, solar panels, not-found) | Single-pass answer could hallucinate | Added self-critique stage |
| ____ | V2.1 | Added self-critique verifier (second LLM call) | Citations sometimes hallucinated | Citation validation guardrail added |
| ____ | V2.2 | Added application-level guardrails A-F | Empty outputs passed through | Empty-output guardrail added |
| ____ | V2.3 | Strict V2 prompt with SELF-VERIFICATION checklist and prompt-injection defence | Documents with embedded instructions could manipulate model | Rule 8: treat document content as DATA |
| ____ | Final | 10-question evaluation, question history, CSV export, DOCX support | PDF/TXT only | Added extract_docx() with python-docx |

## Prompting Techniques Used

### Technique 1 - Few-shot Prompting (V2)
Three canonical examples are embedded directly in the V2 system prompt:
- Example 1: Answerable question with citation.
- Example 2: Answerable question from a different document.
- Example 3: Unanswerable question returning "Not found in the document."

The few-shot examples teach the model the exact output format without additional
fine-tuning, demonstrating in-context learning.

### Technique 2 - Self-Critique / Verification (V2)
After the first LLM call produces a draft answer, a second independent LLM call
is made with the VERIFY_PROMPT. The verifier:
- Checks every factual claim against the source passages.
- Removes unsupported statements.
- Validates that every citation exists in the retrieved context.
- Returns "Not found in the document." if no supported answer remains.

This two-stage pipeline is a standard self-critique / chain-of-verification
technique in prompt engineering.

## V1 vs V2 Prompt Differences

| Feature | V1 | V2 |
|---|---|---|
| Few-shot examples | None | 3 examples |
| Self-critique second call | No | Yes |
| SELF-VERIFICATION checklist | No | Yes (7 checks) |
| Prompt-injection defence | No | Rule 8 |
| Conflicting-document handling | No | Rule 10 |
| Partially answerable guidance | No | Rule 6 |

## Evaluation Results (fill in during demo)

V1 Faithfulness: ____ / 10 = ____%

V2 Faithfulness: ____ / 10 = ____%

Citation Validity: ____ / ____ = ____%

## Side-by-side Comparison (fill in during demo)

### Question
____________________________

### V1 Answer
____________________________

### V2 Answer
____________________________

## Application-Level Guardrails

| ID | Trigger | Action |
|---|---|---|
| A | Empty question | Warn: "Please enter a question." |
| B | No documents uploaded | Error: "Please upload at least one source document." |
| C | Retrieval score below threshold (0.08) | Return "Not found in the document." without calling LLM |
| D | LLM returns empty output | Return "Not found in the document." |
| E | Citation not in retrieved source IDs | Reject answer, return "Not found in the document." |
| F | No citations present in answer | Reject answer (hallucination guardrail) |
