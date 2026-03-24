"""
Eval Framework Flow Example: Testing Flow Quality

This example demonstrates Water's evaluation framework for measuring
flow output quality against expected results. It shows:
  - EvalSuite with EvalCase for defining test scenarios
  - ExactMatch and ContainsMatch for deterministic evaluation
  - SemanticSimilarity for fuzzy text comparison
  - LLMJudge with a MockProvider for LLM-based scoring
  - EvalReport for analyzing results and detecting regressions

NOTE: The LLMJudge uses OpenAIProvider and requires a valid OPENAI_API_KEY.
      The classification flow itself is rule-based (no LLM needed).
"""

import asyncio
from typing import Any, Dict

from pydantic import BaseModel

from water.core import Flow, create_task
from water.agents.llm import OpenAIProvider
from water.eval import (
    EvalSuite,
    EvalCase,
    Evaluator,
    ExactMatch,
    ContainsMatch,
    LLMJudge,
    SemanticSimilarity,
    EvalReport,
)


# ---------------------------------------------------------------------------
# A mock flow to evaluate
# ---------------------------------------------------------------------------

class ClassifyInput(BaseModel):
    text: str

class ClassifyOutput(BaseModel):
    label: str
    confidence: float
    summary: str


def classify_text(params: Dict[str, Any], context) -> Dict[str, Any]:
    """Simple rule-based classifier for demonstration."""
    data = params["input_data"]
    text = data.get("text", "").lower()

    if any(w in text for w in ["bug", "error", "crash", "fix"]):
        label, conf = "bug_report", 0.9
    elif any(w in text for w in ["feature", "add", "request", "new"]):
        label, conf = "feature_request", 0.85
    elif any(w in text for w in ["question", "how", "what", "why"]):
        label, conf = "question", 0.8
    else:
        label, conf = "other", 0.5

    return {
        "label": label,
        "confidence": conf,
        "summary": f"Classified as {label} with {conf:.0%} confidence",
    }


def build_classify_flow() -> Flow:
    task = create_task(
        id="classifier",
        description="Classify support tickets",
        input_schema=ClassifyInput,
        output_schema=ClassifyOutput,
        execute=classify_text,
    )
    flow = Flow(id="classify_flow", description="Text classifier")
    flow.then(task).register()
    return flow


# ---------------------------------------------------------------------------
# Example 1: ExactMatch and ContainsMatch evaluators
# ---------------------------------------------------------------------------

async def example_deterministic_eval():
    """Run exact and contains-match evaluators on a classification flow."""
    print("=== Example 1: Deterministic Evaluation (ExactMatch + ContainsMatch) ===\n")

    flow = build_classify_flow()

    cases = [
        EvalCase(
            input={"text": "The app crashes when I click submit"},
            expected={"label": "bug_report"},
            name="crash_report",
        ),
        EvalCase(
            input={"text": "Please add dark mode to the settings"},
            expected={"label": "feature_request"},
            name="feature_req",
        ),
        EvalCase(
            input={"text": "How do I reset my password?"},
            expected={"label": "question"},
            name="password_question",
        ),
    ]

    suite = EvalSuite(
        flow=flow,
        evaluators=[
            ExactMatch(key="label"),
            ContainsMatch(keys=["label", "confidence", "summary"]),
        ],
        cases=cases,
        name="classifier_deterministic",
    )

    report = await suite.run()
    print(report.summary())
    print()
    for cr in report.case_results:
        scores_str = ", ".join(f"{s.evaluator}={s.score:.1f}" for s in cr.scores)
        print(f"  Case {cr.case_index} (passed={cr.passed}): {scores_str}")
    print()


# ---------------------------------------------------------------------------
# Example 2: SemanticSimilarity evaluator
# ---------------------------------------------------------------------------

async def example_semantic_eval():
    """Use token-overlap similarity to compare output text."""
    print("=== Example 2: SemanticSimilarity Evaluation ===\n")

    flow = build_classify_flow()

    cases = [
        EvalCase(
            input={"text": "There's a bug in the login page"},
            expected={"summary": "Classified as bug_report with high confidence"},
            name="similarity_test_1",
        ),
        EvalCase(
            input={"text": "Can you add a new export feature?"},
            expected={"summary": "Classified as feature_request with good confidence"},
            name="similarity_test_2",
        ),
    ]

    suite = EvalSuite(
        flow=flow,
        evaluators=[
            SemanticSimilarity(key="summary", threshold=0.3),
        ],
        cases=cases,
        name="classifier_semantic",
    )

    report = await suite.run()
    print(report.summary())
    print()
    for cr in report.case_results:
        actual_summary = cr.actual.get("summary", "")
        expected_summary = cr.expected.get("summary", "")
        score = cr.scores[0].score if cr.scores else 0
        print(f"  Case {cr.case_index}: score={score:.2f}")
        print(f"    Expected: {expected_summary}")
        print(f"    Actual:   {actual_summary}")
    print()


# ---------------------------------------------------------------------------
# Example 3: LLMJudge with MockProvider and regression detection
# ---------------------------------------------------------------------------

async def example_llm_judge_and_regression():
    """Use an LLM judge for scoring, then compare against a baseline."""
    print("=== Example 3: LLMJudge + Regression Detection ===\n")

    flow = build_classify_flow()

    # Real OpenAI provider for LLM judge scoring
    llm_judge_provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0)

    cases = [
        EvalCase(
            input={"text": "App crashes on startup after update"},
            expected={"label": "bug_report"},
            name="judge_test_1",
        ),
        EvalCase(
            input={"text": "Would love a calendar integration"},
            expected={"label": "feature_request"},
            name="judge_test_2",
        ),
        EvalCase(
            input={"text": "What are the supported file formats?"},
            expected={"label": "question"},
            name="judge_test_3",
        ),
    ]

    suite = EvalSuite(
        flow=flow,
        evaluators=[
            ExactMatch(key="label"),
            LLMJudge(
                provider=llm_judge_provider,
                rubric="Is the classification correct and the summary clear?",
                scale=5,
            ),
        ],
        cases=cases,
        name="classifier_with_judge",
    )

    # Run current evaluation
    current_report = await suite.run()
    print("Current Report:")
    print(f"  {current_report.summary()}")
    print()

    # Simulate a baseline report (previous version scored perfectly)
    baseline_report = EvalReport()
    baseline_report.total_cases = 3
    baseline_report.passed_cases = 3
    from water.eval.report import CaseResult
    from water.eval.evaluators import EvalScore
    for i in range(3):
        baseline_report.case_results.append(CaseResult(
            case_index=i,
            input_data=cases[i].input,
            expected=cases[i].expected,
            actual=cases[i].expected,
            scores=[EvalScore(evaluator="exact_match", passed=True, score=1.0)],
            passed=True,
            avg_score=1.0,
        ))
    baseline_report.avg_score = 1.0

    # Compare for regressions
    regressions = current_report.compare(baseline_report)
    print(f"Regressions found: {len(regressions)}")
    for reg in regressions:
        print(f"  Case {reg['case_index']}: {reg['type']} "
              f"(baseline={reg['baseline_score']:.2f}, current={reg['current_score']:.2f})")
    print()

    # Export report as JSON
    report_json = current_report.to_json()
    print(f"Report JSON (first 200 chars): {report_json[:200]}...")
    print()


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------

async def main():
    await example_deterministic_eval()
    await example_semantic_eval()
    await example_llm_judge_and_regression()
    print("All eval examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
