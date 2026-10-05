"""Run the ResearchForge evaluation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.graph.workflow import build_graph
from evals.evaluator import ResearchReportJudge
from evals.models import CaseEvaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "golden_cases.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    """Load benchmark cases from JSONL."""

    cases = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return cases[:limit] if limit else cases


def create_llm() -> ChatOpenAI:
    """Create the configured LLM."""

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-5-mini",
    )

    return ChatOpenAI(model=model_name)


def topic_coverage(
    report: dict[str, Any],
    required_topics: list[str],
) -> float:
    """Estimate required-topic coverage with simple text matching."""

    if not required_topics:
        return 1.0

    report_text = json.dumps(
        report,
        ensure_ascii=False,
    ).lower()

    matched = sum(
        1
        for topic in required_topics
        if topic.lower() in report_text
    )

    return matched / len(required_topics)


def run_case(
    case: dict[str, Any],
    judge: ResearchReportJudge,
) -> CaseEvaluation:
    """Execute and evaluate one benchmark case."""

    llm = create_llm()
    graph = build_graph(llm)

    initial_state: dict[str, Any] = {
        "question": case["question"],
        "plan": [],
        "findings": [],
        "analysis": {},
        "critique": {},
        "report": {},
        "sources": [],
        "events": [],
    }

    planner_tasks: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    source_urls: set[str] = set()
    tool_call_count = 0

    started = time.perf_counter()

    for update in graph.stream(
        initial_state,
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue

        for node_name, node_update in update.items():
            if not isinstance(node_update, dict):
                continue

            normalized_node = str(node_name).lower()

            if normalized_node == "planner":
                planner_tasks.extend(
                    node_update.get("plan", [])
                )

            elif normalized_node == "scout":
                scout_findings = node_update.get(
                    "findings",
                    [],
                )

                findings.extend(
                    scout_findings
                )

                for finding in scout_findings:
                    tool_call_count += len(
                        finding.get("tool_trace", [])
                    )

                    for source in finding.get(
                        "sources",
                        [],
                    ):
                        url = source.get("url")

                        if url:
                            source_urls.add(
                                str(url)
                            )

            elif normalized_node == "synthesizer":
                report = node_update.get(
                    "report",
                    {},
                )

    elapsed = time.perf_counter() - started

    judge_score = judge.evaluate(
        question=case["question"],
        report=report,
        findings=findings,
    )

    planner_count = len(planner_tasks)

    return CaseEvaluation(
        case_id=case["id"],
        question=case["question"],
        pipeline_completed=bool(report),
        planner_task_count=planner_count,
        planner_task_count_ok=3 <= planner_count <= 4,
        findings_count=len(findings),
        source_count=len(source_urls),
        tool_call_count=tool_call_count,
        report_present=bool(report),
        report_section_count=len(
            report.get("sections", [])
        ),
        required_topic_coverage=round(
            topic_coverage(
                report=report,
                required_topics=case.get(
                    "required_topics",
                    [],
                ),
            ),
            3,
        ),
        judge=judge_score,
        elapsed_seconds=round(
            elapsed,
            2,
        ),
    )


def write_results(
    evaluations: list[CaseEvaluation],
) -> Path:
    """Write JSON and Markdown benchmark results."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = RESULTS_DIR / f"{timestamp}.json"
    markdown_path = RESULTS_DIR / f"{timestamp}.md"

    payload = [
        evaluation.model_dump(
            mode="json"
        )
        for evaluation in evaluations
    ]

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    count = len(evaluations)

    average = (
        sum(
            item.judge.overall
            for item in evaluations
        )
        / count
        if count
        else 0.0
    )

    grounded = (
        sum(
            item.judge.groundedness
            for item in evaluations
        )
        / count
        if count
        else 0.0
    )

    completeness = (
        sum(
            item.judge.completeness
            for item in evaluations
        )
        / count
        if count
        else 0.0
    )

    coverage = (
        sum(
            item.required_topic_coverage
            for item in evaluations
        )
        / count
        if count
        else 0.0
    )

    report_lines = [
        "# ResearchForge Evaluation",
        "",
        f"Cases evaluated: {count}",
        f"Average overall: {average:.2f}/10",
        f"Average groundedness: {grounded:.2f}/10",
        f"Average completeness: {completeness:.2f}/10",
        f"Average required-topic coverage: {coverage:.1%}",
        "",
        "| Case | Overall | Groundedness | Completeness | Coverage | Sources | Tools | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for item in evaluations:
        report_lines.append(
            "| "
            f"{item.case_id} | "
            f"{item.judge.overall}/10 | "
            f"{item.judge.groundedness}/10 | "
            f"{item.judge.completeness}/10 | "
            f"{item.required_topic_coverage:.0%} | "
            f"{item.source_count} | "
            f"{item.tool_call_count} | "
            f"{item.elapsed_seconds:.2f} |"
        )

    markdown_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    return json_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run ResearchForge evals."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print summary results."""

    load_dotenv()

    args = parse_args()
    cases = load_cases(
        limit=args.limit
    )

    judge = ResearchReportJudge(
        create_llm()
    )

    evaluations: list[CaseEvaluation] = []

    print("=" * 72)
    print("RESEARCHFORGE EVALUATION")
    print("=" * 72)
    print(f"Cases: {len(cases)}")

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"\n[{index}/{len(cases)}] "
            f"{case['id']}"
        )

        try:
            evaluation = run_case(
                case=case,
                judge=judge,
            )

        except Exception as error:
            print(
                f"  ERROR: {error}"
            )
            continue

        evaluations.append(
            evaluation
        )

        print(
            f"  Pipeline: "
            f"{'PASS' if evaluation.pipeline_completed else 'FAIL'}"
        )
        print(
            f"  Planner tasks: "
            f"{evaluation.planner_task_count}"
        )
        print(
            f"  Findings: "
            f"{evaluation.findings_count}"
        )
        print(
            f"  Sources: "
            f"{evaluation.source_count}"
        )
        print(
            f"  Overall judge: "
            f"{evaluation.judge.overall}/10"
        )
        print(
            f"  Groundedness: "
            f"{evaluation.judge.groundedness}/10"
        )

    result_path = write_results(
        evaluations
    )

    average = (
        sum(
            item.judge.overall
            for item in evaluations
        )
        / len(evaluations)
        if evaluations
        else 0.0
    )

    print("\n" + "=" * 72)
    print(
        f"AVERAGE OVERALL: {average:.2f}/10"
    )
    print(
        f"Results saved to: {result_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
