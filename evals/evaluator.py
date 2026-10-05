"""LLM judge for ResearchForge final reports."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from evals.models import JudgeScore


class ResearchReportJudge:
    """Score a ResearchForge report against the retrieved evidence."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """Initialize the judge with a configured chat model."""

        self._llm = llm.with_structured_output(JudgeScore)

    def evaluate(
        self,
        question: str,
        report: dict[str, Any],
        findings: list[dict[str, Any]],
    ) -> JudgeScore:
        """Score report quality using the retrieved evidence."""

        compact_findings: list[dict[str, Any]] = []

        for finding in findings:
            compact_findings.append(
                {
                    "task": finding.get("task", ""),
                    "summary": finding.get("summary", ""),
                    "key_points": finding.get("key_points", []),
                    "limitations": finding.get("limitations", []),
                    "sources": [
                        {
                            "title": source.get("title", ""),
                            "url": source.get("url", ""),
                            "snippet": source.get("snippet", ""),
                            "source_type": source.get(
                                "source_type",
                                "unknown",
                            ),
                        }
                        for source in finding.get("sources", [])[:10]
                    ],
                }
            )

        response = self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the evaluation judge for an agentic research "
                        "system. Score the final report using ONLY the supplied "
                        "research evidence.\n\n"
                        "Scoring:\n"
                        "- correctness: factual claims are accurate relative "
                        "to the evidence.\n"
                        "- completeness: important parts of the question are "
                        "addressed.\n"
                        "- groundedness: claims stay within the supplied "
                        "evidence and uncertainty is handled properly.\n"
                        "- clarity: the report is organized, readable, and "
                        "appropriately concise.\n"
                        "- overall: holistic score from 0 to 10.\n\n"
                        "Do not reward impressive-sounding unsupported claims. "
                        "A report can be clear but poorly grounded. Explain "
                        "the main reason for the overall score."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": question,
                            "report": report,
                            "research_evidence": compact_findings,
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )

        return response
