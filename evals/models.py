"""Evaluation models for ResearchForge."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """Semantic quality scores for one final research report."""

    correctness: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    groundedness: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)
    overall: int = Field(ge=0, le=10)
    rationale: str


class CaseEvaluation(BaseModel):
    """Evaluation result for one benchmark case."""

    case_id: str
    question: str
    pipeline_completed: bool
    planner_task_count: int
    planner_task_count_ok: bool
    findings_count: int
    source_count: int
    tool_call_count: int
    report_present: bool
    report_section_count: int
    required_topic_coverage: float
    judge: JudgeScore
    elapsed_seconds: float
