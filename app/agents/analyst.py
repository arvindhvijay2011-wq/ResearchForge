"""Analyst agent for ResearchForge."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import Analysis


def create_analyst(
    llm: ChatOpenAI,
):
    """Create the Analyst agent."""

    structured_llm = llm.with_structured_output(
        Analysis
    )

    def analyst(
        state: dict,
    ) -> dict:
        """Analyze findings collected from all Scouts."""

        findings = state["findings"]

        print("\n" + "=" * 70)
        print("[ANALYST]")
        print("=" * 70)
        print(
            f"Analyzing {len(findings)} findings..."
        )

        context = json.dumps(
            findings,
            indent=2,
            ensure_ascii=False,
        )

        response = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Analyst Agent in ResearchForge.\n\n"
                        "Analyze the independent findings produced by "
                        "multiple Scouts.\n\n"
                        "Identify:\n"
                        "- strongest findings\n"
                        "- recurring patterns\n"
                        "- contradictions\n"
                        "- evidence gaps\n"
                        "- differences in source strength\n\n"
                        "Do not invent information."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n\n"
                        f"{state['question']}\n\n"
                        f"Scout findings:\n\n"
                        f"{context}"
                    )
                ),
            ]
        )

        return {
            "analysis": response.model_dump(
                mode="json"
            ),
            "events": [
                {
                    "type": "agent_completed",
                    "agent": "analyst",
                    "detail": (
                        "Cross-source analysis completed."
                    ),
                }
            ],
        }

    return analyst