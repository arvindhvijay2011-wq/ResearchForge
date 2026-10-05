"""Critic agent for ResearchForge."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import Critique


def create_critic(
    llm: ChatOpenAI,
):
    """Create the Critic agent."""

    structured_llm = llm.with_structured_output(
        Critique
    )

    def critic(
        state: dict,
    ) -> dict:
        """Audit research quality."""

        print("\n" + "=" * 70)
        print("[CRITIC]")
        print("=" * 70)

        findings_context = json.dumps(
            state["findings"],
            indent=2,
            ensure_ascii=False,
        )

        analysis_context = json.dumps(
            state["analysis"],
            indent=2,
            ensure_ascii=False,
        )

        response = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Critic Agent in ResearchForge.\n\n"
                        "Audit the research before publication.\n\n"
                        "Check for:\n"
                        "- unsupported claims\n"
                        "- contradictions\n"
                        "- weak or low-quality evidence\n"
                        "- missing perspectives\n"
                        "- conclusions that exceed the retrieved evidence\n\n"
                        "Pay special attention to source types. "
                        "Official and academic sources generally provide "
                        "stronger evidence than commercial or community "
                        "sources, but do not automatically treat any source "
                        "as authoritative without considering its content."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Research question:\n\n"
                        f"{state['question']}\n\n"
                        f"Analysis:\n\n"
                        f"{analysis_context}\n\n"
                        f"Underlying findings:\n\n"
                        f"{findings_context}"
                    )
                ),
            ]
        )

        return {
            "critique": response.model_dump(
                mode="json"
            ),
            "events": [
                {
                    "type": "agent_completed",
                    "agent": "critic",
                    "detail": (
                        "Evidence audit completed."
                    ),
                }
            ],
        }

    return critic