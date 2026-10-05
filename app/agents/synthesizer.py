"""Final synthesis agent for ResearchForge."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import ResearchReport


def create_synthesizer(
    llm: ChatOpenAI,
):
    """Create the Synthesizer agent."""

    structured_llm = llm.with_structured_output(
        ResearchReport
    )

    def synthesizer(
        state: dict,
    ) -> dict:
        """Create the final research report."""

        print("\n" + "=" * 70)
        print("[SYNTHESIZER]")
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

        critique_context = json.dumps(
            state["critique"],
            indent=2,
            ensure_ascii=False,
        )

        response = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Synthesizer Agent in ResearchForge.\n\n"
                        "Create a professional research brief from the "
                        "retrieved findings, cross-source analysis, and critique.\n\n"
                        "Requirements:\n"
                        "- Base factual claims on retrieved evidence.\n"
                        "- Address significant contradictions.\n"
                        "- Make uncertainty explicit.\n"
                        "- Do not introduce unsupported claims.\n"
                        "- Do not fabricate citations or URLs.\n"
                        "- Produce a clear, readable report."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n\n"
                        f"{state['question']}\n\n"
                        f"Findings:\n\n"
                        f"{findings_context}\n\n"
                        f"Analysis:\n\n"
                        f"{analysis_context}\n\n"
                        f"Critique:\n\n"
                        f"{critique_context}"
                    )
                ),
            ]
        )

        unique_sources: dict[str, dict] = {}

        for finding in state["findings"]:
            for source in finding.get(
                "sources",
                [],
            ):
                url = source.get(
                    "url",
                    "",
                )

                if (
                    url
                    and url not in unique_sources
                ):
                    unique_sources[url] = source

        return {
            "report": response.model_dump(
                mode="json"
            ),
            "sources": list(
                unique_sources.values()
            ),
            "events": [
                {
                    "type": "agent_completed",
                    "agent": "synthesizer",
                    "detail": (
                        "Final research brief completed."
                    ),
                }
            ],
        }

    return synthesizer