"""Planner agent for ResearchForge."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.models.schemas import ResearchPlan


def create_planner(
    llm: ChatOpenAI,
):
    """Create the Planner agent."""

    structured_llm = llm.with_structured_output(
        ResearchPlan
    )

    def planner(
        state: dict,
    ) -> dict:
        """Decompose the research question into independent tasks."""

        question = state["question"]

        response = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Planner Agent for ResearchForge.\n\n"
                        "Decompose the user's research question into "
                        "3 to 4 independent, complementary research tasks.\n\n"
                        "Requirements:\n"
                        "- Each task must investigate a distinct dimension.\n"
                        "- Avoid overlapping tasks.\n"
                        "- Create focused web search queries.\n"
                        "- Cover the dimensions most necessary to answer "
                        "the user's question.\n"
                        "- Do not answer the question."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Research question:\n\n{question}"
                    )
                ),
            ]
        )

        print("\n" + "=" * 70)
        print("[PLANNER]")
        print("=" * 70)

        for index, task in enumerate(
            response.tasks,
            start=1,
        ):
            print(
                f"{index}. {task.topic}"
            )
            print(
                f"   Query: {task.search_query}"
            )

        return {
            "plan": [
                task.model_dump(
                    mode="json"
                )
                for task in response.tasks
            ],
            "events": [
                {
                    "type": "agent_completed",
                    "agent": "planner",
                    "detail": (
                        f"Created {len(response.tasks)} research tasks."
                    ),
                }
            ],
        }

    return planner