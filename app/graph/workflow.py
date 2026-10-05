"""LangGraph workflow for ResearchForge."""

from operator import add
from typing import Annotated, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.analyst import create_analyst
from app.agents.critic import create_critic
from app.agents.planner import create_planner
from app.agents.scout import create_scout
from app.agents.synthesizer import create_synthesizer


class ResearchState(TypedDict):
    """Shared state carried through the ResearchForge graph."""

    question: str

    plan: list[dict]

    findings: Annotated[
        list[dict],
        add,
    ]

    analysis: dict

    critique: dict

    report: dict

    sources: list[dict]

    events: Annotated[
        list[dict],
        add,
    ]

    task: NotRequired[dict]


def route_to_scouts(
    state: ResearchState,
) -> list[Send]:
    """Fan planner tasks out into independent Scout executions."""

    return [
        Send(
            "scout",
            {
                "question": state["question"],
                "task": task,
            },
        )
        for task in state["plan"]
    ]


def build_graph(
    llm: ChatOpenAI,
):
    """Build and compile the complete ResearchForge graph."""

    planner = create_planner(
        llm
    )

    scout = create_scout(
        llm
    )

    analyst = create_analyst(
        llm
    )

    critic = create_critic(
        llm
    )

    synthesizer = create_synthesizer(
        llm
    )

    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "planner",
        planner,
    )

    builder.add_node(
        "scout",
        scout,
    )

    builder.add_node(
        "analyst",
        analyst,
    )

    builder.add_node(
        "critic",
        critic,
    )

    builder.add_node(
        "synthesizer",
        synthesizer,
    )

    builder.add_edge(
        START,
        "planner",
    )

    builder.add_conditional_edges(
        "planner",
        route_to_scouts,
    )

    builder.add_edge(
        "scout",
        "analyst",
    )

    builder.add_edge(
        "analyst",
        "critic",
    )

    builder.add_edge(
        "critic",
        "synthesizer",
    )

    builder.add_edge(
        "synthesizer",
        END,
    )

    return builder.compile()