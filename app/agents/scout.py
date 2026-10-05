"""Tool-using Scout agent for ResearchForge."""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.models.schemas import (
    ResearchFinding,
    ScoutSummary,
    Source,
    ToolTrace,
)
from app.tools.extract_webpage import extract_webpage
from app.tools.web_search import web_search


class ScoutAgentState(TypedDict):
    """State for one Scout's internal tool loop."""

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]


def _route_after_model(
    state: ScoutAgentState,
) -> str:
    """Route to tools when the Scout requested tool execution."""

    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"

    return END


def _parse_tool_outputs(
    tool_outputs: list[str],
) -> tuple[list[Source], dict[str, Source]]:
    """Build source metadata while preserving search-result quality data."""

    sources_by_url: dict[str, Source] = {}

    for output in tool_outputs:
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            continue

        for item in parsed.get("results", []):
            url = str(item.get("url", "")).strip()

            if not url:
                continue

            existing = sources_by_url.get(url)

            title = str(
                item.get("title")
                or (existing.title if existing else url)
            )

            snippet = str(
                item.get("content")
                or (existing.snippet if existing else "")
            )[:600]

            relevance_score = float(
                item.get(
                    "score",
                    existing.relevance_score
                    if existing
                    else 0.0,
                )
                or 0.0
            )

            source_type = str(
                item.get(
                    "source_type",
                    existing.source_type
                    if existing
                    else "unknown",
                )
                or "unknown"
            )

            sources_by_url[url] = Source(
                title=title,
                url=url,
                snippet=snippet,
                relevance_score=relevance_score,
                source_type=source_type,
            )

    return (
        list(sources_by_url.values()),
        sources_by_url,
    )


def create_scout(
    llm: ChatOpenAI,
):
    """Create the tool-using Scout agent."""

    tools = [
        web_search,
        extract_webpage,
    ]

    model_with_tools = llm.bind_tools(
        tools
    )

    structured_llm = llm.with_structured_output(
        ScoutSummary
    )

    def call_model(
        state: ScoutAgentState,
    ) -> dict:
        """Ask the Scout to decide its next research action."""

        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a Scout Agent in ResearchForge.\n\n"
                        "Your job is to investigate ONE assigned research "
                        "task using the available web tools.\n\n"
                        "Tools:\n"
                        "- web_search: search for relevant evidence.\n"
                        "- extract_webpage: inspect an important source "
                        "in greater detail.\n\n"
                        "Behavior:\n"
                        "- Start by searching.\n"
                        "- Inspect important sources when necessary.\n"
                        "- Search again when the evidence is incomplete.\n"
                        "- Stop when you have sufficient evidence.\n\n"
                        "Evidence rules:\n"
                        "- Never invent facts.\n"
                        "- Never invent URLs.\n"
                        "- Prefer official and academic material when "
                        "available.\n"
                        "- Treat community posts as anecdotal evidence, "
                        "not authoritative evidence.\n"
                        "- Remain focused on the assigned task."
                    )
                ),
                *state["messages"],
            ]
        )

        return {
            "messages": [
                response
            ]
        }

    tool_node = ToolNode(
        tools
    )

    scout_graph = StateGraph(
        ScoutAgentState
    )

    scout_graph.add_node(
        "model",
        call_model,
    )

    scout_graph.add_node(
        "tools",
        tool_node,
    )

    scout_graph.add_edge(
        START,
        "model",
    )

    scout_graph.add_conditional_edges(
        "model",
        _route_after_model,
    )

    scout_graph.add_edge(
        "tools",
        "model",
    )

    compiled_scout = scout_graph.compile()

    def scout(
        state: dict,
    ) -> dict:
        """Execute one complete research assignment."""

        question = state["question"]
        task = state["task"]

        print("\n" + "-" * 70)
        print(
            f"[SCOUT] {task['topic']}"
        )
        print("-" * 70)

        initial_message = HumanMessage(
            content=(
                f"Overall question:\n\n"
                f"{question}\n\n"
                f"Assigned task:\n\n"
                f"{task['topic']}\n\n"
                f"Search query:\n\n"
                f"{task['search_query']}\n\n"
                f"Rationale:\n\n"
                f"{task['rationale']}\n\n"
                "Begin the investigation."
            )
        )

        result = compiled_scout.invoke(
            {
                "messages": [
                    initial_message
                ]
            }
        )

        messages = result["messages"]

        tool_trace: list[ToolTrace] = []
        tool_outputs: list[str] = []

        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    tool_trace.append(
                        ToolTrace(
                            tool_name=tool_call["name"],
                            input_summary=str(
                                tool_call.get(
                                    "args",
                                    {},
                                )
                            )[:300],
                        )
                    )

            if message.type == "tool":
                tool_outputs.append(
                    str(message.content)
                )

        if not tool_outputs:
            raise RuntimeError(
                f"Scout produced no tool results for: "
                f"{task['topic']}"
            )

        evidence_context = "\n\n".join(
            tool_outputs
        )

        summary = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the evidence summarization component "
                        "of a research Scout.\n\n"
                        "Use ONLY the supplied tool evidence.\n"
                        "Do not add outside knowledge.\n"
                        "Do not fabricate sources.\n"
                        "State uncertainty when evidence is limited."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Research task:\n\n"
                        f"{task['topic']}\n\n"
                        f"Retrieved evidence:\n\n"
                        f"{evidence_context}"
                    )
                ),
            ]
        )

        sources, _sources_by_url = _parse_tool_outputs(
            tool_outputs
        )

        finding = ResearchFinding(
            task=task["topic"],
            summary=summary.summary,
            key_points=summary.key_points,
            limitations=summary.limitations,
            sources=sources,
            tool_trace=tool_trace,
        )

        print(
            f"Tool calls: {len(tool_trace)}"
        )
        print(
            f"Sources gathered: {len(sources)}"
        )

        return {
            "findings": [
                finding.model_dump(
                    mode="json"
                )
            ],
            "events": [
                {
                    "type": "agent_completed",
                    "agent": "scout",
                    "task": task["topic"],
                    "detail": (
                        f"{len(sources)} sources, "
                        f"{len(tool_trace)} tool calls."
                    ),
                }
            ],
        }

    return scout
