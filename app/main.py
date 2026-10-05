"""Command-line entry point for ResearchForge."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.graph.workflow import build_graph


def create_llm() -> ChatOpenAI:
    """Create the configured OpenAI chat model."""

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-5-mini",
    )

    return ChatOpenAI(
        model=model_name,
    )


def main() -> None:
    """Run a ResearchForge research session."""

    load_dotenv()

    question = input(
        "\nWhat would you like to research?\n> "
    ).strip()

    if not question:
        raise ValueError(
            "Research question cannot be empty."
        )

    llm = create_llm()

    graph = build_graph(
        llm
    )

    result = graph.invoke(
        {
            "question": question,
            "plan": [],
            "findings": [],
            "analysis": {},
            "critique": {},
            "report": {},
            "sources": [],
            "events": [],
        }
    )

    report = result["report"]

    print("\n")
    print("=" * 80)
    print(
        report["title"].upper()
    )
    print("=" * 80)

    print("\nEXECUTIVE SUMMARY")
    print("-" * 80)
    print(
        report["executive_summary"]
    )

    for section in report["sections"]:
        print(
            f"\n{section['heading'].upper()}"
        )
        print("-" * 80)
        print(
            section["content"]
        )

    if report["caveats"]:
        print("\nCAVEATS")
        print("-" * 80)

        for caveat in report["caveats"]:
            print(
                f"• {caveat}"
            )

    print("\nSOURCES")
    print("-" * 80)

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"{index}. {source['title']}"
        )

        print(
            f"   Type: {source.get('source_type', 'unknown')}"
        )

        print(
            f"   Relevance: "
            f"{source.get('relevance_score', 0.0):.3f}"
        )

        print(
            f"   {source['url']}"
        )

    print("\nTOOL TRACE")
    print("-" * 80)

    for finding in result["findings"]:
        print(
            f"\n{finding['task']}"
        )

        for trace in finding["tool_trace"]:
            print(
                f"  → {trace['tool_name']}"
            )


if __name__ == "__main__":
    main()