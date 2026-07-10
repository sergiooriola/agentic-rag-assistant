"""Evaluate the assistant against an annotated question set.

Two metrics, separating the two ways the system can fail:

- **Retrieval hit rate**: is the expected source among the retrieved chunks?
  (Free — no LLM calls. If this fails, the problem is chunking/search.)
- **Citation accuracy**: does the final answer cite the expected source
  (and page, when annotated)? (Runs the full agent — costs API tokens.
  If retrieval hit but citation missed, the problem is generation/prompting.)

Requires a built index (run scripts/ingest.py first) and OPENAI_API_KEY.

Usage:
    python scripts/evaluate.py [--dataset evaluation/dataset.json]
"""

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agentic_rag.agent.graph import build_agent
from agentic_rag.agent.prompts import REFUSAL_MESSAGE
from agentic_rag.config import get_settings
from agentic_rag.retrieval.hybrid import HybridRetriever


def _citation_matches(source: dict[str, Any], case: dict[str, Any]) -> bool:
    if source["source"] != case["expected_source"]:
        return False
    expected_page = case.get("expected_page")
    return expected_page is None or source.get("page") == expected_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and citation accuracy")
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/dataset.json"),
        help="path to the annotated question set",
    )
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))

    settings = get_settings()
    retriever = HybridRetriever.from_settings(settings)
    # No checkpointer: every question is an independent, memoryless run, and
    # evaluation must not pollute the production conversation database.
    llm = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)
    agent = build_agent(llm, retriever, corpus_description=settings.corpus_description)

    retrieval_hits = citation_hits = refusals = 0
    print(f"{'#':>2}  {'retrieval':9}  {'citation':8}  question")
    for number, case in enumerate(cases, start=1):
        retrieved = retriever.retrieve(case["question"])
        retrieval_hit = any(
            doc.metadata.get("source") == case["expected_source"] for doc in retrieved
        )

        result = agent.invoke({"messages": [HumanMessage(case["question"])]})
        answer = str(result["messages"][-1].content)
        refused = answer == REFUSAL_MESSAGE
        citation_hit = any(_citation_matches(s, case) for s in result.get("sources", []))

        retrieval_hits += retrieval_hit
        citation_hits += citation_hit
        refusals += refused
        citation_label = "hit" if citation_hit else ("refused" if refused else "MISS")
        print(
            f"{number:>2}  {'hit' if retrieval_hit else 'MISS':9}  {citation_label:8}  "
            f"{case['question']}"
        )

    total = len(cases)
    print(f"\nRetrieval hit rate: {retrieval_hits}/{total} ({100 * retrieval_hits / total:.0f}%)")
    print(f"Citation accuracy:  {citation_hits}/{total} ({100 * citation_hits / total:.0f}%)")
    print(f"Refusals:           {refusals}/{total}")


if __name__ == "__main__":
    main()
