"""Integration tests: HTTP request -> FastAPI -> real LangGraph agent -> fake LLM.

Unlike test_api.py (which fakes the whole agent), these tests run the real
graph — routing, groundedness gate, citation parsing, and checkpointing — so
they catch wiring bugs between the API layer and the agent that unit tests
cannot see. Only the LLM and the retriever are fakes.
"""

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.agent.graph import build_agent
from agentic_rag.agent.prompts import REFUSAL_MESSAGE
from agentic_rag.api.app import create_app
from fakes import FakeLLM, FakeRetriever

DOCS = [
    Document(
        page_content="PUT replaces the target resource entirely with the request payload.",
        metadata={"chunk_id": "http_essentials.md:p0:c0", "source": "http_essentials.md"},
    ),
    Document(
        page_content="Cursor pagination returns an opaque token pointing at the last seen item.",
        metadata={"chunk_id": "rest_api_design.pdf:p2:c0", "source": "rest_api_design.pdf",
                  "page": 2},
    ),
]


def _client(llm: FakeLLM) -> TestClient:
    agent = build_agent(llm, FakeRetriever(DOCS), checkpointer=MemorySaver())
    return TestClient(create_app(agent_factory=lambda: agent))


def test_rag_flow_end_to_end() -> None:
    llm = FakeLLM(
        intent="retrieval",
        grounded=True,
        generation="PUT replaces the entire resource [1], unlike cursor pagination [2].",
    )
    with _client(llm) as client:
        response = client.post(
            "/chat", json={"question": "What does PUT do?", "session_id": "it-rag"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("PUT replaces")
    assert [s["source"] for s in body["sources"]] == ["http_essentials.md", "rest_api_design.pdf"]
    assert body["sources"][1]["page"] == 2


def test_ungrounded_draft_refused_through_the_api() -> None:
    llm = FakeLLM(intent="retrieval", grounded=False, generation="fabricated claim [1]")
    with _client(llm) as client:
        response = client.post(
            "/chat", json={"question": "What does PATCH do?", "session_id": "it-refuse"}
        )
    assert response.status_code == 200
    assert response.json()["answer"] == REFUSAL_MESSAGE
    assert response.json()["sources"] == []


def test_conversation_memory_survives_across_http_requests() -> None:
    # The fake chat LLM replies with the length of the history it received,
    # so growth proves the checkpointer restored earlier turns between
    # independent HTTP requests.
    llm = FakeLLM(intent="chat", echo_history_length=True)
    with _client(llm) as client:
        first = client.post("/chat", json={"question": "hello", "session_id": "it-mem"})
        second = client.post("/chat", json={"question": "again", "session_id": "it-mem"})
        other = client.post("/chat", json={"question": "hello", "session_id": "it-other"})

    assert first.json()["answer"] == "history=1"  # just this turn's message
    assert second.json()["answer"] == "history=3"  # human + ai + human restored
    assert other.json()["answer"] == "history=1"  # separate session starts clean
