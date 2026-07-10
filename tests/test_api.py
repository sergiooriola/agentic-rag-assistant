"""API tests with an injected fake agent — no OpenAI, no ChromaDB.

TestClient is used as a context manager so the lifespan handler runs and the
agent factory is exercised exactly as in production.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag import __version__
from agentic_rag.api.app import create_app


class FakeAgent:
    """Records invocations and returns a canned grounded answer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail = False

    def invoke(self, payload: dict, config: dict) -> dict:
        if self.fail:
            raise RuntimeError("upstream LLM exploded")
        self.calls.append({"payload": payload, "config": config})
        return {
            "messages": [AIMessage("PUT replaces the whole resource [1].")],
            "sources": [
                {"index": 1, "chunk_id": "http_essentials.md:p0:c0",
                 "source": "http_essentials.md", "page": None},
            ],
        }


@pytest.fixture()
def fake_agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture()
def client(fake_agent: FakeAgent) -> Iterator[TestClient]:
    app = create_app(agent_factory=lambda: fake_agent)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_chat_returns_answer_and_sources(client: TestClient, fake_agent: FakeAgent) -> None:
    response = client.post("/chat", json={"question": "What does PUT do?", "session_id": "s1"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("PUT replaces")
    assert body["session_id"] == "s1"
    assert body["sources"][0]["source"] == "http_essentials.md"


def test_chat_threads_session_id_into_the_agent(
    client: TestClient, fake_agent: FakeAgent
) -> None:
    client.post("/chat", json={"question": "hi", "session_id": "session-abc"})
    config = fake_agent.calls[0]["config"]
    assert config["configurable"]["thread_id"] == "session-abc"


def test_chat_validates_input(client: TestClient) -> None:
    assert client.post("/chat", json={"question": "", "session_id": "s"}).status_code == 422
    assert client.post("/chat", json={"question": "hi"}).status_code == 422


def test_chat_agent_failure_maps_to_502(client: TestClient, fake_agent: FakeAgent) -> None:
    fake_agent.fail = True
    response = client.post("/chat", json={"question": "hi", "session_id": "s1"})
    assert response.status_code == 502
    assert "Agent invocation failed" in response.json()["detail"]


def test_ingest_reindexes_and_rebuilds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_docs = [Document(page_content="x", metadata={"source": "a.md"})]
    monkeypatch.setattr("agentic_rag.api.app.load_documents", lambda _dir: fake_docs)
    monkeypatch.setattr("agentic_rag.api.app.chunk_documents", lambda docs, _s, _o: docs)
    monkeypatch.setattr("agentic_rag.api.app.index_chunks", lambda chunks, _settings: len(chunks))

    built_agents: list[FakeAgent] = []

    def factory() -> FakeAgent:
        agent = FakeAgent()
        built_agents.append(agent)
        return agent

    with TestClient(create_app(agent_factory=factory)) as test_client:
        response = test_client.post("/ingest")
        assert response.status_code == 200
        assert response.json() == {"documents_loaded": 1, "chunks_indexed": 1}
        # One agent from startup, a second one rebuilt after re-indexing.
        assert len(built_agents) == 2
