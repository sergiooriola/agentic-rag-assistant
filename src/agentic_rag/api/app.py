"""FastAPI application exposing the agent.

The agent is built ONCE at startup (via the lifespan handler) and kept in
``app.state``: building it per request would rebuild the BM25 index and
reopen the checkpoint database every time. ``create_app`` accepts an
``agent_factory`` so tests can inject a fake agent without touching OpenAI
or ChromaDB; the same factory is reused by ``/ingest`` to swap in a fresh
agent after the index is rebuilt.

Endpoints are plain ``def`` (not ``async def``) on purpose: the agent and the
ingestion pipeline are blocking, and FastAPI runs sync handlers in a thread
pool, which keeps the event loop responsive without any async plumbing.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage

from agentic_rag import __version__
from agentic_rag.api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from agentic_rag.config import get_settings
from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.indexer import index_chunks
from agentic_rag.ingestion.loaders import load_documents

AgentFactory = Callable[[], Any]


def _default_agent_factory() -> Any:
    # Imported lazily so that importing this module (e.g. for OpenAPI docs or
    # tests) never requires an API key or an existing index.
    from agentic_rag.agent.graph import create_agent_from_settings

    return create_agent_from_settings(get_settings())


def create_app(agent_factory: AgentFactory | None = None) -> FastAPI:
    """Build the FastAPI app; ``agent_factory`` is injectable for tests."""
    factory = agent_factory or _default_agent_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent_factory = factory
        app.state.agent = factory()
        yield

    app = FastAPI(
        title="agentic-rag-assistant",
        description="RAG agent with hybrid retrieval, source citations, and groundedness checks.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        agent = request.app.state.agent
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(payload.question)]},
                config={"configurable": {"thread_id": payload.session_id}},
            )
        except Exception as exc:  # LLM/API failures surface as a gateway error
            raise HTTPException(status_code=502, detail=f"Agent invocation failed: {exc}") from exc

        return ChatResponse(
            answer=str(result["messages"][-1].content),
            sources=result.get("sources", []),
            session_id=payload.session_id,
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: Request) -> IngestResponse:
        settings = get_settings()
        try:
            documents = load_documents(settings.data_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        indexed = index_chunks(chunks, settings)
        # The BM25 index is derived from Chroma at agent build time, so the
        # agent must be rebuilt for the new corpus to become searchable.
        request.app.state.agent = request.app.state.agent_factory()
        return IngestResponse(documents_loaded=len(documents), chunks_indexed=indexed)

    return app


# Module-level app for `uvicorn agentic_rag.api.app:app`. The agent is only
# built when the server actually starts (lifespan), not at import time.
app = create_app()
