"""The LangGraph agent: routing, retrieval, generation, and verification.

Graph topology::

    START -> classify_intent
      |-- "retrieval" -> retrieve --(chunks found)--> generate -> verify
      |                     |                                       |-- grounded --> finalize -> END
      |                     '--(no chunks)--> refuse -> END         '-- ungrounded -> refuse -> END
      '-- "chat" -> chat_respond <-> tools -> END

The RAG branch never appends unverified text to the conversation: ``generate``
writes a *draft*, and only ``finalize`` (after the groundedness check) or
``refuse`` produce the message the user sees. The chat branch is a standard
tool-calling loop (``chat_respond`` emits tool calls, ``tools`` executes them,
control returns to ``chat_respond`` until it answers in plain text).

Dependencies (LLM, retriever, checkpointer) are injected into ``build_agent``
so tests can run the full graph with fakes; ``create_agent_from_settings``
wires the production instances.
"""

import sqlite3
from typing import Literal, Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from agentic_rag.agent.citations import extract_citations
from agentic_rag.agent.prompts import (
    CHAT_PROMPT,
    GENERATION_PROMPT,
    GROUNDEDNESS_PROMPT,
    INTENT_PROMPT,
    REFUSAL_MESSAGE,
    format_context,
)
from agentic_rag.agent.state import AgentState
from agentic_rag.agent.tools import TOOLS
from agentic_rag.config import Settings
from agentic_rag.retrieval.hybrid import HybridRetriever


class Retriever(Protocol):
    """The one method the graph needs from a retriever."""

    def retrieve(self, query: str) -> list[Document]: ...


class IntentDecision(BaseModel):
    """Structured output of the intent classifier."""

    intent: Literal["retrieval", "chat"]


class GroundednessVerdict(BaseModel):
    """Structured output of the groundedness judge."""

    grounded: bool


def _last_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return str(message.content)
    return ""


def build_agent(
    llm: BaseChatModel,
    retriever: Retriever,
    checkpointer: BaseCheckpointSaver | None = None,
    corpus_description: str = "technical documentation",
) -> CompiledStateGraph:
    """Assemble and compile the agent graph from injected dependencies."""
    intent_classifier = llm.with_structured_output(IntentDecision)
    groundedness_judge = llm.with_structured_output(GroundednessVerdict)
    chat_llm = llm.bind_tools(TOOLS)

    # --- Nodes (closures over the injected dependencies) ---

    def classify_intent(state: AgentState) -> dict:
        decision = intent_classifier.invoke(
            [
                SystemMessage(INTENT_PROMPT.format(corpus_description=corpus_description)),
                *state["messages"][-1:],
            ]
        )
        # Reset all per-turn fields so nothing leaks from the previous turn.
        return {
            "intent": decision.intent,
            "context": [],
            "draft": "",
            "grounded": None,
            "sources": [],
        }

    def retrieve(state: AgentState) -> dict:
        documents = retriever.retrieve(_last_user_text(state["messages"]))
        context = [
            {
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "text": doc.page_content,
            }
            for doc in documents
        ]
        return {"context": context}

    def generate(state: AgentState) -> dict:
        system = SystemMessage(
            GENERATION_PROMPT.format(context=format_context(state["context"]))
        )
        draft = llm.invoke([system, *state["messages"]])
        return {"draft": str(draft.content)}

    def verify(state: AgentState) -> dict:
        verdict = groundedness_judge.invoke(
            [
                SystemMessage(GROUNDEDNESS_PROMPT),
                SystemMessage(
                    f"Context excerpts:\n{format_context(state['context'])}\n\n"
                    f"Draft answer:\n{state['draft']}"
                ),
            ]
        )
        return {"grounded": verdict.grounded}

    def finalize(state: AgentState) -> dict:
        return {
            "messages": [AIMessage(state["draft"])],
            "sources": extract_citations(state["draft"], state["context"]),
        }

    def refuse(state: AgentState) -> dict:
        return {"messages": [AIMessage(REFUSAL_MESSAGE)], "sources": []}

    def chat_respond(state: AgentState) -> dict:
        response = chat_llm.invoke([SystemMessage(CHAT_PROMPT), *state["messages"]])
        return {"messages": [response]}

    # --- Routers for conditional edges ---

    def route_intent(state: AgentState) -> str:
        return state["intent"]

    def route_after_retrieve(state: AgentState) -> str:
        return "generate" if state["context"] else "refuse"

    def route_groundedness(state: AgentState) -> str:
        return "finalize" if state["grounded"] else "refuse"

    # --- Wiring ---

    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("verify", verify)
    builder.add_node("finalize", finalize)
    builder.add_node("refuse", refuse)
    builder.add_node("chat_respond", chat_respond)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {"retrieval": "retrieve", "chat": "chat_respond"},
    )
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "refuse": "refuse"},
    )
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges(
        "verify",
        route_groundedness,
        {"finalize": "finalize", "refuse": "refuse"},
    )
    builder.add_edge("finalize", END)
    builder.add_edge("refuse", END)
    # tools_condition routes to the "tools" node when the LLM emitted tool
    # calls, and to END when it answered in plain text.
    builder.add_conditional_edges("chat_respond", tools_condition)
    builder.add_edge("tools", "chat_respond")

    return builder.compile(checkpointer=checkpointer)


def create_agent_from_settings(settings: Settings) -> CompiledStateGraph:
    """Wire the production agent: OpenAI LLM, hybrid retriever, SQLite memory."""
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0,  # deterministic-ish answers; sampling adds nothing to RAG
    )
    retriever = HybridRetriever.from_settings(settings)
    # check_same_thread=False: FastAPI serves requests from a threadpool, and
    # SqliteSaver guards the connection with its own lock.
    connection = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
    return build_agent(
        llm,
        retriever,
        checkpointer=SqliteSaver(connection),
        corpus_description=settings.corpus_description,
    )
