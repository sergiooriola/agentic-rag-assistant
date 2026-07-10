"""Shared state that flows through the agent graph.

``messages`` is the conversation history: the ``add_messages`` reducer appends
new messages instead of replacing the list, and the checkpointer persists it
between invocations — that combination is what gives the agent per-session
memory. Every other field is per-turn scratch space (reset by the intent
classifier at the start of each turn) used to pass data between nodes.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Literal["retrieval", "chat"]
    # Retrieved chunks, serialized as plain dicts (chunk_id, source, page, text)
    # so the state stays JSON-friendly for checkpointing and inspection.
    context: list[dict[str, Any]]
    # Candidate answer awaiting groundedness verification.
    draft: str
    grounded: bool | None
    # Citations backing the final answer (subset of context, in citation order).
    sources: list[dict[str, Any]]
