"""End-to-end graph tests with a scriptable fake LLM — no API calls.

The fake covers the three roles the graph asks of the LLM (intent
classification, generation, groundedness judging) plus the tool-calling chat
branch, so every path through the graph is exercised with real LangGraph
execution and real checkpointing.
"""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.agent.graph import GroundednessVerdict, IntentDecision, build_agent
from agentic_rag.agent.prompts import REFUSAL_MESSAGE

DOCS = [
    Document(
        page_content="POST submits data and typically creates a new resource.",
        metadata={"chunk_id": "c1", "source": "http_essentials.md"},
    ),
    Document(
        page_content="DELETE removes the target resource and is idempotent.",
        metadata={"chunk_id": "c2", "source": "rest_api_design.pdf", "page": 1},
    ),
]


class _Invoker:
    def __init__(self, respond: Callable) -> None:
        self._respond = respond

    def invoke(self, _input: object) -> object:
        return self._respond()


class FakeLLM:
    """Returns scripted outputs for each role the graph uses."""

    def __init__(
        self,
        intent: str = "chat",
        grounded: bool = True,
        generation: str = "generated answer",
        chat_replies: list[AIMessage] | None = None,
    ) -> None:
        self.intent = intent
        self.grounded = grounded
        self.generation = generation
        self.chat_replies = chat_replies or [AIMessage("hi there!")]

    def with_structured_output(self, schema: type) -> _Invoker:
        if schema is IntentDecision:
            return _Invoker(lambda: IntentDecision(intent=self.intent))
        return _Invoker(lambda: GroundednessVerdict(grounded=self.grounded))

    def bind_tools(self, _tools: list) -> _Invoker:
        return _Invoker(lambda: self.chat_replies.pop(0))

    def invoke(self, _messages: list) -> AIMessage:
        return AIMessage(self.generation)


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str) -> list[Document]:
        return self._documents


def _agent(llm: FakeLLM, docs: list[Document] = DOCS):
    return build_agent(llm, FakeRetriever(docs), checkpointer=MemorySaver())


def _run(agent, text: str, thread: str = "t1") -> dict:
    return agent.invoke(
        {"messages": [HumanMessage(text)]},
        config={"configurable": {"thread_id": thread}},
    )


def test_chat_intent_skips_retrieval() -> None:
    result = _run(_agent(FakeLLM(intent="chat")), "hello!")
    assert result["messages"][-1].content == "hi there!"
    assert result["context"] == []
    assert result["sources"] == []


def test_grounded_answer_is_finalized_with_citations() -> None:
    llm = FakeLLM(
        intent="retrieval",
        grounded=True,
        generation="POST creates resources [1]. DELETE is idempotent [2].",
    )
    result = _run(_agent(llm), "what does POST do?")
    assert result["messages"][-1].content.startswith("POST creates")
    assert [s["chunk_id"] for s in result["sources"]] == ["c1", "c2"]
    assert result["sources"][1]["page"] == 1


def test_ungrounded_answer_is_refused() -> None:
    llm = FakeLLM(intent="retrieval", grounded=False, generation="made-up claim [1]")
    result = _run(_agent(llm), "what does PATCH do?")
    assert result["messages"][-1].content == REFUSAL_MESSAGE
    assert result["sources"] == []


def test_empty_retrieval_refuses_without_generating() -> None:
    result = _run(_agent(FakeLLM(intent="retrieval"), docs=[]), "unknown topic?")
    assert result["messages"][-1].content == REFUSAL_MESSAGE


def test_chat_branch_executes_tool_calls() -> None:
    llm = FakeLLM(
        intent="chat",
        chat_replies=[
            AIMessage(
                "",
                tool_calls=[{"name": "calculator", "args": {"expression": "6 * 7"}, "id": "1"}],
            ),
            AIMessage("The answer is 42."),
        ],
    )
    result = _run(_agent(llm), "what is 6 times 7?")
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages and tool_messages[0].content == "42"  # real tool ran
    assert result["messages"][-1].content == "The answer is 42."


def test_memory_is_isolated_per_thread() -> None:
    agent = _agent(FakeLLM(intent="chat", chat_replies=[AIMessage("a"), AIMessage("b")]))
    _run(agent, "first message", thread="session-1")
    result = _run(agent, "second message", thread="session-1")
    # Same thread accumulates: 2 human + 2 AI messages.
    assert len(result["messages"]) == 4

    agent2 = _agent(FakeLLM(intent="chat"))
    fresh = _run(agent2, "hello", thread="session-2")
    assert len(fresh["messages"]) == 2
