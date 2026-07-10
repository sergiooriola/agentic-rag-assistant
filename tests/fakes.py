"""Scriptable fakes shared by graph and API tests.

``FakeLLM`` covers every role the graph asks of the LLM: intent classification
and groundedness judging (via ``with_structured_output``), RAG generation
(via ``invoke``), and the tool-calling chat branch (via ``bind_tools``).
``echo_history_length`` makes the chat branch reply with the size of the
conversation it received — the cheapest way to prove that checkpointer memory
actually flows across separate invocations.
"""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag.agent.graph import GroundednessVerdict, IntentDecision


class _Invoker:
    def __init__(self, respond: Callable) -> None:
        self._respond = respond

    def invoke(self, messages: object) -> object:
        return self._respond(messages)


class FakeLLM:
    """Returns scripted outputs for each role the graph uses."""

    def __init__(
        self,
        intent: str = "chat",
        grounded: bool = True,
        generation: str = "generated answer",
        chat_replies: list[AIMessage] | None = None,
        echo_history_length: bool = False,
    ) -> None:
        self.intent = intent
        self.grounded = grounded
        self.generation = generation
        self.chat_replies = chat_replies or [AIMessage("hi there!")]
        self.echo_history_length = echo_history_length

    def with_structured_output(self, schema: type) -> _Invoker:
        if schema is IntentDecision:
            return _Invoker(lambda _msgs: IntentDecision(intent=self.intent))
        return _Invoker(lambda _msgs: GroundednessVerdict(grounded=self.grounded))

    def bind_tools(self, _tools: list) -> _Invoker:
        return _Invoker(self._chat)

    def _chat(self, messages: list) -> AIMessage:
        if self.echo_history_length:
            # messages = [system prompt, *conversation]; report conversation size.
            return AIMessage(f"history={len(messages) - 1}")
        return self.chat_replies.pop(0)

    def invoke(self, _messages: list) -> AIMessage:
        return AIMessage(self.generation)


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str) -> list[Document]:
        return self._documents
