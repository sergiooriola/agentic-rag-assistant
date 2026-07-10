"""Prompts for every LLM role in the graph, kept in one place for review."""

from typing import Any

INTENT_PROMPT = """You are the router of a RAG assistant. The assistant has an indexed \
document corpus about: {corpus_description}.

Classify the user's message:
- "retrieval": answering it requires looking up information in the corpus.
- "chat": greetings, small talk, questions about the conversation itself, arithmetic, \
dates, or anything unrelated to the corpus.

Classify only; do not answer the message."""

GENERATION_PROMPT = """You are a documentation assistant. Answer the user's question using \
ONLY the numbered context excerpts below. After each factual claim, cite the excerpt that \
supports it with its number in brackets, e.g. [1] or [2].

Rules:
- Never use knowledge that is not in the excerpts.
- If the excerpts do not contain the answer, say so explicitly instead of guessing.
- Keep answers concise and factual.

Context excerpts:
{context}"""

GROUNDEDNESS_PROMPT = """You are a strict fact-checker for a RAG system. You will receive \
context excerpts and a draft answer. Decide whether EVERY factual claim in the draft is \
directly supported by the excerpts.

Return grounded=false if any claim goes beyond the excerpts, even slightly. Citation \
markers like [1] are formatting, not evidence — verify against the actual excerpt text."""

CHAT_PROMPT = """You are the conversational side of a documentation assistant. Handle \
greetings and casual conversation briefly and warmly. Use the available tools when the \
user asks for arithmetic (calculator) or today's date (current_date). If asked what you \
can do, say you answer questions about the indexed documentation."""

REFUSAL_MESSAGE = (
    "I couldn't find enough information in the indexed documents to answer that reliably. "
    "Try rephrasing the question, or ingest documentation that covers this topic."
)


def format_context(context: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered excerpts for prompts.

    The numbering here defines the citation space: ``[1]`` in the model's
    answer refers to the first excerpt in this rendering.
    """
    blocks = []
    for index, chunk in enumerate(context, start=1):
        location = chunk["source"]
        if chunk.get("page"):
            location += f", page {chunk['page']}"
        blocks.append(f"[{index}] ({location})\n{chunk['text']}")
    return "\n\n".join(blocks)
