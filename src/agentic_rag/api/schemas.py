"""Pydantic request/response models — the API's public contract.

These models are the single place where the wire format is defined: FastAPI
validates every request against them (rejecting bad input with a 422 before
any handler code runs) and filters every response through them, so internal
state never leaks into the API by accident.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's question or message.",
        examples=["What is the difference between PUT and PATCH?"],
    )
    session_id: str = Field(
        min_length=1,
        max_length=128,
        description="Conversation identifier; reuse it to continue the same conversation.",
        examples=["user-42"],
    )


class Source(BaseModel):
    """A document excerpt cited by the answer."""

    index: int = Field(description="Citation marker used in the answer, e.g. 1 for [1].")
    chunk_id: str
    source: str = Field(description="File the excerpt comes from.")
    page: int | None = Field(default=None, description="PDF page number, if applicable.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(
        description="Excerpts backing the answer; empty for conversational replies."
    )
    session_id: str


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
