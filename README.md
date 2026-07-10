# agentic-rag-assistant

A conversational RAG agent built with **LangGraph**: it routes each message through an
explicit graph that decides whether to retrieve documentary context, performs **hybrid
retrieval** (semantic embeddings + BM25, fused with reciprocal rank fusion) over a local
**ChromaDB** index, generates answers **with source citations**, and gates every answer
behind a **groundedness check** — if the retrieved documents don't support the answer, the
agent says so instead of hallucinating. Exposed as a REST API with **FastAPI**, with
per-session conversation memory, containerized, tested (49 tests, no API key needed), and
wired to CI.

## Architecture

```mermaid
graph TD
    START([user message]) --> CI[classify_intent<br/><i>structured output: retrieval | chat</i>]
    CI -->|retrieval| R[retrieve<br/><i>BM25 + embeddings, RRF fusion</i>]
    CI -->|chat| CH[chat_respond]
    CH -->|tool calls| T[tools<br/><i>calculator · current_date</i>]
    T --> CH
    CH -->|plain answer| E([END])
    R -->|chunks found| G[generate<br/><i>draft answer with citation markers</i>]
    R -->|no chunks| REF[refuse<br/><i>honest &quot;not enough information&quot;</i>]
    G --> V[verify<br/><i>groundedness judge</i>]
    V -->|grounded| F[finalize<br/><i>parse citations into sources</i>]
    V -->|ungrounded| REF
    F --> E
    REF --> E
```

The defining property of the graph: **unverified text never enters the conversation
history**. `generate` writes a *draft* into graph state; only after the groundedness judge
approves it does `finalize` append it as a message. Since the checkpointer persists
message history between turns, this guarantees the agent's memory only ever contains
verified answers or honest refusals — a hallucination that slipped into history would
otherwise contaminate every future turn.

### Components

| Layer | Module | What it does |
|---|---|---|
| Ingestion | `agentic_rag.ingestion` | Loads PDFs (per page) and Markdown, splits into token-sized chunks with structural boundaries, embeds and indexes into ChromaDB with `source`/`page` metadata and deterministic chunk IDs |
| Retrieval | `agentic_rag.retrieval` | Semantic search (OpenAI embeddings via Chroma) + lexical search (BM25 derived in-memory from the same collection), fused with reciprocal rank fusion |
| Agent | `agentic_rag.agent` | The LangGraph graph above, plus prompts, safe AST-based calculator tool, and citation parsing |
| API | `agentic_rag.api` | FastAPI app: `/chat`, `/ingest`, `/health`, Pydantic contracts, agent built once at startup |

## Key design decisions

- **Hybrid retrieval over pure-vector** — embeddings handle paraphrase but blur exact
  identifiers (`429`, `RFC 7807`, command names), which is exactly where BM25 excels; real
  queries mix both. RRF fuses the rankings using ranks only, avoiding the impossible task
  of normalizing cosine similarities against unbounded BM25 scores.
- **Chunking: structural cuts, token-based budget** — the recursive splitter cuts on the
  strongest boundary available (paragraph → line → word) while measuring size in tiktoken
  tokens, because every hard budget in the system (embedding input, context window,
  billing) is denominated in tokens, not characters.
- **Groundedness as a gate, not a prompt** — a separate LLM-as-judge call receives only
  the chunks and the draft, and returns a structured boolean. Citation markers are
  explicitly excluded as evidence.
- **Memory = LangGraph checkpointer** — `thread_id` maps to the API's `session_id`;
  state is persisted in SQLite so conversations survive restarts. No external services.
- **Dependency injection everywhere** — the graph takes `(llm, retriever, checkpointer)`,
  the app takes an `agent_factory`, the retriever takes a `VectorSearcher` protocol. The
  entire test suite (49 tests), including full-graph and API integration tests, runs
  without an API key or a vector index.
- **BM25 derived from Chroma** — the lexical index is rebuilt in memory from the Chroma
  collection, keeping a single source of truth and making index-drift bugs impossible.
  `/ingest` rebuilds the agent afterwards so the fresh corpus is immediately searchable.

## Getting started

Requirements: Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/<you>/agentic-rag-assistant && cd agentic-rag-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                 # then fill in OPENAI_API_KEY

python scripts/ingest.py                             # index the sample corpus in data/
uvicorn agentic_rag.api.app:app --reload             # http://127.0.0.1:8000/docs
```

### With Docker

```bash
cp .env.example .env    # fill in OPENAI_API_KEY
docker compose up --build
curl -X POST http://localhost:8000/ingest            # index the corpus (first run only)
```

The Chroma index and conversation memory live in a named volume (`state`), so they
survive container restarts. `data/` is mounted read-only; drop new PDFs or Markdown files
there and call `/ingest` to reindex.

## API

Interactive docs at `http://localhost:8000/docs`.

**`POST /chat`** — ask a question; reuse `session_id` to continue a conversation.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the trade-offs between offset and cursor pagination?", "session_id": "demo"}'
```

```json
{
  "answer": "Offset pagination is simple and supports jumping to arbitrary pages, but degrades on large tables [1]. Cursor pagination returns an opaque token, is stable under concurrent writes and fast at any depth, but gives up random page access [1].",
  "sources": [
    {"index": 1, "chunk_id": "rest_api_design.pdf:p2:c0", "source": "rest_api_design.pdf", "page": 2}
  ],
  "session_id": "demo"
}
```

A question the corpus can't answer produces an honest refusal with empty `sources`
instead of a fabricated answer. Conversational messages (greetings, arithmetic, dates)
skip retrieval entirely; arithmetic and date questions are answered through **function
calling** (a sandboxed AST calculator and a `current_date` tool).

**`POST /ingest`** — reload `data/`, rebuild the index, swap in a fresh agent.

**`GET /health`** — liveness probe with version.

## Evaluation

A 12-question annotated dataset (`evaluation/dataset.json`) measures two stage-separable
metrics, so failures are diagnosable:

- **Retrieval hit rate** — is the expected source among the retrieved chunks?
  (No LLM calls; failures point at chunking/search.)
- **Citation accuracy** — does the final answer cite the expected source and page?
  (Full agent; failures with good retrieval point at generation/judging.)

```bash
python scripts/evaluate.py
```

## Testing and CI

```bash
pytest        # 49 tests, no API key required
ruff check .
```

The suite covers unit level (chunking, RRF fusion, BM25, citation parsing, tools,
config), full-graph level (every path through the LangGraph graph with a scriptable fake
LLM, including tool-call execution and per-thread memory), and API level (HTTP contracts
with a fake agent, plus integration tests through the real graph — including a proof that
conversation memory survives across separate HTTP requests). GitHub Actions runs lint +
tests on every push and pull request.

## Project structure

```
src/agentic_rag/
├── config.py        # typed settings (pydantic-settings), overridable via env vars
├── ingestion/       # loaders (pypdf), token-aware chunking, Chroma indexing
├── retrieval/       # BM25 index, RRF fusion, hybrid retriever
├── agent/           # LangGraph graph, state, prompts, tools, citation parsing
└── api/             # FastAPI app + Pydantic schemas
data/                # source documents (sample corpus included)
evaluation/          # annotated Q&A dataset
scripts/             # ingest.py, evaluate.py
tests/               # pytest suite (49 tests, API-key-free)
```

## Limitations and future work

Deliberate simplifications, each with its production-grade evolution:

- **Full index rebuild on ingest** — fine for a small corpus; at scale, switch to
  incremental sync (content hashes, upserts, orphan deletion) and move ingestion to a
  background job with progress reporting.
- **Unbounded conversation history** — each turn replays the full history to the LLM;
  add message trimming or progressive summarization for long sessions.
- **Self-judging groundedness** — the judge is the same model that generated the draft
  and can share its biases; harden with a different judge model, a fine-tuned NLI model,
  or claim-by-claim verification.
- **`session_id` is identifier and credential at once** — production needs real
  authentication, with `thread_id` derived server-side from the authenticated identity.
- **Sync LLM calls in a threadpool** — honest and simple; an async end-to-end stack
  (`ainvoke`, async clients) would scale to higher concurrency.
- **Small eval set (n=12)** — a regression smoke test, not a benchmark; grow it with
  semi-automated question generation and add negative cases (questions the corpus cannot
  answer, where refusing is the correct behavior).
- **Single-worker deployment** — BM25 lives in process memory and SQLite is a local
  file; scaling horizontally means moving lexical search to a shared engine and
  checkpoints to Postgres (LangGraph supports both).

## License

[MIT](LICENSE)
