# ---- Build stage: install the package and its dependencies ----
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage: minimal image, non-root user ----
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY scripts/ scripts/
COPY evaluation/ evaluation/

# Bake the tiktoken encoding into the image so chunking works without
# network access at runtime.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# /app/state holds the Chroma index and the conversation checkpoints; it is
# mounted as a volume by docker-compose so both survive container restarts.
RUN useradd --create-home appuser \
    && mkdir -p /app/state \
    && chown appuser /app/state
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "agentic_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
