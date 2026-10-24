# co-extract web UI. API keys are never baked in: pass them at run time with --env-file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY src/co_extract/__init__.py src/co_extract/__init__.py
RUN pip install -e .

# The app, plus the recorded fixtures the Demo provider replays.
COPY src ./src
COPY tests/fixtures ./tests/fixtures

# data/ holds the history database; deploy.yml mounts a host folder here so it outlives the container.
RUN mkdir -p /app/data && useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/samples', timeout=4)"

CMD ["co-extract", "serve", "--host", "0.0.0.0", "--port", "8000"]
