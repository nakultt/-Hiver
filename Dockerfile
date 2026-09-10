# Pinned to a slim base: this image has no system dependencies beyond libc --
# ReplyBench is pure Python talking to one HTTP endpoint.
FROM python:3.11-slim

# uv gives reproducible, fast installs and is the same tool the README uses
# outside Docker, so the two paths cannot drift.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install cache.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache -e ".[dev]"

COPY tests ./tests
COPY data ./data

# The LLM response cache lives here. Mount a volume over it to make runs
# resumable across container restarts:
#   docker run -v "$PWD/.cache:/app/.cache" ...
ENV LLM_CACHE=1
VOLUME ["/app/.cache", "/app/runs"]

# Default to the offline backend so `docker run` does something useful and
# non-failing with no key. Override with -e LLM_BACKEND=api and -e LLM_API_KEY.
ENV LLM_BACKEND=mock

ENTRYPOINT ["replybench"]
CMD ["--help"]
