FROM python:3.13-alpine

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY main.py .

CMD ["uv", "run", "main.py"]