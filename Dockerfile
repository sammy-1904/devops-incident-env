# Dockerfile — MUST be at the project root (not inside server/).
# Pulkit explicitly said this in the live session.
#
# Build:  docker build -t devops-incident-env .
# Run:    docker run -d -p 8000:8000 devops-incident-env

FROM python:3.11-slim

WORKDIR /app

# Install UV (fast Python package manager used by OpenEnv)
RUN pip install --no-cache-dir uv

# Copy dependency files first (Docker layer cache)
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies
RUN uv sync --frozen --no-dev 2>/dev/null || pip install openenv-core openai fastapi uvicorn

# Copy source code
COPY server/ ./server/
COPY data/ ./data/
COPY client.py .
COPY openenv.yaml .

# Enable the web interface (Gradio UI at /web)
ENV ENABLE_WEB_INTERFACE=true
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

EXPOSE 8000

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
