"""
app.py — FastAPI server for the DevOps Incident environment.

From environments.md (Part 10):
    from core.env_server import create_fastapi_app
    env = YourEnvironment()
    app = create_fastapi_app(env)    ← generates /reset, /step, /state, /health, /ws, /web

That single call creates all required endpoints. We don't write routes manually.
"""

import os
import uvicorn
import gradio as gr

# --- OpenEnv server factory (confirmed in openenv.core) ---
from openenv.core import create_fastapi_app

from .environment import IncidentEnvironment
from .models import IncidentAction, IncidentObservation
from .gradio_ui import create_gradio_app

# create_fastapi_app signature (confirmed by inspection):
#   create_fastapi_app(env_factory, action_cls, observation_cls, ...)
#
# - env_factory: a callable () -> Environment  (we pass the class itself)
# - action_cls:  the Action subclass type
# - observation_cls: the Observation subclass type
#
# Creates all endpoints automatically:
#   POST /reset, POST /step, GET /state, GET /health, GET /ws, GET /web, GET /docs
app = create_fastapi_app(IncidentEnvironment, IncidentAction, IncidentObservation)

# Mount custom Gradio UI at /web (overrides the default OpenEnv web interface)
gradio_demo = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_demo, path="/web")


def main():
    """Entry point called by `uv run server` (defined in pyproject.toml)."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
