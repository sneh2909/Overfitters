"""
FastAPI application for the House M.D. clinical-reasoning environment.

This wires the OpenEnv ``create_app`` factory to ``HouseMDEnvironment`` and
*also* mounts the rich ER scene UI (vendored under ``server/static/``) plus
its companion ``/api/...`` REST surface (see ``playground.py``).

Endpoints exposed
-----------------
OpenEnv contract (used by training jobs and the ``HouseMDEnv`` Python client):
    POST /reset         reset the environment
    POST /step          execute an action
    GET  /state         current introspection snapshot
    GET  /schema        JSON schemas for action / observation / state
    GET  /metadata      environment metadata
    GET  /health        healthcheck (used by Docker HEALTHCHECK)
    WS   /ws            per-session WebSocket
    POST /mcp           JSON-RPC MCP entrypoint
    GET  /docs          FastAPI Swagger UI

Playground surface (used by the static UI at /):
    GET  /              live ER scene (static/index.html)
    GET  /static/*      JS / CSS / assets for the scene
    GET  /api/catalogs                        action vocabularies
    POST /api/episodes                        reset() with optional disease/seed
    GET  /api/episodes/{sid}                  current observation
    POST /api/episodes/{sid}/actions          manual step
    POST /api/episodes/{sid}/agent_step       autopilot step (random|greedy|oracle)
    GET  /api/episodes/{sid}/rewards          terminal reward breakdown
    GET  /api/episodes/{sid}/truth            spectator-mode hidden state
    DELETE /api/episodes/{sid}                drop a session

Why we disable the Gradio auto-UI
---------------------------------
``create_app`` will mount a generic Gradio playground at ``/web`` and a
``/`` -> ``/web/`` redirect when ``ENABLE_WEB_INTERFACE=true``. That UI is
useful for environments that have no bespoke frontend, but House M.D.
ships a much richer animated ER scene we want to be the landing page —
so we hard-disable the Gradio surface here and own ``/`` ourselves. The
canonical OpenEnv API endpoints (``/reset``, ``/step``, etc.) are
unaffected by this and continue to work for both training jobs and the
Python client.

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

# Disable the Gradio auto-UI before importing create_app so we control `/`.
# create_app reads this env var lazily inside its body, so setting it here
# is enough as long as it happens before the call to create_app() below.
os.environ["ENABLE_WEB_INTERFACE"] = "false"

from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from openenv.core.env_server.http_server import create_app  # noqa: E402

try:
    from ..models import HouseMDAction, HouseMDObservation
    from .house_md_environment import HouseMDEnvironment
    from .playground import register_playground
except ImportError:
    from models import HouseMDAction, HouseMDObservation
    from server.house_md_environment import HouseMDEnvironment
    from server.playground import register_playground


# ---------------------------------------------------------------------------
# Build the OpenEnv FastAPI app (REST + WS + MCP + /docs).
# We pass the *class* (factory) rather than an instance so the server can
# allocate one HouseMDEnvironment per WebSocket session.
# ---------------------------------------------------------------------------

app = create_app(
    HouseMDEnvironment,
    HouseMDAction,
    HouseMDObservation,
    env_name="house_md_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


# ---------------------------------------------------------------------------
# Mount the rich playground UI on top of the OpenEnv surface.
# ---------------------------------------------------------------------------
# Layout on disk:
#   server/
#     app.py            <- you are here
#     playground.py     <- /api/... legacy session API
#     static/
#       index.html      <- ER scene
#       app.js          <- frontend state machine
#       styles.css      <- hospital pixel-art styling

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

if _STATIC_DIR.is_dir():
    app.mount(
        "/static", StaticFiles(directory=str(_STATIC_DIR)), name="static"
    )

# Register the legacy /api/* routes that the static UI calls.
register_playground(app, prefix="/api")


# Plain-text fallback if static assets aren't shipped (eg. someone runs the
# package without the bundled UI). Also keeps `openenv validate` happy by
# guaranteeing `/` always returns 200.
_FALLBACK_LANDING = """<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>House M.D. — OpenEnv</title></head>
<body style="font-family: system-ui; max-width: 640px; margin: 4rem auto;">
  <h1>House M.D. — OpenEnv RL environment</h1>
  <p>The animated ER scene UI is not bundled in this build. The OpenEnv
     endpoints are still live:</p>
  <ul>
    <li><a href="/docs">/docs</a> — FastAPI Swagger</li>
    <li><a href="/schema">/schema</a> — Action + observation JSON schemas</li>
    <li><a href="/health">/health</a> — health check</li>
  </ul>
</body></html>
"""


@app.get("/", include_in_schema=False)
async def landing():
    if _INDEX_HTML.is_file():
        return FileResponse(str(_INDEX_HTML))
    return HTMLResponse(_FALLBACK_LANDING)


# The HF Space iframe + health probes sometimes hit `/web` and `/web/`
# because `openenv push` historically wired a Gradio UI there. We disabled
# Gradio in favor of the richer ER scene at `/`, so silently redirect those
# probes to the new home rather than spamming 404s in the logs.
@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
async def web_redirect():
    return RedirectResponse(url="/", status_code=307)


# ---------------------------------------------------------------------------
# `python -m server.app` and `uv run server` entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # `openenv validate` checks for the literal substring `main()` here.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.host == "0.0.0.0" and args.port == 8000:
        main()
    else:
        main(host=args.host, port=args.port)
