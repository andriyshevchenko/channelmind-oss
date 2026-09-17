"""Combined ASGI server: the agent-facing MCP endpoint plus the admin plane.

- `/mcp`   — streamable-HTTP MCP server, Bearer-token auth (agent-facing).
- `/admin` — corpus provisioning API, admin-key auth (dashboard-facing).

The MCP app carries its own lifespan that runs the streamable-HTTP session
manager, so mounting the admin app onto it is all that's needed.
"""
from __future__ import annotations

from starlette.routing import Mount

from .admin import build_admin_app
from .config import Config, load_config
from .corpora import CorpusStore
from .mcp_server import build_mcp


def build_server(cfg: Config | None = None):
    cfg = cfg or load_config()
    cstore = CorpusStore(cfg.data_dir)
    mcp = build_mcp(cfg, cstore)
    app = mcp.streamable_http_app()  # Starlette app serving MCP at /mcp
    app.routes.append(Mount("/admin", app=build_admin_app(cfg, cstore)))
    return app
