"""Where the local Showdown server is.

``ARENA_SHOWDOWN_URL`` (default ``http://localhost:8000``, poke-env's own default) moves it, e.g.
``ARENA_SHOWDOWN_URL=http://localhost:8100`` on a machine where 8000 is taken. Accepted forms:
``host:port``, ``http(s)://host:port`` or the full ``ws(s)://host:port/showdown/websocket``.
Players, external bots and the watch / browser links all read it from here.
"""

from __future__ import annotations

import os

DEFAULT_URL = "http://localhost:8000"
_WS_PATH = "/showdown/websocket"


def http_url() -> str:
    url = (os.environ.get("ARENA_SHOWDOWN_URL") or "").strip() or DEFAULT_URL
    if url.endswith(_WS_PATH):
        url = url[:-len(_WS_PATH)]
    url = url.rstrip("/")
    if url.startswith("ws://"):
        return "http://" + url[5:]
    if url.startswith("wss://"):
        return "https://" + url[6:]
    return url if "://" in url else "http://" + url


def websocket_url() -> str:
    url = http_url()
    scheme = "wss://" if url.startswith("https://") else "ws://"
    return scheme + url.split("://", 1)[1] + _WS_PATH


def server_configuration():
    """poke-env ServerConfiguration: our websocket, Smogon's login endpoint (unused with --no-security)."""
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration, ServerConfiguration
    return ServerConfiguration(websocket_url(), LocalhostServerConfiguration.authentication_url)
