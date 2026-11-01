"""HTTP/WebSocket hardening for the local Clinura server.

Clinura listens on 127.0.0.1, but "local" is not "safe": any web page open in your
browser can send requests to localhost. The defences here:

* **DNS-rebinding protection** – only ``Host: localhost/127.0.0.1/[::1]`` (or the
  configured host) is served;
* **Cross-origin protection** – state-changing requests and the WebSocket (live video and
  vitals) are refused when they come from another website's ``Origin``;
* **Access token** – required whenever the server listens on a non-loopback address;
* **Security headers** – CSP, no framing, no MIME sniffing, no referrer, no caching of API
  responses, camera/geolocation off for the page.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
TOKEN_COOKIE = "clinura_token"

CSP = (
    "default-src 'self'; "
    "img-src 'self' data: blob:; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; "
    "connect-src 'self' ws: wss:; "
    "script-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def is_loopback(host: str) -> bool:
    return host in LOOPBACK


def allowed_hosts(host: str) -> set[str]:
    hosts = set(LOOPBACK)
    if host not in ("0.0.0.0", "::"):  # nosec B104 - comparison, not a bind
        hosts.add(host)
    return hosts


def allowed_origins(host: str, port: int) -> set[str]:
    names = {"localhost", "127.0.0.1", "[::1]"}
    if host not in ("0.0.0.0", "::"):  # nosec B104 - comparison, not a bind
        names.add(host)
    origins = set()
    for name in names:
        for p in {port, 5173, 4173}:  # app, Vite dev server, Vite preview
            origins.add(f"http://{name}:{p}")
            origins.add(f"https://{name}:{p}")
    return origins


def _host_only(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _token_ok(
    expected: str | None,
    headers: Mapping[str, str],
    cookies: Mapping[str, str],
    query: Mapping[str, str],
) -> bool:
    if not expected:
        return True
    auth = headers.get("authorization", "")
    candidates = [
        auth[7:] if auth.lower().startswith("bearer ") else "",
        cookies.get(TOKEN_COOKIE, ""),
        query.get("token", ""),
    ]
    return any(c and secrets.compare_digest(c, expected) for c in candidates)


def websocket_allowed(headers, cookies, query, origins: set[str], config) -> bool:
    host = _host_only(headers.get("host", ""))
    if host not in allowed_hosts(config.host):
        return False
    origin = headers.get("origin")
    if origin is not None and origin not in origins:
        return False
    return _token_ok(config.access_token, headers, cookies, query)


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app, allowed_hosts: set[str], allowed_origins: set[str], token: str | None
    ) -> None:
        super().__init__(app)
        self.hosts = allowed_hosts
        self.origins = allowed_origins
        self.token = token

    async def dispatch(self, request: Request, call_next) -> Response:
        host = _host_only(request.headers.get("host", ""))
        if host not in self.hosts:
            return JSONResponse({"detail": "Host not allowed."}, status_code=400)
        origin = request.headers.get("origin")
        if request.method not in SAFE_METHODS:
            if origin is not None and origin not in self.origins:
                return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
            if origin is None and request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
            if (
                request.url.path.startswith("/api/")
                and not request.headers.get("content-type", "").startswith("application/json")
                and int(request.headers.get("content-length") or 0) > 0
            ):
                return JSONResponse({"detail": "JSON body required."}, status_code=415)
        set_cookie = False
        if self.token:
            if not _token_ok(self.token, request.headers, request.cookies, request.query_params):
                return JSONResponse({"detail": "Access token required."}, status_code=401)
            set_cookie = request.query_params.get("token") is not None
        response = await call_next(request)
        headers = response.headers
        headers["Content-Security-Policy"] = CSP
        headers["X-Frame-Options"] = "DENY"
        headers["X-Content-Type-Options"] = "nosniff"
        headers["Referrer-Policy"] = "no-referrer"
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"
        headers["Permissions-Policy"] = "camera=(), geolocation=(), payment=(), microphone=(self)"
        if request.url.path.startswith("/api/"):
            headers["Cache-Control"] = "no-store"
        if set_cookie:
            response.set_cookie(
                TOKEN_COOKIE,
                self.token,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
            )
        return response
