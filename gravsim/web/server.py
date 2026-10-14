"""FastAPI application: static ``web/`` at ``/`` and the scenario API under ``/api``.

Routes
------
* ``GET /api/health`` -> ``{"status": "ok", "version", "scenarios", "cache_entries"}``.
* ``GET /api/scenarios`` -> JSON list of ``scenarios.list_scenarios()`` entries, each with
  ``url = /api/scenario/<id>``.
* ``GET /api/scenario/{id}?<params>`` -> ``application/octet-stream`` payload
  (``payload.encode``) of ``scenarios.build_scenario``; query parameters are validated against
  the scenario's declared parameters (unknown id: 404; unknown/invalid parameter: 422).
  Results are cached in process, keyed by the scenario id and the validated parameters, so
  identical requests return identical bytes without recomputing (``X-Gravsim-Cache: hit|miss``).
* ``/`` -> ``web/index.html`` and the other static files of ``web_dir`` (mounted only if the
  directory exists; API routes are registered first and take precedence).

The endpoints are plain ``def`` functions, so FastAPI runs them in its thread pool and a
30-second scenario build does not block the event loop or the other routes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

from gravsim import __version__
from gravsim.web.payload import encode
from gravsim.web.scenarios import SCENARIOS, build_scenario, list_scenarios, validate_params

__all__ = ["WEB_DIR", "ScenarioCache", "create_app"]

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
"""The front end directory (``<repo>/web``) served at ``/``."""


class ScenarioCache:
    """In-process cache of encoded payloads keyed by scenario id and validated parameters.

    Builds of distinct keys run concurrently; a second request for a key already being built
    waits for it instead of building again.  The oldest entries are evicted beyond ``maxsize``.
    """

    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = int(maxsize)
        self._data: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._building: dict[tuple, threading.Lock] = {}

    @staticmethod
    def key(scenario_id: str, params: dict) -> tuple:
        return (scenario_id, tuple(sorted(params.items())))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get_or_build(self, scenario_id: str, params: dict) -> tuple[bytes, bool]:
        """Return ``(payload_bytes, hit)`` for validated ``params``."""
        key = self.key(scenario_id, params)
        with self._lock:
            data = self._data.get(key)
            if data is not None:
                self._data.move_to_end(key)
                return data, True
            build_lock = self._building.setdefault(key, threading.Lock())
        with build_lock:
            with self._lock:
                data = self._data.get(key)
            if data is not None:
                return data, True
            data = encode(build_scenario(scenario_id, **params))
            with self._lock:
                self._data[key] = data
                while len(self._data) > self.maxsize:
                    self._data.popitem(last=False)
                self._building.pop(key, None)
        return data, False


def create_app(web_dir: str | Path | None = WEB_DIR, cache: ScenarioCache | None = None) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    web_dir : path-like or None
        Directory served at ``/`` (``index.html`` at the root).  Skipped when it does not
        exist or is ``None``; the API still works.
    cache : ScenarioCache, optional
        Cache to use (a fresh one by default); available as ``app.state.cache``.
    """
    app = FastAPI(title="gravsim web", version=__version__, docs_url="/api/docs")
    cache = cache if cache is not None else ScenarioCache()
    app.state.cache = cache

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "scenarios": list(SCENARIOS),
            "cache_entries": len(cache),
        }

    @app.get("/api/scenarios")
    def scenarios() -> list[dict]:
        return [{**entry, "url": f"/api/scenario/{entry['id']}"} for entry in list_scenarios()]

    @app.get("/api/scenario/{scenario_id}")
    def scenario(scenario_id: str, request: Request) -> Response:
        if scenario_id not in SCENARIOS:
            raise HTTPException(status_code=404, detail=f"unknown scenario {scenario_id!r}")
        try:
            params = validate_params(scenario_id, dict(request.query_params))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        data, hit = cache.get_or_build(scenario_id, params)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "public, max-age=3600",
                "X-Gravsim-Cache": "hit" if hit else "miss",
            },
        )

    if web_dir is not None and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app
