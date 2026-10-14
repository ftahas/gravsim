"""Web front end back end (Phase 8): scenarios, binary payloads and the FastAPI server.

* :mod:`gravsim.web.scenarios` -- pure scenario builders (``list_scenarios``, ``build_scenario``).
* :mod:`gravsim.web.payload` -- ``encode``/``decode`` of the binary payload the browser loads.
* :mod:`gravsim.web.server` -- ``create_app()`` (FastAPI) serving ``web/`` and ``/api/...``.
* ``gravsim-web [--port 8000]`` (or ``python -m gravsim.web``) runs the server with uvicorn.

FastAPI is imported only by ``server``; this package and the two pure modules need numpy only.
"""

from gravsim.web.payload import decode, encode
from gravsim.web.scenarios import build_scenario, list_scenarios, validate_params

__all__ = ["build_scenario", "list_scenarios", "validate_params", "encode", "decode"]
