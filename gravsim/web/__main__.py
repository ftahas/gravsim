"""``python -m gravsim.web [--host 127.0.0.1] [--port 8000] [--web-dir web]`` runs the server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gravsim.web.server import WEB_DIR, create_app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m gravsim.web",
        description="Serve the gravsim web front end and its scenario API with uvicorn.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default 8000)")
    parser.add_argument(
        "--web-dir", default=str(WEB_DIR), help=f"static front end directory (default {WEB_DIR})"
    )
    args = parser.parse_args(argv)

    import uvicorn  # imported here so the pure modules never need it

    uvicorn.run(create_app(web_dir=args.web_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
