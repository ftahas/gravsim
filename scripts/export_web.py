"""Export every web scenario to static files so ``web/`` works without the API server.

Writes ``<out>/manifest.json``, a bare JSON list with the same shape as the ``/api/scenarios``
response (one entry per scenario, ``url = <out-name>/<id>.bin``, i.e. ``data/<id>.bin`` for the
default ``web/data``, plus ``built_with`` = the parameters used), and one ``<out>/<id>.bin``
payload per scenario built with its default parameters (``--set id.param=value`` overrides one
value; ``--only id ...`` restricts the set).  The front end (``web/app.js``) decodes the manifest
and the API response through one code path, so the manifest must stay a list, never a wrapping
object.  Afterwards ``python -m http.server -d web`` serves a working front end.  Prints only a
one-line summary per file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from gravsim.web.payload import encode
from gravsim.web.scenarios import build_scenario, default_params, list_scenarios, validate_params

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "web" / "data"


def export(
    out_dir: Path | str = DEFAULT_OUT,
    only: Sequence[str] | None = None,
    overrides: dict[str, dict] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """Build the scenarios and write ``manifest.json`` plus the ``.bin`` files into ``out_dir``.

    Parameters
    ----------
    out_dir : path-like
        Output directory (created).  The manifest ``url`` of each entry is
        ``<out_dir.name>/<id>.bin``, relative to the parent of ``out_dir`` (the web root).
    only : sequence of str, optional
        Scenario ids to export (all by default).
    overrides : dict, optional
        ``{scenario_id: {param: value}}`` applied on top of the defaults.
    log : callable, optional
        Receives one summary line per file.

    Returns
    -------
    list of dict
        The manifest written: the ``list_scenarios()`` entries in registry order, each with
        ``url`` (``<out_dir.name>/<id>.bin``) and ``built_with`` (the validated parameters).
        This is exactly the shape of ``GET /api/scenarios``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    entries = []
    for entry in list_scenarios():
        sid = entry["id"]
        if only and sid not in only:
            continue
        params = validate_params(sid, {**default_params(sid), **overrides.get(sid, {})})
        tic = time.perf_counter()
        data = encode(build_scenario(sid, **params))
        path = out / f"{sid}.bin"
        path.write_bytes(data)
        entries.append({**entry, "url": f"{out.name}/{sid}.bin", "built_with": params})
        if log is not None:
            log(f"{sid}: {path} ({len(data) / 1e6:.2f} MB, {time.perf_counter() - tic:.1f} s)")
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")
    if log is not None:
        log(f"manifest: {manifest_path} ({len(entries)} scenarios)")
    return entries


def _parse_set(items: Sequence[str]) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    for item in items:
        try:
            target, value = item.split("=", 1)
            sid, name = target.split(".", 1)
        except ValueError:
            raise SystemExit(f"--set expects id.param=value, got {item!r}") from None
        overrides.setdefault(sid, {})[name] = value
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help=f"output directory (default {DEFAULT_OUT})"
    )
    parser.add_argument("--only", nargs="*", default=None, help="scenario ids to export")
    parser.add_argument(
        "--set", action="append", default=[], metavar="ID.PARAM=VALUE", help="override a parameter"
    )
    args = parser.parse_args(argv)
    export(args.out, only=args.only, overrides=_parse_set(args.set), log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
