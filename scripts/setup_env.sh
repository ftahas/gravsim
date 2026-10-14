#!/usr/bin/env bash
# One-shot environment setup: uv venv + editable install + REBOUNDx rpath fix.
# REBOUNDx built from source links librebound by bare soname and ships an rpath pointing at
# uv's temporary build dir, so the loader cannot find librebound; point the rpath at $ORIGIN.
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv -q .venv
uv pip install -q -e ".[dev]"
uv pip install -q patchelf
SP=.venv/lib/python3.12/site-packages
if ls "$SP"/libreboundx.*.so >/dev/null 2>&1; then
  .venv/bin/patchelf --set-rpath '$ORIGIN' "$SP"/libreboundx.*.so
fi
.venv/bin/python -c "import rebound, reboundx; print('rebound', rebound.__version__, 'reboundx', reboundx.__version__, 'ok')"
