#!/usr/bin/env bash
# One-shot environment setup (also used by CI): uv venv + editable install + REBOUNDx rpath fix.
#
#   scripts/setup_env.sh                 # all extras: dev, app, docs, web
#   GRAVSIM_EXTRAS=dev scripts/setup_env.sh
#
# REBOUNDx has no prebuilt wheel for every Python, so it is often built from source; that build
# links librebound by bare soname and records an rpath pointing at uv's temporary build
# directory, so the loader cannot find librebound afterwards. Point the rpath at $ORIGIN (the
# site-packages directory where both libraries live).
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRAS="${GRAVSIM_EXTRAS:-dev,app,docs,web}"
uv venv -q --allow-existing .venv
uv pip install -q --python .venv/bin/python -e ".[${EXTRAS}]" patchelf
for so in .venv/lib/python3*/site-packages/libreboundx*.so; do
  [ -e "$so" ] && .venv/bin/patchelf --set-rpath '$ORIGIN' "$so"
done
if .venv/bin/python -c "import reboundx" 2>/dev/null; then
  .venv/bin/python -c "import rebound, reboundx; print('rebound', rebound.__version__, 'reboundx', reboundx.__version__, 'ok')"
else
  echo "rebound/reboundx not installed (only in the dev extra); oracle tests will be skipped"
fi
