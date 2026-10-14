# Web front end (Three.js)

A browser viewer for five precomputed scenarios: the solar system under Newton and GR with the
GR − Newton displacement amplified until Mercury's perihelion advance is visible, the $c/100$
rosette, an exact Schwarzschild zoom-whirl geodesic, a bundle of light rays past a compact mass
or the Sun, and the Earth–Moon system among the planets. **All physics stays in Python**
(`gravsim/web/scenarios.py`); the browser only decodes the arrays and renders them with
[three.js](https://threejs.org/) `0.160.0`, pinned from cdn.jsdelivr.net through an import map
in `web/index.html` — no build step, no npm. The page needs WebGL.

## Running it

With the API server (scenarios are built on demand, the first `solar_system` takes ~30 s,
then they are cached in memory):

```bash
uv pip install -e ".[web]"                       # fastapi + uvicorn
.venv/bin/gravsim-web --port 8000      # then open http://127.0.0.1:8000/
```

> **Run it from any folder.** Use the `gravsim-web` command (`.venv/bin/gravsim-web`, or
> `/home/<you>/gravsim/.venv/bin/gravsim-web` from elsewhere). `python -m gravsim.web` fails
> with "No module named gravsim.web.__main__" when started from the project's parent folder:
> `-m` puts the current folder first on the import path, and the project directory, which has
> no `__init__.py`, then shadows the package as a namespace package named `gravsim`.


From any static server, after exporting the default scenarios once (a few MB, git-ignored):

```bash
.venv/bin/python scripts/export_web.py --out web/data   # manifest.json + <id>.bin, ~30 s
python -m http.server -d web 8000                       # then open http://127.0.0.1:8000/
```

`scripts/export_web.py --only solar_system --set solar_system.years=10` exports a subset with
other parameters. The front end looks for `data/manifest.json` first and falls back to
`GET /api/scenarios`; both are the same JSON list and every scenario goes through one decoder.

## Scenarios

| id | what is computed | parameters (defaults) |
|---|---|---|
| `solar_system` | Sun + 8 planets from the J2000 snapshot, Newton and EIH on one DOP853 grid; Mercury's perihelion indices in both runs; fitted GR − Newton rates (42.98″/cy Mercury, 8.6 Venus, 3.85 EMB, 1.35 Mars) and the Sun's wobble | `years` 100, `step_days` 2 |
| `rosette` | Sun + Mercury with $c$ scaled down: the 1PN advance grows as $1/c^2$ (0.287°/orbit at $c/100$, fitted 0.2870) | `c_scale` 0.01, `n_orbits` 50 |
| `black_hole` | exact Schwarzschild geodesic (`metric.schwarzschild.geodesic_orbit`) from apoapsis $r_0$ with the periapsis just outside the separatrix; coordinates in $GM/c^2$, $E$, $L$, turns per radial period | `r0` 20, `peri` 4.6, `n_orbits` 6 |
| `light_rays` | null geodesics (`metric.null`) past a black hole (impact parameters from $1.05\,b_{\rm crit}$ to $12\,GM/c^2$) or past the Sun with the 1.75″ limb deflection exaggerated | `n_rays` 9, `compact` true |
| `earth_moon` | Sun + planets + Earth + Moon, Newton vs EIH, with the sidereal and anomalistic months and the perigee (+40.7°/yr) and node (−19.3°/yr) rates fitted from the run | `years` 2, `step_days` 0.1 |

Controls: play/step along the samples, a time slider with the calendar date, the model
(Newton, GR, both), the **amplification** $k$ of the GR − Newton displacement
($\mathbf r_N + k\,\Delta$, $\Delta$ from the float64 `delta_gr` array), the **perihelion
trail** (markers at the perihelia of the *displayed* Mercury orbit, so the advance becomes an
arc), the frame (barycentric, heliocentric, Earth-centred), trail length, labels, an ecliptic
grid and camera presets. The right panel lists every number in `meta` (fitted rates, analytic
predictions, $E$, $L$, deflection angles) and the array index. `web/README.md` documents the
controls, the `window.__gravsim` test hook and the binary format in full.

## Data format

`GET /api/scenario/<id>?<params>` (or `data/<id>.bin`) is one payload written by
`gravsim.web.payload.encode`: a 4-byte little-endian header length, a UTF-8 JSON header
(`meta` plus an `index` of `{name, dtype, shape, offset, nbytes}`), then the concatenated
little-endian `float32`/`int32` arrays. `decode` in Python and the loader in `web/app.js`
mirror each other; `tests/test_web_backend.py` round-trips every scenario.

## Tests

`tests/test_web_backend.py` covers the pure modules (scenario registry and physics, payload
encoding, the FastAPI app through `httpx`, the export script) and runs in the fast suite;
`tests/test_web.py` (marker `web`, skipped without playwright + chromium) starts
`python -m gravsim.web` on a free port, loads the page headlessly, requires `__gravsim.ready`
with zero console errors, loads every scenario, exercises `setTime` and `setAmplification`,
checks the static export served by `http.server`, and saves `figures/web_<id>.png`:

```bash
.venv/bin/python -m playwright install chromium   # once
.venv/bin/python -m pytest -q -m web tests/test_web.py
```

The screenshots they save (`figures/web_<id>.png`) are indexed in the README's figure table.
