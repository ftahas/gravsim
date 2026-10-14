# gravsim web front end

A Three.js viewer for the gravsim scenarios: the solar system under Newton and GR (with the
GR − Newton displacement amplified until Mercury's perihelion advance is visible), the c/100
rosette, an exact Schwarzschild geodesic, a bundle of light rays past a compact mass, and the
Earth–Moon system. All physics is computed in Python (`gravsim.web.scenarios`); the browser
only decodes the precomputed arrays and renders them.

No build step and no npm: `index.html` pins `three@0.160.0` from cdn.jsdelivr.net through an
import map, `app.js` is a plain ES module and `style.css` the dark theme (body colours from
`gravsim/viz/plots.py`). The page needs WebGL; without it a message is shown instead of a
blank canvas.

## Running

With the FastAPI server (scenarios are built on demand and cached in memory):

```bash
.venv/bin/python -m gravsim.web --port 8000      # then open http://127.0.0.1:8000/
```

The page tries both `data/manifest.json` (static export) and `GET /api/scenarios` and loads
every scenario from the `url` of its entry (`data/<id>.bin` or `/api/scenario/<id>`). A 404
on the first probe would be logged by the browser as a console error, so the page looks at
the `server` header of its own stylesheet (served from the browser cache): behind uvicorn it
asks the API first, behind any other server the manifest first, falling back to the other
either way.

From any static server, after exporting the default scenarios:

```bash
.venv/bin/python scripts/export_web.py --out web/data   # writes manifest.json + <id>.bin
python -m http.server -d web 8000                       # then open http://127.0.0.1:8000/
```

`web/data/` is git-ignored (`web/.gitignore`); the exported files are a few MB.

## Controls

Left panel (collapses behind the "Controls" button on screens narrower than 900 px; the info
panel likewise behind "Info"):

| Control | Effect |
|---|---|
| Scenario | pick one of the manifest entries |
| Play / ◀ ▶ / ⏮, Speed | animate along the samples with linear interpolation; speed is a log slider around "one full run per minute"; space bar toggles play |
| Time | slider over the series; the readout gives `year +12.3 (2012-04-13)` from `t` in days since J2000.0 (2000-01-01T12:00), proper time for the geodesic, path fraction for the rays |
| Model | Newton, GR, or both (Newton drawn as dim paths and wireframe ghosts) |
| Amplification k | shown for `solar_system` and `earth_moon` only (in the c/100 rosette GR − Newton is already millions of km, so the slider is hidden and `setAmplification` is a no-op there); log slider 1 … 10⁵; displayed position = Newton + k · (GR − Newton), the difference taken from `delta_gr` (computed in float64 before the float32 cast) when the payload has it; readout of Mercury's true and displayed GR − Newton displacement in km |
| Perihelion trail | shown only when the payload has `peri_idx_*` arrays (on by default in the rosette); markers at the perihelia of the *displayed* Mercury orbit up to the current time, so the advance becomes an arc. Each marker is the local minimum of the displayed Sun–Mercury distance nearest to a `peri_idx_gr` sample, refined between samples by the parabola through the three distances around it (as `analysis.precession.perihelion_passages`) with the position interpolated at that time; `peri_idx_newton` seeds the grey markers on the Newton path in the same way. The GR samples themselves are *not* the perihelia of Newton + k · (GR − Newton) once k ≠ 1: the GR passages lag Newton's, so at a GR perihelion the difference points backwards along the orbit and k × that would drag the marker behind Newton's perihelion (a retrograde arc); the displayed perihelia advance k × 43″ per century ahead of the Newtonian ones |
| Frame | barycentric, heliocentric (minus the Sun), or Earth-centred (earth_moon) |
| Trail | length of the bright trail behind each body (log slider, days); "Full paths" toggles the dim complete orbits |
| Labels, Ecliptic grid | CSS2D body labels; rings at 1-2-5 radii in the ecliptic plane |
| Camera | presets inner / outer / top / edge, plus "earth" (earth_moon: Earth-centred frame, close enough to see the Moon's orbit); drag to orbit, wheel to zoom, right-drag to pan |

The right panel shows the scenario blurb, a short explanation, every number in `meta` (fitted
rates, analytic predictions, E, L, …), the notes, the manifest parameters and the array index.
The light-ray scenario adds a legend colouring each ray by impact parameter with its deflection
angle.

## Browser-test hook

`window.__gravsim` exposes, for `tests/test_web.py`:

```js
{
  ready,                  // true once the first scenario has been rendered
  scenario,               // current scenario id (getter)
  loadScenario(id),       // Promise, resolves after the scenario is built and a frame rendered
  setTime(fraction),      // 0 … 1 along the series
  setAmplification(k),    // k >= 1; no-op where the slider is hidden
  play(), pause(),
  stats: { frames, lastError },   // rendered-frame counter; last error message or null
}
```

Playwright's headless Chromium renders through SwiftShader; it sometimes logs the *warning*
`GL Driver Message (OpenGL, Performance, …): GPU stall due to ReadPixels` when its software
compositor reads the WebGL frame back. That message comes from the browser, not from the
page, so tests should fail on console messages of type `error` only.

## Data format

The scenario list is a JSON array of `{id, title, blurb, url, params}`; `url` is resolved
against the page (`data/<id>.bin` for the static export, `/api/scenario/<id>` for the server).

Each scenario is one binary payload written by `gravsim.web.payload.encode`:

```
uint32 little-endian  header length  (4 bytes)
UTF-8 JSON header     {"meta": {...}, "index": [{"name", "dtype", "shape", "offset", "nbytes"}, ...]}
raw arrays            little-endian, concatenated; `offset` counts from the start of this block
```

`dtype` is `float32` or `int32` (the decoder also accepts the other numpy scalar types and
absolute file offsets). The decoder builds typed-array views over the fetched `ArrayBuffer`
when the array is 4-byte aligned and copies otherwise.

Arrays per scenario (units in `meta.units`, J2000 ecliptic axes, +z north):

| Scenario | Arrays |
|---|---|
| `solar_system`, `rosette`, `earth_moon` | `t` (T,) days since J2000.0; `pos_newton`, `pos_gr` (T, N, 3) AU in the order of `meta.names`; `delta_gr` (T, N, 3) = GR − Newton computed in float64 before the float32 cast (optional, used for the amplified display when present); `peri_idx_newton`, `peri_idx_gr` (K,) int32 sample indices of Mercury's perihelia (seeds for the trail markers) |
| `black_hole` | `geodesic` (T, 3) x, y, z in GM/c²; optional `tau` (T,) |
| `light_rays` | `ray_<k>` (P, 2 or 3) polylines; `meta` lists the impact parameters and deflection angles |

`meta` always carries `id`, `title`, `names`, `units`, `frame`, `t0` and `n_samples`; any other
numeric entry is shown in the info panel, strings go to the notes, and `meta.arrays` documents
the arrays (shown as tooltips in the array index).

## Labels and the info panel

* Labels are anchored with `CSS2DObject.center` (three r160's `CSS2DRenderer` rewrites each
  label's `style.transform` every frame, so stylesheet offsets would be ignored). Body labels
  sit on the drawn sphere's edge, above-right; the Moon's goes below-right so Earth and Moon
  stay readable.
* The same renderer ignores the visibility of a label's ancestors; `syncLabelVisibility` combines
  each label's own toggle (`userData.show`) with its parents' before every label render, so the
  "Ecliptic grid" toggle hides the ring labels too. Grid ring labels also hide when their ring is
  under 40 px on screen, inside the drawn Sun, or inside the named rings (horizon, photon sphere,
  ISCO).
* The Numbers panel lists the headline results first (`HEADLINE_KEYS` in `app.js`, with
  human-readable names and units derived from the key suffixes) and folds everything else
  (solver settings, GM values, raw lists) under "Run details".
