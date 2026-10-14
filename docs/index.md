# gravsim: Newton vs. Einstein in the solar system

gravsim is a guided undergraduate project: a Python package that integrates the solar system
first with Newtonian gravity and then with General Relativity, and that makes the difference
*measurable*. Its spine is a detective story — Le Verrier's unexplained 43 arcseconds per
century of Mercury's perihelion advance, the hunt for planet Vulcan, Einstein's 1915
calculation — and every phase moves toward reproducing that anomaly from the simulation itself.

These pages hold the theory behind the code (with the derivations the module docstrings only
summarise), the index of the lessons, a description of the interactive lab and the validation
results. The engineering contract (units, frames, API) is `CONTRIBUTING.md` in the repository; the
physics plan is `ROADMAP.md`.

## The ladder of models

| level | model | what it teaches | module |
|---|---|---|---|
| 0 | Kepler (analytic) | ground truth for validation | `analysis/kepler.py` |
| 1 | Newton, two bodies | integrators, conservation laws | `forces/newton.py`, `integrators/` |
| 2 | Newton, Sun + 8 planets | perturbations; Mercury's 531″/century from the planets | `forces/newton.py` |
| 3 | Schwarzschild geodesics | effective potential, ISCO, photon sphere | `metric/schwarzschild.py` |
| 4 | 1PN test-particle correction | the "GR term"; 43″/century; agrees with level 3 | `forces/pn1.py` |
| 5 | EIH N-body 1PN | what JPL ephemerides do; 532 + 43 ≈ 575″/century; agrees with level 4 to 10⁻⁵″/cy | `forces/eih.py` |
| 6 | PPN parameters β, γ | precession ∝ (2 + 2γ − β)/3; "what if gravity were a different theory?" | `forces/pn1.py`, `forces/eih.py` (both take them) |
| 7 | light (null geodesics) | Eddington 1919: 1.75″ at the limb; capture and shadow; Shapiro delay | `metric/null.py` |

All seven levels are implemented, validated and taught in Lessons 01–08; the theory pages are
[Newton](theory/newton.md), [Integrators](theory/integrators.md),
[Measurement](theory/measurement.md), [1PN](theory/pn1.md),
[Schwarzschild](theory/schwarzschild.md), [EIH](theory/eih.md), [PPN](theory/ppn.md) and
[Light](theory/light.md).

## Conventions used throughout

* **Units**: AU, day, masses as $GM$ in AU³/day² ($GM_\odot = k^2 = 2.959\times10^{-4}$); the
  speed of light $c = 173.1446$ AU/day is always an explicit parameter so that lessons can
  scale it down.
* **Frame**: solar-system barycentre, J2000 ecliptic; $t$ in days since J2000.0 TDB.
* **Bodies**: Sun + 8 planets (Earth = Earth–Moon barycentre, outer planets = system
  barycentres). The Moon, Pluto and the asteroids are out of scope (≈0.1″/century on Mercury).
* **Visualisation**: orbits in 3-D (matplotlib for static figures, plotly in the app);
  measurement plots (precession versus time, residuals) in 2-D.

## Running things

```bash
scripts/setup_env.sh                              # uv venv + editable install (dev extras)
.venv/bin/python -m pytest -q                     # fast suite (~1 min)
.venv/bin/python -m pytest -q -m slow             # century runs + notebook execution (~4 min)
.venv/bin/python scripts/validate.py              # all 47 validation checks + figures/ (~75 s)
.venv/bin/jupyter lab notebooks/                  # the lessons
uv pip install -e ".[app]" && .venv/bin/streamlit run app/streamlit_app.py   # the lab
uv pip install -e ".[docs]" && .venv/bin/mkdocs serve                       # these pages
```
