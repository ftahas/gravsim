# Validation

The validation targets of `ROADMAP.md` §4 are the test suite: `tests/test_validation.py`
(fast variants over ten years, century-long variants marked `slow`) and
`scripts/validate.py`, which prints the table below and draws the headline figures in
`figures/`. Everything runs from the cached ephemerides; no test touches the network.

```bash
.venv/bin/python -m pytest -q                 # fast suite: unit tests + 10-year validation, ~15 s
.venv/bin/python -m pytest -q -m slow         # century runs + notebook execution, ~2 min
.venv/bin/python scripts/validate.py          # 100-year runs, table + figures, ~40 s
```

## Targets and results

Produced by `scripts/validate.py` (DOP853, `rtol=1e-12`, `atol=1e-15`, daily samples, 100
years; "fit" = least-squares slope of the osculating longitude of perihelion, "passages" = the
same from perihelion passages). All 31 checks pass.

| check | value | expected | tolerance |
|---|---|---|---|
| 2-body Newton noise floor, Mercury, fit / passages | 0.0012 / 0.0012 ″/cy | 0 | 0.5 |
| Leapfrog, Yoshida4 at dt = 0.5 d, 10 yr: spurious precession | −196 972 / 2 634 ″/cy | — | report |
| 2-body (Newton + 1PN) − Newton, Mercury | 42.9805 ″/cy | 42.98 | 0.10 |
| 2-body Newton + 1PN, undifferenced fit | 42.9817 ″/cy | 42.98 | 0.10 |
| analytic 6πGM/(c²a(1−e²)), Mercury's osculating a, e | 42.9807 ″/cy | 42.98 | 0.10 |
| Schwarzschild exact (Binet), Mercury; exact − 1PN | 42.9808; 1.0e−4 ″/cy | 42.98; 0 | 0.10; 0.01 |
| c/100, 50 orbits: fit / differenced / passages over analytic | 0.9981 / 0.9981 / 0.9985 | 1 | 0.01 |
| N-body Newton, Mercury, fit / passages | 529.2 / 529.3 ″/cy | 531.6 | 3 |
| N-body Newton + 1PN, Mercury, fit / passages | 572.2 / 572.2 ″/cy | 574.6 | 3 |
| N-body (Newton + 1PN) − Newton: Mercury / Venus / EMB / Mars | 42.977 / 8.637 / 3.851 / 1.351 ″/cy | 42.98 / 8.62 / 3.84 / 1.35 | 0.10 |
| Horizons series, Mercury, fit / passages | 572.2 / 572.3 ″/cy | 574.1 | 5 |
| Newton vs Horizons, Mercury: mean / final separation | 5 162 / 5 743 km | — | report |
| Newton + 1PN vs Horizons, Mercury: mean / final separation | 72 / 176 km | — | report |
| relative energy error, N-body DOP853, 100 yr | 6.8e−14 | 0 | 1e−9 |
| relative energy error, N-body Yoshida4, dt = 0.5 d, 10 yr | 2.2e−9 | 0 | 1e−8 |
| light deflection at the limb, 4GM/(c²R☉) | 1.7512″ | 1.751 | 0.001 |

## Reading the table

* **Differences are exact, absolutes are biased.** The N-body absolute rates (529.2 Newton,
  572.2 with 1PN) sit 2.4″/cy below the textbook 531.6/574.6 because an OLS slope of the
  osculating series over one century is biased by the long-period planetary terms (the
  Horizons series itself, fitted the same way, gives 572.2 rather than 574.1), and the omitted
  Moon, asteroids and Pluto contribute ~0.1″/cy. The differences on a common grid are free of
  this: $42.977\pm0.001$ (N-body) and $42.9805\pm0.001$ (two-body) against 42.98.
* **The noise floor** (0.0012″/cy) is four orders of magnitude below the signal; the fixed-step
  schemes at half-day steps are not, which is Lesson 01.
* **Horizons contains GR**, so the Newtonian run drifts away along the orbit by thousands of km
  per century while the 1PN run stays within ~200 km (`figures/mercury_vs_horizons.png`).
* The ROADMAP row "2-body vs Kepler < 1e−10 AU over 100 orbits" is not attainable in double
  precision with absolute coordinates (floor ~4e−10 AU); the noise-floor row replaces it.

## Unit-level tests

One `tests/test_<module>.py` per package module: `test_constants.py` (unit conversions against
IAU/CODATA definitions, $GM_\odot = k^2$, DE440 mass ratios, body tables), `test_bodies.py`
(containers, shape validation, frames, immutability), `test_horizons.py` (cache consistency:
snapshot = series sample 0, ecliptic frame from the inclinations, Kepler periods, round trips,
mocked Horizons queries; one real-network test marked `network`), `test_forces.py`,
`test_integrators.py`, `test_analysis.py`, `test_schwarzschild.py`, `test_viz.py`, plus
`test_app.py` (lab back end and a headless Streamlit run) and `test_notebooks.py` (structure;
execution under `slow`). Analytic references (Kepler, harmonic oscillator, known formulas) are
preferred over snapshot tests throughout.
