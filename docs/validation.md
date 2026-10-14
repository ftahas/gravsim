# Validation

The validation targets of `ROADMAP.md` §4 are the test suite: `tests/test_validation.py`
(fast variants over ten years, century-long variants marked `slow`) and
`scripts/validate.py`, which prints the table below and draws the headline figures in
`figures/`. Everything runs from the cached ephemerides; no test touches the network.

```bash
.venv/bin/python -m pytest -q                 # fast suite: unit tests + 10-year validation, ~2 min
.venv/bin/python -m pytest -q -m slow         # century runs + notebook execution, ~6 min
.venv/bin/python -m pytest -q -m web          # browser tests of the web front end (playwright)
.venv/bin/python scripts/validate.py          # 100-year runs, table + figures, ~2.5 min
```

## Targets and results

Produced by `scripts/validate.py` (DOP853, `rtol=1e-12`, `atol=1e-15`, daily samples, 100
years; "fit" = least-squares slope of the osculating longitude of perihelion, "passages" = the
same from perihelion passages). All 64 checks pass.

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
| N-body EIH (every body relativistic), Mercury, fit | 572.2 ″/cy | 574.6 | 3 |
| N-body EIH − Newton: Mercury / Venus / EMB / Mars | 42.977 / 8.637 / 3.851 / 1.351 ″/cy | 42.98 / 8.62 / 3.84 / 1.35 | 0.10 |
| N-body EIH − (Newton + 1PN), Mercury: secular difference; max separation | 1.5e−5 ″/cy; 5.1 km | 0; — | 0.02; report |
| REBOUNDx `gr_full` (IAS15) vs EIH, Sun + 8 planets, 10 yr: max separation | 0.035 km | 0 | 10 |
| PPN two-body, 10 yr: fit / [(2+2γ−β)/3 × 42.98], (β,γ) = (1,1) / (0,1) / (1,0) / (2,1) | 1.0002 / 1.0001 / 1.0003 / 1.0004 | 1 | 0.01 |
| light deflection at the limb, exact null geodesic; Soldner 2GM/(c²R☉) | 1.7512″; 0.8756″ | 1.751; 0.8755 | 0.001 |
| Shapiro delay Earth → Mars grazing the Sun, one way; exact / weak field | 123.6 μs; 1.0001 | ~124; 1 | report; 0.01 |
| Sun + Mercury + solar J₂ (2.2·10⁻⁷) − Newton, 100 yr | +0.0278 ″/cy | 0.028 | 0.005 |
| J₂ × 1000, 10 yr: node / argument rates over Kaula (solar-equator frame) | 1.0000 / 0.9999 | 1 | 0.03 |
| J × 10⁵, 10 yr: node / argument rates over Lense–Thirring | 1.0000 / 1.0000 | 1 | 0.03 |
| Sun + Mercury + Lense–Thirring (real spin) − Newton, 100 yr | −0.0020 ″/cy | ~−0.002 | report |
| Moon, ten-body run, 10 yr: sidereal month; perigee / node rates | 27.3217 d; +40.59 / −19.36 °/yr | 27.3217; 40.69 / −19.34 | 0.05; 1.5 / 1.0 |
| … minus the Horizons series fitted over the same span | −0.002 / +0.007 °/yr | 0 | 0.3 |
| Moon vs Horizons, max over 2 yr, Newton + 1PN / Newton | 37.6 / 157 km | 0 / — | 500 / report |
| real Earth–Moon barycentre vs Horizons, 10 yr; EMB run vs real barycentre after 10 yr | 6.8 km; 1170 km (2εnat = 1101) | — | report |
| Mercury (1PN − Newton) with Earth + Moon minus the EMB value; Newtonian shift | 6·10⁻⁸ ″/cy; −0.0025 | 0; ~−0.0025 | 0.1; report |

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
* **EIH and the Sun-only 1PN term agree** on every secular rate ($1.5\times10^{-5}$″/cy for
  Mercury) and differ by at most 5 km per century: the Sun's barycentric velocity
  ($v_\odot/v_\mercury \approx 4\times10^{-4}$) and the planets' potentials at Mercury enter the
  1PN acceleration only through periodic terms (see [EIH](theory/eih.md)). REBOUNDx `gr_full`
  integrates the same equations independently and agrees with `forces.eih` to 35 m over ten
  years (`tests/test_eih.py` also checks the accelerations themselves to $10^{-10}$).
* **PPN**: Mercury measures only $(2+2\gamma-\beta)/3$ ([PPN](theory/ppn.md)); the four runs
  reproduce the factor to a few parts in $10^4$ (`figures/ppn_precession_grid.png`).
* **Light**: the exact null geodesic exceeds $4GM/(c^2 b)$ by $(15\pi/16)\,GM/(c^2 b)$
  ($10^{-5}$ at the limb) and the Shapiro transit exceeds the weak-field formula by $10^{-4}$
  (second-order end-point and Fermat terms, [Light](theory/light.md)); both are far inside the
  tolerances and far outside the integrator noise.
* **The competitors** ([What competes with GR](theory/extras.md)): the oblate Sun gives
  $+0.0278''$/cy for $J_2 = 2.2\times10^{-7}$, three times the ranging precision; frame dragging
  $-0.0020''$; the Earth–Moon pair as one body $-0.0025''$. The differenced node and argument
  rates reproduce Kaula's and the Lense–Thirring formulas to $10^{-4}$ with the effects amplified
  a thousand- and a hundred-thousand-fold (`figures/lense_thirring_amplified.png`), and the
  modelled terms sum to the $+0.025''$ Horizons − EIH residual of the [EIH page](theory/eih.md)
  (`figures/precession_budget.png`).
* **The Moon**: the ten-body run gives the sidereal month to $10^{-5}$ d and the perigee and
  node rates to $0.01°$/yr of the Horizons series over the same span (the $0.1°$/yr offsets from
  the secular 40.69 / −19.34 are the long-period lunar inequalities over ten years, present in
  the ephemeris too). The 2-year "Moon within 100 km of the EMB" target of the plan is replaced
  by two physical statements: the real Earth–Moon barycentre follows Horizons to 7 km over ten
  years, while the EMB *point mass* lags it by $2\varepsilon nat$ ≈ 110 km/yr because the pair
  feels on average $\varepsilon = \tfrac34 w_Ew_M(a_M/a_E)^2 \approx 6\times10^{-8}$ more solar
  pull than a point mass. Mercury's GR signal is unchanged to $6\times10^{-8}$″/cy
  (`figures/moon_orbit.png`).

## Unit-level tests

One `tests/test_<module>.py` per package module: `test_constants.py` (unit conversions against
IAU/CODATA definitions, $GM_\odot = k^2$, DE440 mass ratios, body tables), `test_bodies.py`
(containers, shape validation, frames, immutability), `test_horizons.py` (cache consistency:
snapshot = series sample 0, ecliptic frame from the inclinations, Kepler periods, round trips,
mocked Horizons queries; one real-network test marked `network`), `test_forces.py`,
`test_integrators.py`, `test_analysis.py`, `test_schwarzschild.py`, `test_eih.py` (hand-written
EIH sums, test-particle limit = Newton + pn1, permutation symmetry, REBOUNDx accelerations and
positions, PPN scaling), `test_null.py` (deflection series and strong-field limit, ray paths,
Shapiro forms and the exact transit), `test_viz.py`, `test_extras.py` (J₂ as the gradient of its potential, momentum conservation,
the Lense–Thirring vector formula, Kaula and Lense–Thirring rates), `test_moon.py` (snapshot
consistency, lunar orbit, EMB drift, Mercury with the Moon), `test_web_backend.py` (scenario
registry and physics, payload round trips, the FastAPI app, the export script), plus
`test_app.py`, `test_lab_ppn.py`, `test_lab_light.py` and `test_lab_extras.py` (lab back ends
and headless Streamlit runs), `test_notebooks.py` (structure; execution under `slow`) and
`test_web.py` (headless Chromium, marker `web`). Analytic references (Kepler, harmonic oscillator, known formulas) are
preferred over snapshot tests throughout.
