# Solar System: Newton vs. Einstein — Project Roadmap

A pedagogical Python simulation that integrates the solar system first with Newtonian
gravity, then with General Relativity (post-Newtonian corrections and exact Schwarzschild
geodesics), and makes the differences measurable and visible.

The spine of the project is the historical detective story: Le Verrier's unexplained
43"/century of Mercury's perihelion advance → the hunt for planet Vulcan → Einstein (1915).
Every phase should move toward reproducing that anomaly *from the simulation itself*.

---

## 1. Guiding design decisions

### 1.1 The pedagogical ladder of models

| Level | Model | What it teaches | Where it lives |
|---|---|---|---|
| 0 | Kepler (analytic) | Ground truth for validation | `analysis/kepler.py` |
| 1 | Newton, 2-body numeric | Integrators, conservation laws | `forces/newton.py` |
| 2 | Newton, N-body (Sun + 8 planets) | Perturbations; Mercury's ~532"/century from planets alone | `forces/newton.py` |
| 3 | Schwarzschild geodesic (exact, test particle) | Effective potential, ISCO, photon sphere; GR intuition | `metric/schwarzschild.py` |
| 4 | 1PN test-particle correction | The standard "GR term"; show it agrees with level 3 in weak field | `forces/pn1.py` |
| 5 | EIH 1PN N-body (all bodies relativistic) | What JPL ephemerides actually do; 532 + 43 ≈ 575"/century | `forces/eih.py` |
| 6 | PPN parameters β, γ | Precession ∝ (2 + 2γ − β)/3; "what if gravity were a different theory?" | `beta`, `gamma` arguments of `forces/pn1.py` and `forces/eih.py` |
| 7 | Null geodesics (light) | Eddington 1919: 1.75" deflection at the limb; Shapiro delay | `metric/null.py` |
| 8 (opt) | Solar J2, Lense-Thirring | What competes with GR at the 0.03" level | `forces/extras.py` |

Key formulas:

- 1PN test-particle acceleration (harmonic coords, central mass M):
  `a = -GM r/r^3 + (GM/(c^2 r^3)) * [ (4GM/r - v^2) r + 4 (r·v) v ]`
- Precession per orbit: `Δφ = (2+2γ−β)/3 · 6πGM / (c^2 a (1−e^2))`; GR has β=γ=1.
- Schwarzschild effective potential (per unit mass, L = specific angular momentum):
  `V_eff(r) = -GM/r + L^2/(2 r^2) - G M L^2/(c^2 r^3)`; the last term is the GR correction.
- Light deflection: `δ = 4GM/(c^2 b)` → 1.751" for b = R_sun.

Reference numbers for Mercury (inertial, barycentric frame):

| Contribution | "/century |
|---|---|
| Planetary perturbations (Newtonian N-body) | ~531.6 |
| GR (1PN) | 42.98 |
| Solar oblateness J2 | ~0.03 |
| Total observed (inertial frame) | ~574.1 |

Other planets, GR only: Venus 8.62, Earth 3.84, Mars 1.35 "/century.

### 1.2 GR effects are invisible to the eye — design for measurement

In the real solar system 1PN corrections are ~1e-8 relative. An animation with GR "on" looks
identical to one with it "off". So the primary outputs are **measurement plots**, and only
secondarily animations:

1. **Perihelion longitude vs. time** for Mercury: a straight line with slope 43"/century
   when GR is on. This is the smoking-gun plot.
2. **Newton minus GR difference** of the same quantity: the periodic planetary wobble cancels,
   leaving a clean secular drift. (Same integrator, same step → numerical error also cancels.)
3. **Residuals vs. JPL Horizons**: Horizons ephemerides include GR. Newton-only run vs. Horizons
   over decades shows a systematic Mercury drift; the 1PN run shows none. This is Le Verrier's
   anomaly appearing "from data".
4. **Precession rate vs. a(1−e²)** for all planets: simulation points on the GR curve.

For *visible* GR, provide an **exaggeration mode**:

- Scale c down (e.g. c/100 → precession ×10⁴ → ~0.3°/orbit; a rosette appears in ~100 orbits).
  1PN remains valid as long as v/c stays small (~0.02 at c/100 for Mercury).
- Or switch to exact Schwarzschild geodesics around a compact mass (valid for any field strength
  outside the ISCO): capture orbits, zoom-whirl, photon sphere.

### 1.3 Numerics are the hidden hard part

The GR signal is ~5e-7 rad per Mercury orbit. Integrator error must be well below that:

- Leapfrog's spurious precession on a Kepler orbit is O((2π/N)²) per orbit, N = steps/orbit.
  Resolving 5e-7 would need N ≳ 10⁴ (≈10-minute steps). Not pedagogically wrong, but slow.
- Use higher order for production runs: Yoshida 4th/6th-order symplectic, or SciPy DOP853
  (adaptive, 8th order, rtol ~1e-12), or REBOUND IAS15 as an external oracle.
- The 1PN force depends on velocity, which breaks the assumptions of plain kick-drift-kick
  leapfrog. Options: implicit (iterated) kick, or use RK/DOP853/IAS15 for GR runs.
  REBOUNDx handles this internally — useful cross-check.
- Build a **noise-floor test first**: Newton 2-body must give precession ≈ 0 (< 0.5"/century)
  before trusting any GR number.
- Double precision is enough (1PN ~1e-8 relative vs. 1e-16 machine epsilon) but accumulate
  carefully; energy error target ΔE/E < 1e-10 over a century.

### 1.4 Units and frames

- Internal units: AU, solar mass, day (or year). G = k² with Gaussian k, c = 173.1446 AU/day.
  Alternative: astropy Quantities at the boundaries, plain floats inside.
- Frame: solar-system barycentric, ICRF axes, J2000.0 epoch. Keep the Sun as a body.
- Initial conditions from JPL Horizons via `astroquery.jplhorizons`, cached to JSON so the
  project runs offline. Bodies: Sun, Mercury … Neptune; optional Moon (or Earth-Moon
  barycenter), Pluto, Ceres/Vesta/Pallas (together they shift Mercury's rate by ~0.1").
- Mention (don't model) TDB vs. TT coordinate-time subtleties in the GR lesson.

### 1.5 Stack

| Need | Choice | Why |
|---|---|---|
| Arrays, ODEs | numpy, scipy (`solve_ivp` DOP853) | standard, fast enough |
| Hot loops | numba (`@njit`) | century-long runs in seconds |
| Constants, units, time | astropy | authoritative values |
| Ephemerides | astroquery (Horizons) | real initial conditions and ground truth |
| External oracle | rebound + reboundx (`gr`, `gr_potential`, `gr_full`) | independent check of 1PN/EIH |
| Plots | matplotlib; plotly for 3D/interactive | lessons + app |
| Lessons | Jupyter notebooks | narrative, one per phase |
| App | Streamlit (v1) → optional FastAPI + Three.js (v2) | fastest route to an interactive lab |
| Tests | pytest | validation targets below are the test suite |
| Docs | mkdocs-material | theory pages with derivations |

Principle: **implement the physics and integrators yourself** (that is the pedagogy);
use REBOUND/Horizons only as independent references.

---

## 2. Repository layout

```
gravsim/
  gravsim/
    constants.py        # G, c, masses, unit conversions
    bodies.py           # Body, SystemState (positions, velocities, masses)
    forces/
      newton.py         # pairwise Newtonian acceleration
      pn1.py            # 1PN test-particle correction (central mass)
      eih.py            # Einstein–Infeld–Hoffmann N-body 1PN
      extras.py         # solar J2, Lense–Thirring (optional)
    integrators/
      euler.py  leapfrog.py  rk4.py  yoshida.py  scipy_wrap.py
    metric/
      schwarzschild.py  # exact timelike geodesics, effective potential
      null.py           # photon geodesics, deflection, Shapiro delay
    analysis/
      kepler.py         # analytic 2-body solution
      elements.py       # osculating elements, LRL vector
      precession.py     # perihelion tracking and rate fits
      conserved.py      # energy, angular momentum, LRL drift
      compare.py        # run-vs-run and run-vs-Horizons residuals
    data/
      horizons.py       # fetch + cache
      j2000.json        # cached state vectors
    viz/
      plots.py  animate.py
  notebooks/
    01_integrators … 05_black_hole_orbits, 06_le_verrier_vulcan_einstein, 07_ppn_playground, 08_eddington_1919
  app/
    streamlit_app.py
  tests/
  docs/
```

---

## 3. Phases

Each phase ends with a notebook (the lesson) and tests (the validation).

### Phase 0 — Foundations (small)
- Package skeleton, constants, unit conventions, `SystemState` dataclass.
- Horizons fetch + JSON cache for J2000.0 (Sun + 8 planets + Moon).
- Decide on the integrator interface: `step(state, dt, accel_fn) -> state`,
  `accel_fn(state) -> accelerations` so forces and integrators compose freely.
- CI: pytest, ruff.

### Phase 1 — Newton, two-body (small)
- Euler, leapfrog, RK4, Yoshida-4 on Sun–Mercury.
- Plots: orbit, energy/angular-momentum error vs. time, error vs. step size (log-log).
- Validation: position vs. Kepler analytic solution; energy drift characteristics.
- Lesson 01: "Why the integrator matters" (Euler spirals out, RK4 drifts, leapfrog stays bounded).

### Phase 2 — Newton, N-body solar system (medium)
- Vectorized/numba pairwise forces; run 1 year → 1 century.
- Validation: positions vs. Horizons at 1 and 10 years (measured: Newton drifts to ~1.5e3 km at 10 yr and ~1e4 km at 100 yr, almost all of it the missing GR term; Newton+1PN stays within 5 km at 10 yr, 200 km at 100 yr).
- Lesson 02: "The clockwork solar system" (Jupiter's pull on Mercury, the Sun's wobble).

### Phase 3 — Measurement toolkit (medium, the critical phase)
- Osculating elements each step; Laplace–Runge–Lenz vector → perihelion longitude ω(t).
- Perihelion-passage detection (local minima of r, refined by interpolation).
- Linear fit of ω(t) with uncertainty; run-vs-run differencing.
- Validation: 2-body Newton precession < 0.5"/century with the production integrator;
  N-body Newton Mercury ≈ 531.6 ± 1 "/century.
- Lesson 03: "Measuring a drift of 0.1 arcsecond per orbit".

### Phase 4 — GR I: one body curves spacetime (medium)
- `metric/schwarzschild.py`: exact geodesic integration (proper time or coordinate time),
  effective potential, circular-orbit stability, ISCO.
- `forces/pn1.py`: 1PN test-particle correction; `c` as a parameter.
- Validation: Mercury 2-body 1PN = 42.98 ± 0.1 "/century; exact geodesic agrees with 1PN
  for v/c ≪ 1; exaggerated-c rosette matches the analytic formula.
- Lesson 04: "Mercury's extra 43 arcseconds" and Lesson 05: "Orbits around a black hole"
  (effective potential explorer, zoom-whirl, capture).

### Phase 5 — GR II: everybody curves spacetime (medium–large) — DONE
- [x] `forces/eih.py`: EIH 1PN N-body equations (as in JPL DE ephemerides), PPN-parametrised.
- [x] β, γ knobs: taken directly by `make_pn1` and `make_eih` (no separate `forces/ppn.py`).
- [x] Velocity-dependent-force integrator strategy: DOP853 for production, implicit kick in
      the symplectic classes.
- [x] Cross-check against REBOUNDx `gr_full`: accelerations to 1e-10, positions to 0.035 km
      over 10 yr (`tests/test_eih.py`, `scripts/validate.py` check 12).
- [x] Validation: N-body EIH Mercury 572.2 (574.6 ± 3, same OLS bias as Newton's 529.2 vs
      531.6); EIH − Newton 42.977 / 8.637 / 3.851 / 1.351 for Mercury / Venus / EMB / Mars;
      EIH − (Sun-only 1PN) 1.5e-5 "/century; Horizons residual: Newton drifts, EIH doesn't.
- [x] Lesson 06: "Le Verrier, Vulcan, Einstein" (the anomaly reproduced from data).
- [x] Lesson 07: "What if gravity were a different theory?" (PPN).
- Status: implemented, reviewed and validated; theory pages `docs/theory/eih.md`, `ppn.md`;
  app lab 5 (PPN playground). Deviation: no `forces/ppn.py` module.

### Phase 6 — Light (medium) — DONE
- [x] Null geodesics in Schwarzschild (`metric/null.py`): exact deflection 1.7512" at the limb
      (Soldner 0.8756"), ray paths and bundles, capture at b_crit = 3√3 GM/c², Shapiro delay
      in weak-field, conjunction and exact-geodesic forms.
- [x] Shapiro delay for the Earth–Mars signal at superior conjunction: 123.6 μs one way, 247 μs
      round trip; exact geodesic / weak field = 1.0001.
- [x] Lesson 08: "Eddington 1919".
- Status: implemented, reviewed and validated; theory page `docs/theory/light.md`; app lab 6
  (light-bending lab).

### Phase 7 — Interactive app (medium–large) — DONE
Streamlit lab with cached precomputed runs (`app/streamlit_app.py`):
1. [x] Integrator lab (choose method/step, watch energy error).
2. [x] Solar system: Newton/GR toggle, c slider, difference-vector view (amplified).
3. [x] Precession tracker: live ω(t) for selected planet, fitted rate readout.
4. [x] Effective-potential explorer: drag E and L, see the orbit.
5. [x] PPN playground (`app/lab_ppn.py`).
6. [x] Light-bending lab (`app/lab_light.py`).
- Deviation: labs 5 and 6 are numbered in lesson order (PPN = lessons 06–07, light = lesson 08),
  the reverse of the list above.

### Phase 8 — Polish (ongoing)
- [x] mkdocs theory pages (derivations of 1PN term, EIH, PPN precession formula, light); `mkdocs build --strict` passes.
- [x] Packaging (`pip install -e .`), README with the headline plots and the validation table.
- [x] CI: `.github/workflows/ci.yml` runs ruff and the fast suite (REBOUNDx oracle tests skip when absent).
- [ ] Optional: Three.js front end, Lense-Thirring/J2, asteroid belt, Moon, CITATION file.

---

## 4. Validation targets (= test suite)

| Test | Expected | Tolerance |
|---|---|---|
| 2-body vs. Kepler analytic, 100 orbits | positions | < 1e-7 AU at DOP853 defaults (rtol 1e-12); floor ~4e-10 AU at rtol 2e-14 |
| Symplectic energy error, 1 century | ΔE/E | < 1e-10 |
| 2-body Newton precession (noise floor) | 0 | < 0.5 "/century |
| 2-body 1PN Mercury | 42.98 "/century | ± 0.1 |
| Schwarzschild exact vs. 1PN, Mercury | agree | < 0.01 "/century |
| Exaggerated c (c/100) rosette | analytic Δφ | < 1 % |
| N-body Newton Mercury | 531.6 "/century | ± 3 (100-yr OLS slope is biased ~−2.4 by long-period terms; Horizons itself gives 572.2 vs 574.1 by the same method) |
| N-body EIH Mercury | 574.6 "/century | ± 3 (same bias); run-vs-run difference vs Newton 42.98 ± 0.1 |
| Venus / Earth / Mars GR (run-vs-run) | 8.62 / 3.84 / 1.35 | ± 0.1 |
| PPN with β=γ=1 | equals GR run | bitwise-ish |
| PPN two-body (β,γ) = (0,1), (1,0), (2,1) | (2+2γ−β)/3 × 42.98 "/century | ± 1 % (measured 1e-4 to 4e-4) |
| Light deflection at limb | 1.751" | ± 0.001 (constants, and the exact null geodesic: 1.7512") |
| Light deflection, Soldner corpuscle | 0.8756" (half of GR) | ± 0.001 |
| Shapiro delay, Earth–Mars at the limb, one way | 124 μs (247 round trip) | reported; exact geodesic / weak field 1 ± 0.01 |
| EIH − (Sun-only 1PN) Mercury secular rate | 0 | < 0.02 "/century |
| REBOUNDx `gr_full` vs EIH, 10 yr | positions | < 10 km (measured 0.035) |
| Newton vs. Horizons, 10 yr | km-level | documented, not asserted tightly |

---

## 5. Decisions (settled 2026-10-13)

1. **Audience**: guided project for undergraduate studies → EIH 1PN N-body and PPN β/γ are
   core deliverables, not optional. Every lesson states the physics before the code.
2. **Primary artifact**: notebooks and the interactive app advance in lockstep; each phase
   ships its notebook *and* the corresponding app lab.
3. **Own implementations** of forces and integrators; REBOUND/REBOUNDx and Horizons are
   independent oracles only.
4. **Visualisation is 3D** (matplotlib 3D for static figures, plotly for interactive; the app
   renders orbits in 3D). Measurement plots (precession vs. time, residuals) stay 2D.
5. **Bodies**: Sun + 8 planets. Earth is represented by the Earth–Moon barycenter with the
   combined GM; outer planets by their system barycenters with system GMs. Pluto, the Moon and
   asteroids are out of scope (they shift Mercury's Newtonian rate by ~0.1"/century, noted in
   the lesson).
6. **Frame and units**: barycentric, J2000 ecliptic frame; AU, day, GM in AU³/day².

## 6. Pitfalls checklist

- Forgetting the Sun moves (barycentric frame).
- Measuring precession from raw perihelion passages in the N-body case without differencing
  → periodic planetary wobble swamps the secular term.
- Using leapfrog with a velocity-dependent 1PN force and expecting symplectic behaviour.
- Reading off precession from an integrator whose own numerical precession is ≫ 43"/century.
- Pushing the exaggerated-c mode until v/c is no longer small while still using 1PN.
- Confusing the 5025"/century equinox precession (Earth-frame coordinate effect) with the
  574" inertial-frame value.
- Comparing Schwarzschild-coordinate positions with harmonic-coordinate (Horizons) positions;
  precession rates are coordinate-invariant, positions are not.
