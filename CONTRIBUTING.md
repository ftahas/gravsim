# gravsim — package contract

Pedagogical solar-system simulator: Newtonian gravity, then General Relativity (1PN test
particle, EIH N-body, PPN, exact Schwarzschild geodesics, light). See ROADMAP.md for the
phases and the physics; this file is the engineering contract every module must follow.

## Environment
- Python 3.12 venv at `.venv/` (managed with uv). Run everything through it:
  `.venv/bin/python -m pytest -q` (fast tests), `.venv/bin/python -m pytest -q -m slow` (validation).
- Dependencies: numpy, scipy, matplotlib, astropy, astroquery, numba. Dev: pytest, ruff, jupyter.
- `ruff check gravsim tests` must pass (line length 100).

## Units, frame, conventions
- Length AU, time day, masses as **GM in AU^3/day^2**. Never use G or kg internally.
- Constants live in `gravsim/constants.py` (`GM`, `GM_SUN`, `C_AU_DAY`, `ARCSEC_PER_RAD`, ...).
  The speed of light is **always a parameter** (`c=C_AU_DAY`) so lessons can scale it down.
- Frame: barycentric, J2000 ecliptic. Time `t` is days since J2000.0 TDB. Cached ephemeris in
  `gravsim/data/j2000.json` (snapshot) and `gravsim/data/horizons_ref.npz` (100-year series).
- Default system: `constants.DEFAULT_BODIES` = Sun + 8 planets (Earth = EMB, outer planets =
  system barycenters). Body 0 is the Sun; code takes the central body as an explicit argument rather than assuming
  it. Force laws (which only see arrays) take an integer index (`central=state.index("Sun")`);
  analysis routines that receive a Trajectory take the name (`central="Sun"`).
- Arrays are float64, C-contiguous; positions/velocities shape (N, 3); time series (T, N, 3).
- Functions are pure: never mutate input arrays; return new arrays.
- A body with `gm == 0` is a test particle: it feels forces but exerts none.

## Core types (`gravsim/bodies.py`)
- `SystemState(t, pos, vel, gm, names)` frozen dataclass; `.index(name)`, `.with_(...)`,
  `.relative_to(name)`, `.to_barycentric()`, `.subset(names)`.
- `Trajectory(t, pos, vel, gm, names, meta)`; `.body(name)`, `.relative(name, central)`,
  `.state_at(i)`, `.final_state`, `Trajectory.from_states(list)`.
  `meta` records integrator, dt, and whatever the caller adds (force law, c); free-form dict.

## Force laws (`gravsim/forces/`)
Every force law is an `AccelFn`: `accel(pos, vel, gm, t) -> acc` with shapes (N,3),(N,3),(N,),float -> (N,3).
- `newton.accel_newton(pos, vel, gm, t)` — pairwise Newtonian acceleration. numba-jitted kernel
  with a pure-numpy fallback (`accel_newton_numpy`), both exported and tested against each other.
- `pn1.make_pn1(c=C_AU_DAY, central=0, beta=1.0, gamma=1.0) -> AccelFn` — the 1PN **correction
  only** (test-particle form in harmonic coordinates, PPN-parametrised):
  `a_i = (gm_c / (c^2 r^3)) * [ (2(beta+gamma) gm_c / r - gamma v^2) r + 2(1+gamma) (r.v) v ]`
  with `r = pos_i - pos_central`, `v = vel_i - vel_central`, `gm_c = gm[central]`; row `central`
  is zero. beta=gamma=1 is GR: `(4 gm/r - v^2) r + 4 (r.v) v`.
- `pn1.make_newton_pn1(...)` — convenience: Newton + 1PN correction.
- `forces.combine(*fns) -> AccelFn` — sum of accelerations.
- `eih.make_eih(c=C_AU_DAY, beta=1.0, gamma=1.0) -> AccelFn` — (Phase 5) full Einstein–Infeld–
  Hoffmann acceleration **including** the Newtonian term (EIH does not separate cleanly).

## Integrators (`gravsim/integrators/`)
- Protocol: `class Integrator: name: str; order: int; step(pos, vel, gm, t, dt, accel) -> (pos, vel)`.
- Fixed-step classes: `Euler`, `SymplecticEuler`, `Leapfrog` (kick-drift-kick), `RK4`, `Yoshida4`,
  `Yoshida6`. Symplectic/KDK integrators take `vel_dependent: bool = False`; when True the final
  kick is made implicit by fixed-point iteration (`n_iter=2`) because 1PN forces depend on velocity.
- `driver.integrate(state, accel, t_end, dt, integrator, save_every=1) -> Trajectory`. Stores the
  initial state as sample 0. `dt` is adjusted (downwards, minimally) so an integer number of steps
  lands exactly on `t_end`; the actual dt goes in `meta["dt"]`.
- `scipy_wrap.integrate_scipy(state, accel, t_eval, method="DOP853", rtol=1e-12, atol=1e-15) -> Trajectory`
  wraps `scipy.integrate.solve_ivp` on the flattened state `y = [pos.ravel(), vel.ravel()]`.
  This is the production integrator for century-long GR runs.

## Analysis (`gravsim/analysis/`)
All routines are vectorised over a leading time axis: inputs `(..., 3)` give outputs `(...)`.
- `kepler.py`: `solve_kepler(M, e)`, `elements_to_state(a, e, inc, raan, argp, nu, gm)`,
  `state_to_elements(r, v, gm) -> dict` with keys `a, e, inc, raan, argp, nu, M, period, varpi`
  (angles in radians, `varpi = raan + argp` = longitude of perihelion), `propagate(r0, v0, gm, dt)`
  (analytic two-body propagation, elliptic orbits).
- `elements.py`: `lrl_vector(r, v, gm)` = (v x h)/gm - r/|r| (the eccentricity vector),
  `perihelion_longitude(r, v, gm)` (= varpi, radians, in (-pi, pi]), `node_vector`, `specific_h`.
- `precession.py`: `perihelion_longitude_series(traj, body, central="Sun") -> (t, varpi)` using
  relative coordinates and `gm = gm[central] + gm[body]`; `perihelion_passages(traj, body, central)
  -> (t_peri, varpi_peri)` from radial minima refined by parabolic interpolation;
  `fit_secular_rate(t, angle) -> SecularFit(rate, rate_err, intercept, residuals)` (unwraps, least
  squares, rate in rad/day); `rate_arcsec_per_century(rate_rad_per_day)`;
  `gr_precession_arcsec_per_century(a, e, gm, c=C_AU_DAY, beta=1, gamma=1)` analytic
  `(2+2gamma-beta)/3 * 6 pi gm / (c^2 a (1-e^2))` per orbit converted with the Kepler period.
- `conserved.py`: `energy(pos, vel, gm)`, `angular_momentum(pos, vel, gm)` (both over (T,N,3) or
  (N,3)), `relative_energy_error(traj)`.
- `compare.py`: `separation(trajA, trajB, body)`, `perihelion_rate_difference(trajA, trajB, body,
  central="Sun") -> (rate_arcsec_cy, err)` (differences the two varpi series then fits).

## Metric (`gravsim/metric/`)
- `schwarzschild.py` (gm and c explicit; r in AU; per unit mass):
  `schwarzschild_radius(gm, c)=2gm/c^2`, `photon_sphere_radius=3gm/c^2`, `isco_radius=6gm/c^2`,
  `effective_potential(r, L, gm, c) = -gm/r + L^2/(2 r^2) - gm L^2/(c^2 r^3)`,
  `newton_effective_potential(r, L, gm)`,
  `binet_orbit(r0, v0, gm, c, n_orbits, n_per_orbit=2000) -> (phi, r, t)` integrating
  `u'' + u = gm/L^2 + 3 gm u^2/c^2` for a planar orbit given 2D `r0, v0`,
  `geodesic_orbit(r0, v0, gm, c, tau_end, ...)` integrating the timelike geodesic in proper time
  with conserved E and L, returning `(tau, t, r, phi)`,
  `precession_per_orbit_numeric(a, e, gm, c, n_orbits=50)` from the Binet orbit (rad/orbit),
  `precession_per_orbit_1pn(a, e, gm, c) = 6 pi gm / (c^2 a (1 - e^2))`.
- `null.py` (Phase 6): photon geodesics, `deflection_angle(b, gm, c)`, Shapiro delay.

## Extra forces (`gravsim/forces/extras.py`, Phase 8 "what competes with GR")
Corrections only (add to Newton/1PN/EIH with `combine`), applied to every body `i != central` from the
central body's position; the central body gets the reaction `-sum_i gm_i a_i / gm_central` so momentum
is conserved. `axis` is the unit spin vector of the central body (default `constants.SUN_POLE_ECLIPTIC`).
- `make_j2(j2=SUN_J2, r_eq=R_SUN_AU, central=0, axis=SUN_POLE_ECLIPTIC) -> AccelFn`: quadrupole field
  `a = (3 j2 gm_c r_eq^2 / (2 r^5)) * [ (5 (r.n)^2 / r^2 - 1) r - 2 (r.n) n ]` with `n = axis`.
  Secular rates (Kaula/Gauss, inclination `i` relative to the central body's equator, p = a(1-e^2)):
  `dOmega/dt = -(3/2) n_mean j2 (r_eq/p)^2 cos i`, `domega/dt = (3/4) n_mean j2 (r_eq/p)^2 (5 cos^2 i - 1)`;
  Mercury: dvarpi/dt ~ +0.028 "/cy for j2 = 2.2e-7.
- `make_lense_thirring(gj=GJ_SUN, c=C_AU_DAY, central=0, axis=SUN_POLE_ECLIPTIC, scale=1.0) -> AccelFn`:
  gravitomagnetic (frame-dragging) acceleration of a test body, `J = gj * scale * axis` carried as G*J
  in AU^5/day^3: `a = (2 / (c^2 r^3)) * [ (3 / r^2) (r . GJ) (r x v) + (v x GJ) ]`.
  Secular rates: `dOmega/dt = 2 GJ / (c^2 a^3 (1-e^2)^1.5)`, `domega/dt = -6 GJ cos i / (c^2 a^3 (1-e^2)^1.5)`
  (i relative to the spin equator). Mercury: ~ -0.002 "/cy; `scale` exaggerates J for lessons.
- Rates of the node and argument are measured with `analysis.precession.element_series(traj, body,
  key, central="Sun") -> (t, values)` for any key of `kepler.state_to_elements` (Phase 8 addition;
  angles unwrapped by the caller via `fit_secular_rate`). Inclination relative to the solar equator:
  `analysis.elements.inclination_to_axis(r, v, axis)`.

## Web front end (`gravsim/web/` + `web/`, Phase 8)
Physics stays in Python; the browser only renders. Three.js (pinned `three@0.160.0` from
cdn.jsdelivr.net via an import map, no build step) in `web/index.html`, `web/app.js`, `web/style.css`.
- `gravsim/web/scenarios.py` (pure, no FastAPI): `list_scenarios() -> list[dict]` (id, title, blurb,
  params with defaults/ranges) and `build_scenario(scenario_id, **params) -> dict` returning
  `{"meta": {...}, "arrays": {name: float32 ndarray}}`. `meta` holds id, title, names, units
  ("AU", "day"), frame, t0, n_samples, per-model notes and the precomputed numbers to display (fitted
  rates, analytic predictions). Array names: `t` (T,), `pos_newton` (T,N,3), `pos_gr` (T,N,3) for the
  N-body scenarios; `peri_idx_newton`/`peri_idx_gr` (K,) sample indices of Mercury's perihelia;
  scenario-specific arrays (`ray_<k>` polylines, `geodesic` (T,3)) documented in `meta["arrays"]`.
- `gravsim/web/payload.py`: `encode(payload) -> bytes` = 4-byte little-endian uint32 header length +
  UTF-8 JSON header (`meta` plus an `index` of {name, dtype:"float32"|"int32", shape, offset, nbytes})
  + concatenated little-endian arrays; `decode(bytes) -> payload`. The JS loader mirrors `decode`.
- `gravsim/web/server.py`: `create_app() -> FastAPI` serving `web/` at `/` (index.html at `/`),
  `GET /api/health`, `GET /api/scenarios` (JSON list; each entry has `url` = `/api/scenario/<id>`),
  `GET /api/scenario/{id}?<params>` (`application/octet-stream`, in-process cache keyed by id+params).
  `python -m gravsim.web [--port 8000]` runs uvicorn.
- `scripts/export_web.py [--out web/data]` writes `manifest.json` (same entries, `url` = `data/<id>.bin`)
  and the `.bin` files for the default params so `web/` works from any static server
  (`python -m http.server -d web`). The front end fetches `data/manifest.json`, falls back to
  `/api/scenarios`, and decodes every scenario through one code path.
- Front end exposes `window.__gravsim = {ready, scenario, loadScenario(id), setTime(frac),
  setAmplification(k), stats: {frames}}` for browser tests. Scenarios: `solar_system` (100 yr, Newton
  vs EIH, amplification slider, Mercury perihelion trail, Sun-wobble/barycentric toggle, camera
  presets), `rosette` (Sun + Mercury at c/100), `black_hole` (zoom-whirl geodesic from
  `metric.schwarzschild.geodesic_orbit`, coordinates in GM/c^2), `light_rays` (null-geodesic bundle
  from `metric.null.ray_bundle`), `earth_moon` (Sun + Earth + Moon + planets, 2 yr).
- Browser tests (`tests/test_web.py`, marker `web`, skipped without playwright/chromium): start the
  server on a free port, load the page headlessly, require no console errors, `__gravsim.ready`, a
  rendered frame count > 0 after each `loadScenario`, and save `figures/web_<id>.png`.

## Data (`gravsim/data/`)
- `horizons.py`: `fetch_state(epoch_jd, names) -> SystemState`, `load_j2000(with_moon=False) ->
  SystemState` (`with_moon=True` gives `DEFAULT_BODIES_MOON`: Earth and Moon separate),
  `load_reference_series() -> Trajectory` (100 yr, 20-day), `load_moon_reference_series()` (10 yr,
  daily, Earth + Moon), all cached with no network; `fetch_series(...)` (network).
- Tests must never hit the network; use the cached files. Network tests are marked `@pytest.mark.network`.

## Testing
- `tests/test_<module>.py` per module; validation targets in `tests/test_validation.py`.
- Markers: `slow`, `network`, `web` (all deselected by default).
- Fast suite must run in < 90 s serially (about 60 s today; `-n auto` with pytest-xdist for a quick
  parallel run); mark century-long runs `@pytest.mark.slow`.
- Validation tolerances (ROADMAP §4): Mercury 2-body 1PN 42.98 ± 0.1 "/cy; Newton 2-body noise floor
  < 0.5 "/cy; N-body Newton Mercury 531.6 ± 3 "/cy (absolute 100-yr fits are biased ~−2.4 by
  long-period terms, so GR checks are run-vs-run differences); Schwarzschild exact vs 1PN < 0.01 "/cy.
- Prefer analytic references (Kepler, harmonic oscillator, known formulas) over snapshot tests.

## Style
- Type hints, numpy-style docstrings with the physics (formula + reference) at the top of each
  module. No print statements in library code. Keep lessons' narrative in notebooks, not modules.
