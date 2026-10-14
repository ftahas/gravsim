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
  system barycenters). Body 0 is the Sun; code takes the central body as an explicit argument
  (`central=0` or `central="Sun"`) rather than assuming it.
- Arrays are float64, C-contiguous; positions/velocities shape (N, 3); time series (T, N, 3).
- Functions are pure: never mutate input arrays; return new arrays.
- A body with `gm == 0` is a test particle: it feels forces but exerts none.

## Core types (`gravsim/bodies.py`)
- `SystemState(t, pos, vel, gm, names)` frozen dataclass; `.index(name)`, `.with_(...)`,
  `.relative_to(name)`, `.to_barycentric()`, `.subset(names)`.
- `Trajectory(t, pos, vel, gm, names, meta)`; `.body(name)`, `.relative(name, central)`,
  `.state_at(i)`, `.final_state`, `Trajectory.from_states(list)`.
  `meta` records integrator, dt, force law, c, etc. (free-form dict).

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

## Data (`gravsim/data/`)
- `horizons.py`: `fetch_state(epoch_jd, names) -> SystemState`, `load_j2000() -> SystemState`,
  `load_reference_series() -> Trajectory` (cached, no network), `fetch_series(...)` (network).
- Tests must never hit the network; use the cached files. Network tests are marked `@pytest.mark.network`.

## Testing
- `tests/test_<module>.py` per module; validation targets in `tests/test_validation.py`.
- Fast suite must run in < 60 s; mark century-long runs `@pytest.mark.slow`.
- Validation tolerances (ROADMAP §4): Mercury 2-body 1PN 42.98 ± 0.1 "/cy; Newton 2-body noise floor
  < 0.5 "/cy; N-body Newton Mercury 531.6 ± 3 "/cy; Schwarzschild exact vs 1PN < 0.01 "/cy.
- Prefer analytic references (Kepler, harmonic oscillator, known formulas) over snapshot tests.

## Style
- Type hints, numpy-style docstrings with the physics (formula + reference) at the top of each
  module. No print statements in library code. Keep lessons' narrative in notebooks, not modules.
