"""Tests for ``gravsim.integrators``: fixed-step schemes, the driver and the scipy wrapper.

All references are analytic (harmonic oscillator, Kepler orbits via a local Kepler-equation
solver, linear drag, a forced oscillator) and the force laws are defined here:
``gravsim.forces`` is written in parallel and must not be imported.  Nothing touches the
network.

Two families of tests exist only to pin down the *arguments* that reach the force law, which
a time-independent, gm-ignoring force cannot detect:

* ``forced_harmonic`` reads the absolute time ``t`` (``x'' = -x + cos 2t``), so a wrong stage
  time inside an integrator, a driver passing ``t0`` instead of ``t_i`` or a scipy wrapper
  passing the wrong ``t`` all change the answer; the runs start at ``t0 != 0`` and go in both
  time directions.
* ``pairwise_newton`` reads ``gm`` (Sun + Mercury, both massive), so a driver that drops or
  zeroes ``gm`` turns the orbit into straight lines.  ``SpyForce`` additionally records the
  ``(t, gm)`` of every call.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import ARCSEC_PER_RAD, GM, GM_SUN, JULIAN_CENTURY_DAYS
from gravsim.integrators import (
    INTEGRATORS,
    RK4,
    AccelFn,
    Euler,
    Integrator,
    Leapfrog,
    SymplecticEuler,
    Yoshida4,
    Yoshida6,
    drift,
    get_integrator,
    implicit_kick,
    integrate,
    integrate_scipy,
    kick,
    step_grid,
)

# ----------------------------------------------------------------------------- force laws


def make_central(gm_c: float) -> AccelFn:
    """Every body attracted to a fixed mass ``gm_c`` at the origin (``gm`` is ignored)."""

    def accel(pos, vel, gm, t):
        r = np.linalg.norm(pos, axis=1)[:, None]
        return -gm_c * pos / r**3

    return accel


def harmonic(pos, vel, gm, t):
    """Unit-frequency harmonic oscillator ``x'' = -x`` (component-wise)."""
    return -pos


def make_drag(k: float) -> AccelFn:
    """Linear drag ``a = -k v``: exact ``v = v0 e^{-kt}``, ``x = x0 + v0 (1 - e^{-kt}) / k``."""

    def accel(pos, vel, gm, t):
        return -k * vel

    return accel


def make_gyro_kepler(gm_c: float, b: float) -> AccelFn:
    """Kepler force plus a magnetic-type term ``v x B`` with ``B = (0, 0, b)``.

    The gyroscopic term does no work, so ``E = v^2/2 - gm_c/r`` is conserved exactly.
    """
    bvec = np.array([0.0, 0.0, b])

    def accel(pos, vel, gm, t):
        r = np.linalg.norm(pos, axis=1)[:, None]
        return -gm_c * pos / r**3 + np.cross(vel, bvec)

    return accel


E_X = np.array([1.0, 0.0, 0.0])


def forced_harmonic(pos, vel, gm, t):
    """Forced oscillator ``x'' = -x + cos(2t)`` along x: the force reads the *absolute* time.

    Exact solution with ``x(0) = v(0) = 0``: ``x = (cos t - cos 2t) / 3``.
    """
    return -pos + math.cos(2.0 * t) * E_X


def pairwise_newton(pos, vel, gm, t):
    """Pairwise Newtonian acceleration ``a_i = sum_j gm_j (r_j - r_i) / |r_j - r_i|^3``.

    Reads ``gm`` from the array (a body with ``gm == 0`` feels forces but exerts none) and
    conserves the total momentum ``sum_i gm_i v_i`` exactly (action = reaction).
    """
    d = pos[None, :, :] - pos[:, None, :]  # d[i, j] = r_j - r_i
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    inv_r3 = r2**-1.5
    return np.sum(gm[None, :, None] * d * inv_r3[:, :, None], axis=1)


class SpyForce:
    """Wrap a force law and record the ``t`` and ``gm`` of every call, in call order."""

    def __init__(self, inner: AccelFn) -> None:
        self.inner = inner
        self.times: list[float] = []
        self.gms: list[np.ndarray] = []

    def __call__(self, pos, vel, gm, t):
        self.times.append(float(t))
        self.gms.append(gm)
        return self.inner(pos, vel, gm, t)


# ----------------------------------------------------------------------------- analytic references


def forced_exact(t) -> tuple[np.ndarray, np.ndarray]:
    """``x(t), v(t)`` of ``forced_harmonic`` for the solution ``x = (cos t - cos 2t) / 3``."""
    t = np.asarray(t, dtype=float)
    return (np.cos(t) - np.cos(2.0 * t)) / 3.0, (-np.sin(t) + 2.0 * np.sin(2.0 * t)) / 3.0


def forced_state(t0: float) -> SystemState:
    """State on the exact forced-oscillator solution at absolute time ``t0``."""
    x, v = forced_exact(t0)
    return SystemState(t0, [[float(x), 0.0, 0.0]], [[float(v), 0.0, 0.0]], [0.0])


def two_body_state(a: float, e: float, gm1: float, gm2: float, t0: float = 0.0) -> SystemState:
    """Two massive bodies at perihelion of their relative Kepler orbit, barycentre at rest.

    The relative orbit has ``mu = gm1 + gm2``; body 1 sits at ``-gm2/mu`` and body 2 at
    ``+gm1/mu`` times the relative vector, so the barycentre is at the origin.
    """
    mu = gm1 + gm2
    r = a * (1.0 - e)
    v = math.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    rel_p = np.array([r, 0.0, 0.0])
    rel_v = np.array([0.0, v, 0.0])
    pos = np.stack([-gm2 / mu * rel_p, gm1 / mu * rel_p])
    vel = np.stack([-gm2 / mu * rel_v, gm1 / mu * rel_v])
    return SystemState(t0, pos, vel, [gm1, gm2], ("Sun", "Mercury"))


def lrl_longitude(pos: np.ndarray, vel: np.ndarray, gm_c: float) -> np.ndarray:
    """Unwrapped longitude of the Laplace-Runge-Lenz vector of a planar orbit, shape (T,)."""
    h = np.cross(pos, vel)
    r = np.linalg.norm(pos, axis=1)[:, None]
    lrl = np.cross(vel, h) / gm_c - pos / r
    return np.unwrap(np.arctan2(lrl[:, 1], lrl[:, 0]))


def perihelion_drift_arcsec_per_century(t: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                                        gm_c: float) -> float:
    """Least-squares slope of the LRL longitude, converted to arcsec per Julian century."""
    slope = np.polyfit(t, lrl_longitude(pos, vel, gm_c), 1)[0]  # rad/day
    return float(slope * JULIAN_CENTURY_DAYS * ARCSEC_PER_RAD)


def kepler_period(a: float, gm_c: float) -> float:
    return 2.0 * math.pi * math.sqrt(a**3 / gm_c)


def kepler_state(a: float, e: float, gm_c: float, t: float = 0.0) -> SystemState:
    """Test particle at perihelion (on +x, moving along +y) of a planar Kepler orbit."""
    r = a * (1.0 - e)
    v = math.sqrt(gm_c * (1.0 + e) / (a * (1.0 - e)))
    return SystemState(t, [[r, 0.0, 0.0]], [[0.0, v, 0.0]], [0.0], ("particle",))


def kepler_exact(a: float, e: float, gm_c: float, t) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity (T, 3) of the orbit of ``kepler_state`` at times ``t``."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = math.sqrt(gm_c / a**3)
    mean_anom = n * t
    ecc_anom = mean_anom.copy()
    for _ in range(60):  # Newton on Kepler's equation E - e sin E = M
        ecc_anom = ecc_anom - (ecc_anom - e * np.sin(ecc_anom) - mean_anom) / (
            1.0 - e * np.cos(ecc_anom)
        )
    c, s = np.cos(ecc_anom), np.sin(ecc_anom)
    b = a * math.sqrt(1.0 - e**2)
    edot = n / (1.0 - e * c)
    zeros = np.zeros_like(t)
    pos = np.stack([a * (c - e), b * s, zeros], axis=1)
    vel = np.stack([-a * s * edot, b * c * edot, zeros], axis=1)
    return pos, vel


def rel_energy_error(traj: Trajectory, gm_c: float) -> np.ndarray:
    """|(E(t) - E0) / E0| for a test particle in the field of ``gm_c`` at the origin."""
    r = np.linalg.norm(traj.pos[:, 0], axis=1)
    v2 = np.sum(traj.vel[:, 0] ** 2, axis=1)
    energy = 0.5 * v2 - gm_c / r
    return np.abs((energy - energy[0]) / energy[0])


def state_error(traj: Trajectory, pos_ref: np.ndarray, vel_ref: np.ndarray) -> float:
    """Euclidean error of the final (pos, vel) of body 0 against a reference."""
    dp = np.linalg.norm(traj.pos[-1, 0] - pos_ref)
    dv = np.linalg.norm(traj.vel[-1, 0] - vel_ref)
    return math.hypot(dp, dv)


def measured_order(errors: tuple[float, float]) -> float:
    """Richardson-style order estimate from the errors at dt and dt/2."""
    return math.log2(errors[0] / errors[1])


ALL_NAMES = sorted(INTEGRATORS)
SYMPLECTIC_NAMES = sorted(n for n in ALL_NAMES if INTEGRATORS[n].symplectic)


def all_instances() -> list[Integrator]:
    out: list[Integrator] = []
    for cls in INTEGRATORS.values():
        out.append(cls())
        if cls.symplectic:
            out.append(cls(vel_dependent=True))
    return out


# ----------------------------------------------------------------------------- protocol / registry


def test_registry_matches_contract():
    assert set(INTEGRATORS) == {
        "euler",
        "symplectic_euler",
        "leapfrog",
        "rk4",
        "yoshida4",
        "yoshida6",
    }
    for name, cls in INTEGRATORS.items():
        assert name == name.lower() == cls.name
        assert issubclass(cls, Integrator)
    assert INTEGRATORS["euler"] is Euler and INTEGRATORS["rk4"] is RK4
    assert INTEGRATORS["symplectic_euler"] is SymplecticEuler
    assert INTEGRATORS["leapfrog"] is Leapfrog
    assert INTEGRATORS["yoshida4"] is Yoshida4 and INTEGRATORS["yoshida6"] is Yoshida6
    orders = {n: INTEGRATORS[n].order for n in INTEGRATORS}
    assert orders == {
        "euler": 1,
        "symplectic_euler": 1,
        "leapfrog": 2,
        "rk4": 4,
        "yoshida4": 4,
        "yoshida6": 6,
    }
    assert SYMPLECTIC_NAMES == ["leapfrog", "symplectic_euler", "yoshida4", "yoshida6"]
    assert not Euler.symplectic and not RK4.symplectic


def test_yoshida_weights_sum_to_one():
    assert math.isclose(sum(Yoshida4.weights), 1.0, abs_tol=1e-14)
    assert math.isclose(sum(Yoshida6.weights), 1.0, abs_tol=1e-14)
    # the triple-jump order conditions
    assert math.isclose(sum(w**3 for w in Yoshida4.weights), 0.0, abs_tol=1e-13)
    assert math.isclose(sum(w**3 for w in Yoshida6.weights), 0.0, abs_tol=1e-12)
    assert math.isclose(sum(w**5 for w in Yoshida6.weights), 0.0, abs_tol=1e-12)
    assert Yoshida6.weights == Yoshida6.weights[::-1]  # symmetric composition


def test_get_integrator_and_abstract_base():
    assert isinstance(get_integrator("leapfrog"), Leapfrog)
    assert isinstance(get_integrator("YoShIdA4"), Yoshida4)
    lf = Leapfrog(vel_dependent=True, n_iter=3)
    assert get_integrator(lf) is lf
    vd = get_integrator("leapfrog", vel_dependent=True, n_iter=5)
    assert vd.vel_dependent and vd.n_iter == 5
    with pytest.raises(KeyError):
        get_integrator("ias15")
    with pytest.raises(TypeError):
        get_integrator(lf, vel_dependent=True)
    with pytest.raises(TypeError):
        Integrator()  # abstract
    with pytest.raises(ValueError):
        Leapfrog(n_iter=0)
    assert repr(lf) == "Leapfrog(vel_dependent=True, n_iter=3)"
    assert repr(RK4()) == "RK4()"


@pytest.mark.parametrize("integ", all_instances(), ids=repr)
def test_step_is_pure_and_returns_new_arrays(integ):
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vel = np.array([[0.0, 1.0, 0.0], [-0.5, 0.0, 0.1]])
    gm = np.array([0.0, 0.0])
    pos0, vel0 = pos.copy(), vel.copy()
    pos_new, vel_new = integ.step(pos, vel, gm, 0.3, 0.05, make_central(1.0))
    assert np.array_equal(pos, pos0) and np.array_equal(vel, vel0)
    assert pos_new is not pos and vel_new is not vel
    assert pos_new.shape == (2, 3) and vel_new.shape == (2, 3)
    assert pos_new.dtype == np.float64 and vel_new.dtype == np.float64
    assert not np.shares_memory(pos_new, pos) and not np.shares_memory(vel_new, vel)
    assert np.all(np.isfinite(pos_new)) and np.all(np.isfinite(vel_new))


def test_kick_and_drift_helpers():
    pos = np.array([[1.0, 2.0, 3.0]])
    vel = np.array([[0.1, -0.2, 0.3]])
    acc = np.array([[1.0, 1.0, -1.0]])
    assert np.array_equal(drift(pos, vel, 0.5), pos + 0.5 * vel)
    assert np.array_equal(kick(vel, acc, 0.5), vel + 0.5 * acc)
    gm = np.zeros(1)
    # velocity-independent force: the implicit kick equals the explicit one after 1 iteration
    central = make_central(1.0)
    explicit = kick(vel, central(pos, vel, gm, 0.0), 0.2)
    assert np.allclose(implicit_kick(pos, vel, gm, 0.0, 0.2, central, n_iter=1), explicit)
    assert np.allclose(implicit_kick(pos, vel, gm, 0.0, 0.2, central, n_iter=4), explicit)
    # linear drag: fixed point is v' = v / (1 + k dt); the iterates converge to it
    # geometrically with contraction factor k dt = 0.21 per iteration
    k, dt = 0.7, 0.3
    drag = make_drag(k)
    exact = vel / (1.0 + k * dt)
    errs = [
        np.abs(implicit_kick(pos, vel, gm, 0.0, dt, drag, n_iter=n) - exact).max()
        for n in (1, 2, 4, 16)
    ]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[1] < 0.3 * errs[0] and errs[2] < 0.3 * errs[1]
    assert errs[3] < 1e-9
    with pytest.raises(ValueError):
        implicit_kick(pos, vel, gm, 0.0, dt, drag, n_iter=0)


# ----------------------------------------------------------------------------- (1) oscillator

HARMONIC_TOL = {
    "euler": 2e-2,
    "symplectic_euler": 5e-3,
    "leapfrog": 1e-5,
    "rk4": 1e-10,
    "yoshida4": 1e-9,
    "yoshida6": 1e-12,
}


@pytest.mark.parametrize("name", ALL_NAMES)
def test_harmonic_oscillator_exact(name):
    """x'' = -x with x(0)=1, v(0)=0 over one period: x = cos t, v = -sin t."""
    state = SystemState(0.0, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [0.0])
    period = 2.0 * math.pi
    traj = integrate(state, harmonic, period, period / 2000, INTEGRATORS[name]())
    x, v = traj.pos[:, 0, 0], traj.vel[:, 0, 0]
    err = max(np.abs(x - np.cos(traj.t)).max(), np.abs(v + np.sin(traj.t)).max())
    assert err < HARMONIC_TOL[name], f"{name}: max error {err:.3e}"
    # the other components stay exactly zero
    assert np.all(traj.pos[:, 0, 1:] == 0.0) and np.all(traj.vel[:, 0, 1:] == 0.0)


def test_harmonic_oscillator_scipy():
    state = SystemState(0.0, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [0.0])
    t_eval = np.linspace(0.0, 4.0 * math.pi, 101)
    traj = integrate_scipy(state, harmonic, t_eval)
    assert np.abs(traj.pos[:, 0, 0] - np.cos(t_eval)).max() < 1e-10
    assert np.abs(traj.vel[:, 0, 0] + np.sin(t_eval)).max() < 1e-10


# ----------------------------------------------------------------------------- (1b) forced
#
# x'' = -x + cos(2t) with the exact solution x = (cos t - cos 2t)/3.  The forcing reads the
# absolute time, the runs start at t0 = 5 and go forwards *and* backwards, so a stage
# evaluated at the wrong time (RK4 k2..k4 at t, a Yoshida substep that does not advance tau,
# a driver that hands the force t0 instead of t_i) changes the answer and the measured order.

FORCED_T0 = 5.0
FORCED_SPAN = 1.37 * 2.0 * math.pi

# name -> (steps at dt, steps at dt/2, nominal order, error bound at dt/2)
FORCED_PLAN = {
    "euler": (2000, 4000, 1, 1e-2),
    "symplectic_euler": (2000, 4000, 1, 5e-3),
    "leapfrog": (500, 1000, 2, 5e-5),
    "rk4": (200, 400, 4, 3e-8),
    "yoshida4": (100, 200, 4, 4e-6),
    "yoshida6": (50, 100, 6, 4e-8),
}


def forced_error(integ: Integrator, t_end: float, n: int) -> float:
    traj = integrate(forced_state(FORCED_T0), forced_harmonic, t_end, FORCED_SPAN / n, integ)
    assert traj.meta["n_steps"] == n
    x_ref, v_ref = forced_exact(traj.t)
    assert np.all(traj.pos[:, 0, 1:] == 0.0) and np.all(traj.vel[:, 0, 1:] == 0.0)
    return max(np.abs(traj.pos[:, 0, 0] - x_ref).max(), np.abs(traj.vel[:, 0, 0] - v_ref).max())


@pytest.mark.parametrize("direction", [+1, -1], ids=["forward", "backward"])
@pytest.mark.parametrize("integ", all_instances(), ids=repr)
def test_forced_oscillator_absolute_time_order(integ, direction):
    n1, n2, expected, bound = FORCED_PLAN[integ.name]
    t_end = FORCED_T0 + direction * FORCED_SPAN
    errors = (forced_error(integ, t_end, n1), forced_error(integ, t_end, n2))
    assert errors[1] < bound, f"{integ!r}: error {errors[1]:.3e} at {n2} steps"
    p = measured_order(errors)
    assert abs(p - expected) < 0.3, f"{integ!r}: measured order {p:.2f}, errors {errors}"


@pytest.mark.parametrize("direction", [+1, -1], ids=["forward", "backward"])
def test_forced_oscillator_scipy_nonzero_t0(direction):
    t_end = FORCED_T0 + direction * 2.0 * FORCED_SPAN
    t_eval = np.linspace(FORCED_T0, t_end, 101)
    traj = integrate_scipy(forced_state(FORCED_T0), forced_harmonic, t_eval)
    x_ref, v_ref = forced_exact(t_eval)
    assert np.abs(traj.pos[:, 0, 0] - x_ref).max() < 1e-10
    assert np.abs(traj.vel[:, 0, 0] - v_ref).max() < 1e-10


# ----------------------------------------------------------------------------- (2) order

# (name, steps at dt, steps at dt/2, nominal order); the error is measured at a generic time
# (1.37 periods) so that a whole number of periods does not hide first-order terms.
ORDER_PLAN = [
    ("euler", 2000, 4000, 1),
    ("symplectic_euler", 2000, 4000, 1),
    ("leapfrog", 500, 1000, 2),
    ("rk4", 200, 400, 4),
    ("yoshida4", 100, 200, 4),
    ("yoshida6", 50, 100, 6),
]


@pytest.mark.parametrize(("name", "n1", "n2", "expected"), ORDER_PLAN)
def test_convergence_order(name, n1, n2, expected):
    a, e, gm_c = 1.0, 0.2, 1.0
    t_end = 1.37 * kepler_period(a, gm_c)
    state = kepler_state(a, e, gm_c)
    pos_ref, vel_ref = kepler_exact(a, e, gm_c, t_end)
    errors = []
    for n in (n1, n2):
        traj = integrate(state, make_central(gm_c), t_end, t_end / n, INTEGRATORS[name]())
        assert traj.meta["n_steps"] == n
        errors.append(state_error(traj, pos_ref[0], vel_ref[0]))
    p = measured_order((errors[0], errors[1]))
    assert abs(p - expected) < 0.3, f"{name}: measured order {p:.2f}, errors {errors}"


# ----------------------------------------------------------------------------- (3) Kepler energy

KEPLER_A, KEPLER_E = 0.4, 0.2  # AU; period ~92 d around the Sun
KEPLER_P = kepler_period(KEPLER_A, GM_SUN)
N_ORBITS = 200


@pytest.mark.parametrize(
    ("name", "steps_per_orbit", "save_every", "bound"),
    [
        ("symplectic_euler", 200, 2, 2e-2),
        ("leapfrog", 64, 1, 5e-3),
        ("yoshida4", 32, 1, 2e-3),
        ("yoshida6", 32, 1, 2e-5),
    ],
)
def test_symplectic_energy_bounded_over_200_orbits(name, steps_per_orbit, save_every, bound):
    state = kepler_state(KEPLER_A, KEPLER_E, GM_SUN)
    traj = integrate(
        state,
        make_central(GM_SUN),
        N_ORBITS * KEPLER_P,
        KEPLER_P / steps_per_orbit,
        INTEGRATORS[name](),
        save_every=save_every,
    )
    assert traj.meta["n_steps"] == N_ORBITS * steps_per_orbit
    d_e = rel_energy_error(traj, GM_SUN)
    per_orbit = steps_per_orbit // save_every
    early = d_e[: 20 * per_orbit].max()  # first 20 orbits
    late = d_e[-20 * per_orbit :].max()  # last 20 orbits
    assert d_e.max() < bound, f"{name}: max |dE/E| = {d_e.max():.3e}"
    assert late < 1.2 * early, f"{name}: energy error grows {early:.3e} -> {late:.3e}"


def test_rk4_energy_drifts_monotonically_at_coarse_step():
    steps_per_orbit = 32
    state = kepler_state(KEPLER_A, KEPLER_E, GM_SUN)
    traj = integrate(
        state,
        make_central(GM_SUN),
        N_ORBITS * KEPLER_P,
        KEPLER_P / steps_per_orbit,
        RK4(),
        save_every=steps_per_orbit,  # one sample per orbit, same orbital phase
    )
    r = np.linalg.norm(traj.pos[:, 0], axis=1)
    energy = 0.5 * np.sum(traj.vel[:, 0] ** 2, axis=1) - GM_SUN / r
    d_e = (energy - energy[0]) / abs(energy[0])
    assert np.all(np.diff(d_e[1:]) < 0.0), "RK4 energy error is not monotonic"
    assert abs(d_e[-1]) > 5.0 * abs(d_e[20]), "RK4 energy error does not grow secularly"
    assert abs(d_e[-1]) > 1e-3  # far above any symplectic scheme at this cost


# ----------------------------------------------------------------------------- (4) circular orbit


@pytest.mark.parametrize(
    ("integ", "steps_per_orbit", "tol"),
    [(Leapfrog(), 1000, 3e-4), (Yoshida4(), 200, 3e-5), (Yoshida6(), 100, 2e-7)],
    ids=repr,
)
def test_circular_orbit_matches_analytic(integ, steps_per_orbit, tol):
    r0 = 0.5
    v_c = math.sqrt(GM_SUN / r0)
    omega = v_c / r0
    period = 2.0 * math.pi / omega
    state = SystemState(0.0, [[r0, 0.0, 0.0]], [[0.0, v_c, 0.0]], [0.0])
    traj = integrate(state, make_central(GM_SUN), 2.0 * period, period / steps_per_orbit, integ)
    exact = r0 * np.stack(
        [np.cos(omega * traj.t), np.sin(omega * traj.t), np.zeros_like(traj.t)], axis=1
    )
    err = np.linalg.norm(traj.pos[:, 0] - exact, axis=1).max()
    assert err < tol, f"{integ!r}: max position error {err:.3e} AU"
    assert np.abs(np.linalg.norm(traj.pos[:, 0], axis=1) - r0).max() < tol


def test_circular_orbit_scipy():
    r0 = 0.5
    v_c = math.sqrt(GM_SUN / r0)
    omega = v_c / r0
    period = 2.0 * math.pi / omega
    state = SystemState(0.0, [[r0, 0.0, 0.0]], [[0.0, v_c, 0.0]], [0.0])
    t_eval = np.linspace(0.0, 2.0 * period, 201)
    traj = integrate_scipy(state, make_central(GM_SUN), t_eval)
    exact = r0 * np.stack([np.cos(omega * t_eval), np.sin(omega * t_eval), 0 * t_eval], axis=1)
    assert np.linalg.norm(traj.pos[:, 0] - exact, axis=1).max() < 1e-10


# ----------------------------------------------------------------------------- (5) driver


@pytest.mark.parametrize(
    ("t0", "t_end", "dt", "save_every", "n_expected", "dt_expected"),
    [
        (0.0, 10.0, 3.0, 1, 4, 2.5),  # shrink minimally: 3 -> 2.5
        (0.0, 10.0, 2.5, 1, 4, 2.5),  # exact multiple: unchanged
        (0.0, 10.0, 10.0 / 3.0, 1, 3, 10.0 / 3.0),  # exact up to roundoff: no extra step
        (0.0, 0.3, 0.1, 1, 3, 0.1),  # 0.3/0.1 = 2.9999...: still 3 steps
        (0.0, 100.0, 1.0, 7, 105, 100.0 / 105.0),  # rounded up to a multiple of save_every
        (0.0, -10.0, 3.0, 1, 4, -2.5),  # backwards
        (5.0, 2.0, 1.0, 1, 3, -1.0),  # backwards from a non-zero start
        (0.0, 10.0, 100.0, 1, 1, 10.0),  # dt larger than the span: one step
        (3.0, 3.0, 1.0, 1, 0, 0.0),  # nothing to do
    ],
)
def test_step_grid(t0, t_end, dt, save_every, n_expected, dt_expected):
    n, dt_actual = step_grid(t0, t_end, dt, save_every)
    assert n == n_expected
    assert math.isclose(dt_actual, dt_expected, rel_tol=1e-12, abs_tol=0.0)
    if n:
        assert abs(dt_actual) <= abs(dt) * (1 + 1e-12)
        assert t0 + n * dt_actual == pytest.approx(t_end, abs=1e-12)


def test_step_grid_rejects_bad_input():
    with pytest.raises(ValueError):
        step_grid(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        step_grid(0.0, 1.0, math.nan)
    with pytest.raises(ValueError):
        step_grid(0.0, 1.0, 0.1, save_every=0)


def test_driver_lands_exactly_on_t_end_and_records_meta():
    state = kepler_state(0.7, 0.1, GM_SUN, t=12.5)
    pos0, vel0 = state.pos.copy(), state.vel.copy()
    t_end = 12.5 + 100.0
    traj = integrate(state, make_central(GM_SUN), t_end, 0.3, Leapfrog())
    assert traj.t[-1] == t_end  # exact, not approximately
    assert traj.t[0] == state.t
    assert traj.meta["n_steps"] == 334 and traj.n_steps == 335
    assert traj.meta["dt"] == pytest.approx(100.0 / 334)
    assert traj.meta["dt"] <= 0.3
    assert traj.meta["dt_requested"] == 0.3
    assert traj.meta["integrator"] == "leapfrog"
    assert traj.meta["order"] == 2 and traj.meta["symplectic"] is True
    assert traj.meta["save_every"] == 1 and traj.meta["t_end"] == t_end
    assert traj.meta["vel_dependent"] is False and traj.meta["n_iter"] == 2
    assert np.allclose(np.diff(traj.t), traj.meta["dt"], rtol=0, atol=1e-12)
    # sample 0 is the initial state and the input was not modified
    assert np.array_equal(traj.pos[0], pos0) and np.array_equal(traj.vel[0], vel0)
    assert np.array_equal(state.pos, pos0) and np.array_equal(state.vel, vel0)
    assert traj.names == state.names
    assert np.array_equal(traj.gm, state.gm) and not np.shares_memory(traj.gm, state.gm)
    assert traj.pos.flags.c_contiguous and traj.pos.dtype == np.float64


def test_driver_save_every_subsamples_the_full_run():
    state = kepler_state(0.5, 0.3, GM_SUN)
    central = make_central(GM_SUN)
    full = integrate(state, central, 100.0, 1.0, Yoshida4())
    sub = integrate(state, central, 100.0, 1.0, Yoshida4(), save_every=10)
    assert full.meta["n_steps"] == sub.meta["n_steps"] == 100
    assert sub.n_steps == 11 and sub.meta["save_every"] == 10
    assert np.array_equal(sub.t, full.t[::10])
    assert np.array_equal(sub.pos, full.pos[::10])  # identical arithmetic, bitwise equal
    assert np.array_equal(sub.vel, full.vel[::10])
    assert sub.t[-1] == 100.0
    # save_every that does not divide the step count: steps are rounded up, grid stays uniform
    odd = integrate(state, central, 100.0, 1.0, Leapfrog(), save_every=7)
    assert odd.meta["n_steps"] == 105 and odd.n_steps == 16
    assert odd.t[-1] == 100.0
    assert np.allclose(np.diff(odd.t), 7 * odd.meta["dt"], rtol=0, atol=1e-12)


def test_driver_negative_time_and_leapfrog_reversibility():
    state = kepler_state(KEPLER_A, KEPLER_E, GM_SUN, t=50.0)
    central = make_central(GM_SUN)
    t_end = 50.0 - 3.0 * KEPLER_P
    back = integrate(state, central, t_end, KEPLER_P / 128, Yoshida6(), save_every=4)
    assert back.t[0] == 50.0 and back.t[-1] == t_end
    assert back.meta["dt"] < 0.0 and np.all(np.diff(back.t) < 0.0)
    assert back.meta["n_steps"] == 3 * 128 and back.n_steps == 3 * 32 + 1
    # the backward run is the exact Kepler orbit run in reverse (negative times)
    pos_ref, _ = kepler_exact(KEPLER_A, KEPLER_E, GM_SUN, back.t - 50.0)
    assert np.linalg.norm(back.pos[:, 0] - pos_ref, axis=1).max() < 1e-5
    # forward then backward returns to the start to roundoff (leapfrog is time-reversible)
    fwd = integrate(state, central, 50.0 + 10.0 * KEPLER_P, KEPLER_P / 100, Leapfrog())
    rev = integrate(fwd.final_state, central, 50.0, KEPLER_P / 100, Leapfrog())
    assert rev.t[-1] == 50.0
    assert np.abs(rev.pos[-1] - state.pos).max() < 1e-11
    assert np.abs(rev.vel[-1] - state.vel).max() < 1e-12


def test_driver_zero_span_string_integrator_and_errors():
    state = kepler_state(1.0, 0.0, GM_SUN, t=7.0)
    central = make_central(GM_SUN)
    traj = integrate(state, central, 7.0, 1.0, "rk4")
    assert traj.n_steps == 1 and traj.meta["n_steps"] == 0 and traj.meta["dt"] == 0.0
    assert np.array_equal(traj.pos[0], state.pos)
    by_name = integrate(state, central, 17.0, 1.0, "Yoshida6")
    by_cls = integrate(state, central, 17.0, 1.0, Yoshida6())
    assert by_name.meta["integrator"] == "yoshida6"
    assert np.array_equal(by_name.pos, by_cls.pos)
    with pytest.raises(KeyError):
        integrate(state, central, 17.0, 1.0, "ias15")
    with pytest.raises(ValueError):
        integrate(state, central, 17.0, 0.0, "rk4")
    with pytest.raises(ValueError):
        integrate(state, central, 17.0, 1.0, "rk4", save_every=0)


def test_driver_many_bodies_shapes():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(4, 3)) + np.array([3.0, 0.0, 0.0])
    vel = 0.01 * rng.normal(size=(4, 3))
    state = SystemState(0.0, pos, vel, [GM_SUN, 0.0, 0.0, 1e-9], ("Sun", "a", "b", "c"))
    traj = integrate(state, make_central(GM_SUN), 30.0, 0.25, Yoshida4(), save_every=4)
    assert traj.pos.shape == (31, 4, 3) and traj.vel.shape == (31, 4, 3)
    assert traj.names == ("Sun", "a", "b", "c")
    assert traj.index("b") == 2
    assert np.all(np.isfinite(traj.pos))


# ----------------------------------------------------------------------------- (6) scipy wrapper


def test_scipy_eccentric_orbit_one_period():
    a, e = 0.387, 0.2056  # Mercury-like
    period = kepler_period(a, GM_SUN)
    state = kepler_state(a, e, GM_SUN, t=3.0)
    t_eval = 3.0 + np.linspace(0.0, period, 9)
    traj = integrate_scipy(state, make_central(GM_SUN), t_eval)
    assert np.array_equal(traj.t, t_eval)
    assert np.array_equal(traj.pos[0], state.pos) and np.array_equal(traj.vel[0], state.vel)
    assert np.linalg.norm(traj.pos[-1, 0] - state.pos[0]) < 1e-10
    pos_ref, vel_ref = kepler_exact(a, e, GM_SUN, t_eval - 3.0)
    assert np.linalg.norm(traj.pos[:, 0] - pos_ref, axis=1).max() < 1e-10
    assert np.linalg.norm(traj.vel[:, 0] - vel_ref, axis=1).max() < 1e-11
    meta = traj.meta
    assert meta["integrator"] == "solve_ivp" and meta["method"] == "DOP853"
    assert meta["rtol"] == 1e-12 and meta["atol"] == 1e-15
    assert meta["nfev"] > 0 and meta["n_eval_steps"] == 9
    assert traj.names == state.names and np.array_equal(traj.gm, state.gm)


def test_scipy_backward_single_time_and_other_method():
    a, e = 0.387, 0.2056
    period = kepler_period(a, GM_SUN)
    state = kepler_state(a, e, GM_SUN)
    central = make_central(GM_SUN)
    # backwards in time: t_eval decreasing, first point strictly before state.t is fine too
    back = integrate_scipy(state, central, np.array([-0.25 * period, -period]))
    pos_ref, _ = kepler_exact(a, e, GM_SUN, back.t)
    assert np.linalg.norm(back.pos[:, 0] - pos_ref, axis=1).max() < 1e-10
    # only the initial time requested
    same = integrate_scipy(state, central, np.array([0.0]))
    assert same.n_steps == 1 and np.array_equal(same.pos[0], state.pos)
    assert same.meta["nfev"] == 0
    # another solve_ivp method with looser tolerances, extra kwargs forwarded
    rk = integrate_scipy(
        state, central, np.array([0.0, period]), method="RK45", rtol=1e-8, atol=1e-10,
        max_step=period / 50,
    )
    assert rk.meta["method"] == "RK45" and rk.meta["rtol"] == 1e-8
    assert np.linalg.norm(rk.pos[-1, 0] - state.pos[0]) < 1e-5


def kepler_run_diagnostics(traj: Trajectory, a: float, e: float) -> tuple[float, float, float]:
    """(max position error [AU], max |dE/E|, perihelion drift [arcsec/cy]) vs. Kepler."""
    pos_ref, _ = kepler_exact(a, e, GM_SUN, traj.t - traj.t[0])
    pos_err = np.linalg.norm(traj.pos[:, 0] - pos_ref, axis=1).max()
    d_e = rel_energy_error(traj, GM_SUN).max()
    drift = perihelion_drift_arcsec_per_century(traj.t, traj.pos[:, 0], traj.vel[:, 0], GM_SUN)
    return float(pos_err), float(d_e), drift


def test_scipy_kepler_100_orbits_accuracy_floor():
    """Measured DOP853 floor at the defaults (see the ``scipy_wrap`` docstring).

    The ROADMAP §4 figure of 1e-10 AU over 100 orbits is *not* reached (the error is ~3e-8 AU
    and ~4e-10 AU is the best any ``rtol`` achieves); what matters for the project is that the
    numerical perihelion drift is ~1e-3 arcsec/cy, far below the 0.5 arcsec/cy noise floor.
    """
    a, e = 0.387, 0.2056
    period = kepler_period(a, GM_SUN)
    state = kepler_state(a, e, GM_SUN)
    t_eval = np.linspace(0.0, 100.0 * period, 1001)
    traj = integrate_scipy(state, make_central(GM_SUN), t_eval)
    pos_err, d_e, drift = kepler_run_diagnostics(traj, a, e)
    assert pos_err < 1e-7, f"max position error {pos_err:.3e} AU over 100 orbits"
    assert d_e < 1e-9, f"max |dE/E| {d_e:.3e}"
    assert abs(drift) < 0.05, f"numerical perihelion drift {drift:.4f} arcsec/cy"
    assert traj.meta["nfev"] < 2e5  # ~7e4 at the defaults; guards against a step-size collapse


@pytest.mark.slow
def test_scipy_kepler_century_accuracy_and_precession_floor():
    """One Julian century at the defaults: ~5e-7 AU, |dE/E| ~5e-10, drift ~1e-3 arcsec/cy."""
    a, e = 0.387, 0.2056
    state = kepler_state(a, e, GM_SUN)
    t_eval = np.linspace(0.0, JULIAN_CENTURY_DAYS, 2001)
    traj = integrate_scipy(state, make_central(GM_SUN), t_eval)
    pos_err, d_e, drift = kepler_run_diagnostics(traj, a, e)
    assert pos_err < 2e-6, f"max position error {pos_err:.3e} AU over a century"
    assert d_e < 2e-9, f"max |dE/E| {d_e:.3e}"
    assert abs(drift) < 0.05, f"numerical perihelion drift {drift:.4f} arcsec/cy"


def test_scipy_failure_and_bad_t_eval_raise():
    central = make_central(1.0)
    # radial infall into the singularity: the step size underflows -> status -1
    state = SystemState(0.0, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [0.0])
    with pytest.raises(RuntimeError, match="solve_ivp"):
        integrate_scipy(state, central, np.array([0.0, 2.0]))
    orbit = kepler_state(1.0, 0.1, 1.0, t=1.0)
    with pytest.raises(ValueError):
        integrate_scipy(orbit, central, np.array([1.0, 3.0, 2.0]))  # non-monotonic
    with pytest.raises(ValueError):
        integrate_scipy(orbit, central, np.array([0.5, 3.0]))  # straddles state.t
    with pytest.raises(ValueError):
        integrate_scipy(orbit, central, np.array([]))


# ----------------------------------------------------------------------------- (7) vel-dependent

DRAG_K = 0.7
DRAG_STATE = SystemState(0.0, [[0.0, 0.0, 0.0]], [[1.0, 0.5, 0.0]], [0.0])
DRAG_T = 2.0


def drag_exact(t: float) -> tuple[np.ndarray, np.ndarray]:
    v0 = DRAG_STATE.vel[0]
    return v0 * (1.0 - math.exp(-DRAG_K * t)) / DRAG_K, v0 * math.exp(-DRAG_K * t)


def drag_error(integ: Integrator, n: int) -> float:
    traj = integrate(DRAG_STATE, make_drag(DRAG_K), DRAG_T, DRAG_T / n, integ)
    pos_ref, vel_ref = drag_exact(DRAG_T)
    return state_error(traj, pos_ref, vel_ref)


@pytest.mark.parametrize(
    ("integ", "expected"),
    [
        (Leapfrog(vel_dependent=True), 2),  # implicit final kick: nominal order kept
        (Leapfrog(vel_dependent=False), 1),  # explicit kicks: degrades to first order
        (SymplecticEuler(vel_dependent=True), 1),
        (Yoshida4(vel_dependent=True, n_iter=4), 4),  # n_iter >= order keeps 4th order
        (RK4(), 4),
    ],
    ids=repr,
)
def test_velocity_dependent_drag_order(integ, expected):
    errors = (drag_error(integ, 100), drag_error(integ, 200))
    p = measured_order(errors)
    assert abs(p - expected) < 0.3, f"{integ!r}: measured order {p:.2f}, errors {errors}"


def test_velocity_dependent_leapfrog_matches_rk4():
    n = 200
    drag = make_drag(DRAG_K)
    rk4 = integrate(DRAG_STATE, drag, DRAG_T, DRAG_T / n, RK4())
    implicit = integrate(DRAG_STATE, drag, DRAG_T, DRAG_T / n, Leapfrog(vel_dependent=True))
    explicit = integrate(DRAG_STATE, drag, DRAG_T, DRAG_T / n, Leapfrog())
    d_implicit = np.abs(implicit.pos[-1] - rk4.pos[-1]).max()
    d_explicit = np.abs(explicit.pos[-1] - rk4.pos[-1]).max()
    assert d_implicit < 2e-5  # O(dt^2) agreement with the 4th-order reference
    assert d_explicit > 5e-4  # the explicit scheme is visibly worse (O(dt))
    assert d_explicit > 20 * d_implicit
    # drag only slows the particle down: speed decays monotonically and stays positive
    speed = np.linalg.norm(implicit.vel[:, 0], axis=1)
    assert np.all(np.diff(speed) < 0.0) and speed[-1] > 0.0


@pytest.mark.parametrize(
    ("integ", "steps_per_orbit", "bound"),
    [(Leapfrog(vel_dependent=True), 100, 2e-3), (Yoshida4(vel_dependent=True), 50, 5e-4)],
    ids=repr,
)
def test_velocity_dependent_gyroscopic_force_conserves_energy(integ, steps_per_orbit, bound):
    """Kepler + v x B: the magnetic term does no work, so the energy must stay bounded."""
    gm_c = 1.0
    period = kepler_period(1.0, gm_c)
    state = kepler_state(1.0, 0.2, gm_c)
    n_orbits = 30
    traj = integrate(
        state, make_gyro_kepler(gm_c, 0.02), n_orbits * period, period / steps_per_orbit, integ
    )
    d_e = rel_energy_error(traj, gm_c)
    early = d_e[: 5 * steps_per_orbit].max()
    late = d_e[-5 * steps_per_orbit :].max()
    assert d_e.max() < bound, f"{integ!r}: max |dE/E| = {d_e.max():.3e}"
    assert late < 1.2 * early, f"{integ!r}: energy error grows {early:.3e} -> {late:.3e}"


def test_explicit_leapfrog_with_gyroscopic_force_drifts():
    """Without the implicit kick the scheme is no longer time-symmetric and energy drifts."""
    gm_c = 1.0
    period = kepler_period(1.0, gm_c)
    state = kepler_state(1.0, 0.2, gm_c)
    spo = 100
    traj = integrate(state, make_gyro_kepler(gm_c, 0.02), 30 * period, period / spo, Leapfrog())
    d_e = rel_energy_error(traj, gm_c)
    early = d_e[: 5 * spo].max()
    late = d_e[-5 * spo :].max()
    assert late > 3.0 * early
    assert d_e.max() > 1e-2


# ----------------------------------------------------------------------------- (8) force arguments
#
# What reaches the force law: the stage times of every scheme, the grid times of the driver,
# the times of the scipy wrapper, and the ``gm`` array (which must arrive unchanged -- the real
# force laws read it, and a driver that zeroed it would silently give wrong N-body physics).


def expected_stage_times(integ: Integrator, t: float, dt: float) -> list[float]:
    """Times at which one ``step`` must evaluate the force, from the schemes' definitions."""
    if isinstance(integ, Euler):
        return [t]
    if isinstance(integ, RK4):
        return [t, t + 0.5 * dt, t + 0.5 * dt, t + dt]
    n_iter = integ.n_iter if integ.vel_dependent else 1
    if isinstance(integ, SymplecticEuler):
        return [t] * n_iter  # single kick at the step start (implicit: n_iter iterations)
    weights = type(integ).weights
    out, tau = [t], t
    for i, w in enumerate(weights):
        tau = tau + w * dt  # substep ends: w1, w1+w0, ... for Yoshida (outside [t, t+dt]!)
        out.extend([tau] * n_iter)
        if integ.vel_dependent and i + 1 < len(weights):
            out.append(tau)  # re-evaluation at the converged velocity for the next substep
    return out


@pytest.mark.parametrize("integ", all_instances(), ids=repr)
def test_step_passes_stage_times_and_gm_to_force(integ):
    t, dt = 0.3, 0.05
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vel = np.array([[0.0, 1.0, 0.0], [-0.5, 0.0, 0.1]])
    gm = np.array([1.0, 1e-3])
    spy = SpyForce(make_central(1.0))
    integ.step(pos, vel, gm, t, dt, spy)
    expected = expected_stage_times(integ, t, dt)
    assert len(spy.times) == len(expected)
    assert np.allclose(spy.times, expected, rtol=0.0, atol=1e-15), (spy.times, expected)
    assert all(np.array_equal(g, gm) for g in spy.gms)
    if integ.name in ("rk4", "leapfrog", "yoshida4", "yoshida6"):
        assert spy.times[-1] == pytest.approx(t + dt)  # the last evaluation is at t + dt
    if integ.name in ("yoshida4", "yoshida6"):
        # the substeps overshoot (w1 > 1) and the negative one goes back past the start
        assert max(spy.times) > t + dt and min(spy.times) < t


def test_driver_passes_grid_times_and_gm_to_force():
    state = kepler_state(0.5, 0.1, GM_SUN, t=12.5)
    state = state.with_(gm=np.array([3e-4]))  # non-zero so a zeroed gm is distinguishable
    spy = SpyForce(make_central(GM_SUN))
    traj = integrate(state, spy, 12.5 + 10.0, 0.4, Leapfrog())
    n = traj.meta["n_steps"]
    h = traj.meta["dt"]
    grid = 12.5 + h * np.arange(n + 1)
    # leapfrog evaluates at the start and end of every step: t0, t1, t1, t2, t2, ..., tn
    expected = np.concatenate([[grid[0]], np.repeat(grid[1:-1], 2), [grid[-1]]])
    assert len(spy.times) == 2 * n
    assert np.allclose(spy.times, expected, rtol=0.0, atol=1e-12)
    assert all(np.array_equal(g, state.gm) for g in spy.gms)
    # the same for RK4 (stage times inside every step) and a backward run
    spy = SpyForce(make_central(GM_SUN))
    traj = integrate(state, spy, 12.5 - 3.0, 0.5, RK4())
    h = traj.meta["dt"]
    assert h < 0.0 and traj.meta["n_steps"] == 6
    starts = 12.5 + h * np.arange(6)
    expected = np.concatenate([[s, s + h / 2, s + h / 2, s + h] for s in starts])
    assert np.allclose(spy.times, expected, rtol=0.0, atol=1e-12)
    assert all(np.array_equal(g, state.gm) for g in spy.gms)


def test_scipy_passes_times_and_gm_to_force():
    state = kepler_state(0.5, 0.1, GM_SUN, t=12.5).with_(gm=np.array([3e-4]))
    for t_end in (12.5 + 20.0, 12.5 - 20.0):
        spy = SpyForce(make_central(GM_SUN))
        traj = integrate_scipy(state, spy, np.linspace(12.5, t_end, 5))
        times = np.array(spy.times)
        assert len(times) == traj.meta["nfev"] > 20
        assert times[0] == 12.5  # first evaluation at the initial state
        lo, hi = sorted((12.5, t_end))
        # every stage lies in [t0, t_end] and the FSAL stage of the last step lands on t_end
        assert abs(times.min() - lo) < 1e-9 and abs(times.max() - hi) < 1e-9
        assert np.unique(times).size > 20  # genuinely distinct stage times, not all t0
        assert all(np.array_equal(g, state.gm) for g in spy.gms)


# ----------------------------------------------------------------------------- (9) pairwise 2-body

TB_A, TB_E = 0.387, 0.2056  # Mercury-like relative orbit
TB_GM1, TB_GM2 = GM_SUN, GM["Mercury"]
TB_MU = TB_GM1 + TB_GM2
TB_P = kepler_period(TB_A, TB_MU)
TB_T0 = 10.0
TB_STATE = two_body_state(TB_A, TB_E, TB_GM1, TB_GM2, t0=TB_T0)


def two_body_checks(traj: Trajectory, rel_bound: float, range_tol: float = 2e-4) -> None:
    """Momentum and barycentre conservation, r range, and the relative orbit vs. Kepler."""
    assert np.array_equal(traj.gm, TB_STATE.gm)
    rel_pos, rel_vel = traj.relative("Mercury", "Sun")
    # (a) relative orbit = Kepler orbit with mu = gm1 + gm2 (NOT gm1 alone)
    pos_ref, vel_ref = kepler_exact(TB_A, TB_E, TB_MU, traj.t - TB_T0)
    rel_err = np.linalg.norm(rel_pos - pos_ref, axis=1).max()
    assert rel_err < rel_bound, f"relative orbit error {rel_err:.3e} AU"
    # (b) r stays within [a(1-e), a(1+e)]
    r = np.linalg.norm(rel_pos, axis=1)
    assert r.min() > TB_A * (1.0 - TB_E) - range_tol and r.max() < TB_A * (1.0 + TB_E) + range_tol
    assert r.max() > TB_A * (1.0 + TB_E) - range_tol  # it does reach aphelion
    # (c) total momentum sum gm_i v_i and the barycentre are conserved to roundoff
    momentum = np.einsum("n,tnk->tk", traj.gm, traj.vel)
    assert np.abs(momentum - momentum[0]).max() < 1e-25
    barycentre = np.einsum("n,tnk->tk", traj.gm, traj.pos) / traj.gm.sum()
    assert np.abs(barycentre).max() < 1e-19
    # (d) both bodies move: the Sun's reflex orbit has semi-major axis a gm2 / mu
    sun_r = np.linalg.norm(traj.pos[:, 0], axis=1)
    assert sun_r.max() > 0.9 * TB_A * (1.0 + TB_E) * TB_GM2 / TB_MU


@pytest.mark.parametrize(
    ("integ", "steps_per_orbit", "rel_bound"),
    [
        (Leapfrog(), 400, 2e-3),
        (Yoshida4(), 100, 5e-4),
        (Yoshida4(vel_dependent=True), 100, 5e-4),
        (Yoshida6(), 100, 1e-5),
        (RK4(), 100, 5e-5),
    ],
    ids=repr,
)
def test_pairwise_two_body_reads_gm(integ, steps_per_orbit, rel_bound):
    traj = integrate(TB_STATE, pairwise_newton, TB_T0 + 3.0 * TB_P, TB_P / steps_per_orbit, integ)
    assert traj.names == ("Sun", "Mercury")
    two_body_checks(traj, rel_bound)


def test_pairwise_two_body_reads_gm_scipy():
    t_eval = TB_T0 + np.linspace(0.0, 3.0 * TB_P, 61)
    traj = integrate_scipy(TB_STATE, pairwise_newton, t_eval)
    two_body_checks(traj, 1e-10)
    # a gm = 0 body is a test particle: it feels the pair but does not disturb it (bitwise)
    far = np.array([[5.0, 0.0, 0.0]])
    with_tp = SystemState(
        TB_T0,
        np.concatenate([TB_STATE.pos, far]),
        np.concatenate([TB_STATE.vel, np.zeros((1, 3))]),
        np.append(TB_STATE.gm, 0.0),
        ("Sun", "Mercury", "probe"),
    )
    traj_tp = integrate(with_tp, pairwise_newton, TB_T0 + 3.0 * TB_P, TB_P / 100, Yoshida4())
    traj_2 = integrate(TB_STATE, pairwise_newton, TB_T0 + 3.0 * TB_P, TB_P / 100, Yoshida4())
    assert np.array_equal(traj_tp.pos[:, :2], traj_2.pos) and np.array_equal(
        traj_tp.vel[:, :2], traj_2.vel
    )
    assert np.linalg.norm(traj_tp.pos[-1, 2]) < 5.0  # the probe fell inwards


# ----------------------------------------------------------------------------- slow validation


@pytest.mark.slow
def test_yoshida6_energy_error_below_1e10_over_a_century():
    """ROADMAP §4: symplectic energy error over one century < 1e-10 (2-body, Mercury-like)."""
    a, e = 0.387, 0.2056
    period = kepler_period(a, GM_SUN)
    state = kepler_state(a, e, GM_SUN)
    traj = integrate(
        state, make_central(GM_SUN), 36525.0, period / 256, Yoshida6(), save_every=8
    )
    assert traj.t[-1] == 36525.0
    assert rel_energy_error(traj, GM_SUN).max() < 1e-10
