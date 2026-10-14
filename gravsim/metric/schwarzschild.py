"""Timelike geodesics of the Schwarzschild metric: a test particle around a point mass.

Everything is per unit rest mass; ``gm`` (AU^3/day^2) and ``c`` (AU/day) are explicit so that
lessons can scale the speed of light down.  ``r`` is the Schwarzschild areal radius in AU,
``tau`` is proper time and ``t`` is Schwarzschild coordinate time, both in days.

Physics
-------
With the gravitational radius ``mu = gm/c^2`` and Schwarzschild radius ``rs = 2 mu``, the line
element in the equatorial plane (theta = pi/2; every geodesic is planar) is

    c^2 dtau^2 = (1 - rs/r) c^2 dt^2 - dr^2 / (1 - rs/r) - r^2 dphi^2 .

The two Killing vectors d/dt and d/dphi give two constants of the motion,

    E = (1 - rs/r) c^2 dt/dtau      (specific energy, rest mass included, units of c^2),
    L = r^2 dphi/dtau               (specific angular momentum),

and the normalisation g(u, u) = -c^2 of the 4-velocity reduces the radial motion to a
one-dimensional problem with an *exact* effective potential (MTW 25.5, Carroll 5.4, Hartle 9.3):

    (1/2) (dr/dtau)^2 + V_eff(r) = (E^2/c^2 - c^2)/2 ,
    V_eff(r) = -gm/r + L^2/(2 r^2) - gm L^2/(c^2 r^3) .

The first two terms are Newtonian; the last one (the only GR term) overturns the centrifugal
barrier at small r.  Differentiating the energy equation gives the exact radial equation

    d^2r/dtau^2 = -dV_eff/dr = -gm/r^2 + L^2/r^3 - 3 gm L^2/(c^2 r^4) ,

which is integrated by :func:`geodesic_orbit` together with dphi/dtau = L/r^2 and
dt/dtau = E / (c^2 (1 - rs/r)).

Circular orbits sit where dV_eff/dr = 0, i.e. at the roots of gm r^2 - L^2 r + 3 mu L^2 = 0:

    r_(+/-) = (L^2 / 2 gm) [ 1 +/- sqrt(1 - 12 (gm/c)^2 / L^2) ] ,

the outer root being stable (minimum of V_eff) and the inner one unstable (maximum).  They
merge when L^2 = 12 (gm/c)^2 at the innermost stable circular orbit r_isco = 6 mu; below
L^2 = 12 (gm/c)^2 there are no circular orbits at all and every inward-moving particle is
captured.  The unstable branch ends at the photon sphere r = 3 mu as L -> infinity.

Shape of the orbit (Binet equation).  With u = 1/r and ' = d/dphi, using dr/dtau = -L u',

    u'' + u = gm/L^2 + 3 (gm/c^2) u^2 ,

whose Newtonian part (3 mu u^2 dropped) is solved by the Kepler conic u = (1 + e cos phi)/p with
p = L^2/gm.  :func:`binet_orbit` integrates this equation in phi, together with the coordinate
time dt/dphi = (dt/dtau)/(dphi/dtau) = E r^2 / (c^2 L (1 - rs/r)).

Initial data.  Both orbit routines take a 2-D position ``r0`` and velocity ``v0`` interpreted as
(dx/dtau, dy/dtau) in the plane of the orbit, so that L = x vy - y vx (signed; the sign fixes
the sense of rotation), the radial proper velocity is v_r = (r0 . v0)/|r0|, the Binet slope is
u'(phi0) = -v_r/|L|, and the 4-velocity normalisation at the initial point gives

    E = c sqrt( (1 - rs/r0) (c^2 + L^2/r0^2) + v_r^2 ) .

Perihelion advance.  Solving the Binet equation perturbatively in mu u gives the first
post-Newtonian (1PN) advance per orbit (Einstein 1915; Weinberg 8.6)

    Delta phi = 6 pi gm / (c^2 a (1 - e^2)) = 6 pi mu / p ,

which is 5.02e-7 rad/orbit = 42.98"/century for Mercury.  Carrying the expansion of the exact
advance one order further, for an orbit started at perihelion with the Newtonian velocity of the
osculating ellipse (a, e) (the convention of :func:`precession_per_orbit_numeric`),

    Delta phi = 6 pi (mu/p) [ 1 + (30 + 5 e^2)/4 (mu/p) + O((mu/p)^2) ] ,

i.e. the exact value exceeds 1PN by a relative ~7.5 mu/p (2e-7 for Mercury, 2e-3 at c/100).
With the apsides parametrised geometrically instead, u = (1 +/- e)/p at the turning points, the
coefficient is the textbook (18 + e^2)/4.  :func:`precession_per_orbit_numeric` measures the
exact advance from the integrated orbit so the two can be compared at any field strength.

References
----------
* Misner, Thorne & Wheeler, *Gravitation* (1973), ch. 25 (effective potential, ISCO).
* Carroll, *Spacetime and Geometry* (2004), sec. 5.4.
* Hartle, *Gravity* (2003), ch. 9.
* Weinberg, *Gravitation and Cosmology* (1972), sec. 8.6 (perihelion precession).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from gravsim.constants import C_AU_DAY

__all__ = [
    "schwarzschild_radius",
    "photon_sphere_radius",
    "isco_radius",
    "effective_potential",
    "newton_effective_potential",
    "radial_acceleration",
    "circular_orbit_radii",
    "circular_orbit_angular_momentum",
    "conserved_quantities",
    "binet_orbit",
    "geodesic_orbit",
    "periapsis_angles",
    "precession_per_orbit_1pn",
    "precession_per_orbit_numeric",
]

_RTOL = 1e-12
_ATOL = 1e-14
_R_STOP_FACTOR = 1.001  # default stopping radius in units of the Schwarzschild radius


# --------------------------------------------------------------------------- characteristic radii
def schwarzschild_radius(gm: float, c: float = C_AU_DAY) -> float:
    """Schwarzschild radius ``rs = 2 gm / c^2`` (event horizon) in AU."""
    return 2.0 * gm / c**2


def photon_sphere_radius(gm: float, c: float = C_AU_DAY) -> float:
    """Photon-sphere radius ``3 gm / c^2``: the unstable circular orbit of light."""
    return 3.0 * gm / c**2


def isco_radius(gm: float, c: float = C_AU_DAY) -> float:
    """Innermost stable circular orbit ``6 gm / c^2`` (where L^2 = 12 (gm/c)^2)."""
    return 6.0 * gm / c**2


# --------------------------------------------------------------------------- effective potential
def effective_potential(r, L: float, gm: float, c: float = C_AU_DAY) -> np.ndarray:
    """Exact Schwarzschild effective potential per unit mass.

    ``V_eff(r) = -gm/r + L^2/(2 r^2) - gm L^2/(c^2 r^3)`` with ``L = r^2 dphi/dtau``, so that
    ``(1/2)(dr/dtau)^2 + V_eff = (E^2/c^2 - c^2)/2`` along a timelike geodesic.

    Parameters
    ----------
    r : array_like
        Areal radius in AU (vectorised).
    L : float
        Specific angular momentum in AU^2/day.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).

    Returns
    -------
    ndarray
        Potential in AU^2/day^2, same shape as ``r``.
    """
    r = np.asarray(r, dtype=np.float64)
    return -gm / r + L**2 / (2.0 * r**2) - gm * L**2 / (c**2 * r**3)


def newton_effective_potential(r, L: float, gm: float) -> np.ndarray:
    """Newtonian effective potential ``-gm/r + L^2/(2 r^2)`` (the ``c -> infinity`` limit)."""
    r = np.asarray(r, dtype=np.float64)
    return -gm / r + L**2 / (2.0 * r**2)


def radial_acceleration(r, L: float, gm: float, c: float = C_AU_DAY) -> np.ndarray:
    """Proper-time radial acceleration ``d^2r/dtau^2 = -dV_eff/dr``.

    ``-gm/r^2 + L^2/r^3 - 3 gm L^2/(c^2 r^4)``; this is the right-hand side integrated by
    :func:`geodesic_orbit`.
    """
    r = np.asarray(r, dtype=np.float64)
    return -gm / r**2 + L**2 / r**3 - 3.0 * gm * L**2 / (c**2 * r**4)


def circular_orbit_radii(L, gm: float, c: float = C_AU_DAY):
    """Radii of the stable and unstable circular orbits for angular momentum ``L``.

    Roots of ``dV_eff/dr = 0``: ``r_± = (L^2/2gm) [1 ± sqrt(1 - 12 (gm/c)^2/L^2)]``.  The
    inner root is evaluated from the product of the roots, ``r_+ r_- = 3 L^2/c^2``, to avoid
    cancellation.  Both are NaN when ``L^2 < 12 (gm/c)^2`` (no circular orbit exists); they
    coincide at the ISCO ``6 gm/c^2`` when ``L^2 = 12 (gm/c)^2``.

    Parameters
    ----------
    L : float or array_like
        Specific angular momentum (AU^2/day); the sign is irrelevant.
    gm, c : float
        Central GM and speed of light.

    Returns
    -------
    (r_stable, r_unstable) : tuple of float or ndarray
        Radii in AU (floats for scalar ``L``).
    """
    L2 = np.asarray(L, dtype=np.float64) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = 1.0 - 12.0 * (gm / c) ** 2 / L2
        disc = np.where(disc >= 0.0, disc, np.nan)
        r_stable = L2 / (2.0 * gm) * (1.0 + np.sqrt(disc))
        r_unstable = 3.0 * L2 / (c**2 * r_stable)
    if r_stable.ndim == 0:
        return float(r_stable), float(r_unstable)
    return r_stable, r_unstable


def circular_orbit_angular_momentum(r, gm: float, c: float = C_AU_DAY):
    """Angular momentum ``L = sqrt(gm r^2 / (r - 3 gm/c^2))`` of the circular orbit at ``r``.

    Inverse of :func:`circular_orbit_radii`; NaN for ``r <= 3 gm/c^2`` (inside the photon sphere
    no timelike circular orbit exists).  Vectorised over ``r``; returns a float for scalar input.
    """
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        L2 = np.where(r > 3.0 * gm / c**2, gm * r**2 / (r - 3.0 * gm / c**2), np.nan)
        L = np.sqrt(L2)
    return float(L) if L.ndim == 0 else L


# --------------------------------------------------------------------------- initial data
def _planar_initial_data(r0, v0) -> tuple[float, float, float, float, float]:
    """Return ``(x, y, vx, vy, r)`` from 2-D ``r0`` and ``v0`` after validation."""
    r0 = np.asarray(r0, dtype=np.float64).reshape(-1)
    v0 = np.asarray(v0, dtype=np.float64).reshape(-1)
    if r0.shape != (2,) or v0.shape != (2,):
        raise ValueError(f"r0 and v0 must be 2-D vectors, got shapes {r0.shape} and {v0.shape}")
    x, y = float(r0[0]), float(r0[1])
    vx, vy = float(v0[0]), float(v0[1])
    r = math.hypot(x, y)
    if r <= 0.0:
        raise ValueError("r0 must not be the origin")
    return x, y, vx, vy, r


def conserved_quantities(r0, v0, gm: float, c: float = C_AU_DAY) -> tuple[float, float]:
    """Specific energy ``E`` and signed angular momentum ``L`` of the geodesic through ``(r0, v0)``.

    ``v0`` is the proper-time velocity (dx/dtau, dy/dtau).  ``L = x vy - y vx`` and, from the
    normalisation of the 4-velocity, ``E = c sqrt((1 - rs/r)(c^2 + L^2/r^2) + v_r^2)`` with
    ``v_r = (r0 . v0)/|r0|``.  ``E`` has units of c^2 (AU^2/day^2) and tends to ``c^2 + eps``
    (``eps`` the Newtonian specific energy) in the weak-field limit.

    Raises
    ------
    ValueError
        If the starting point is not outside the horizon ``rs = 2 gm/c^2``.
    """
    x, y, vx, vy, r = _planar_initial_data(r0, v0)
    rs = schwarzschild_radius(gm, c)
    if r <= rs:
        raise ValueError(f"starting radius {r} is inside the horizon rs = {rs}")
    L = x * vy - y * vx
    v_r = (x * vx + y * vy) / r
    E = c * math.sqrt((1.0 - rs / r) * (c**2 + L**2 / r**2) + v_r**2)
    return E, L


def _resolve_r_stop(r_stop: float | None, r: float, gm: float, c: float) -> float:
    r_stop = _R_STOP_FACTOR * schwarzschild_radius(gm, c) if r_stop is None else float(r_stop)
    if r <= r_stop:
        raise ValueError(f"starting radius {r} must exceed the stopping radius {r_stop}")
    return r_stop


# --------------------------------------------------------------------------- orbit integrators
def binet_orbit(
    r0,
    v0,
    gm: float,
    c: float = C_AU_DAY,
    n_orbits: float = 1.0,
    n_per_orbit: int = 2000,
    r_stop: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planar orbit ``r(phi)`` from the Binet equation ``u'' + u = gm/L^2 + 3 gm u^2/c^2``.

    The equation is integrated in the azimuth with ``scipy.integrate.solve_ivp`` (DOP853,
    rtol 1e-12, atol 1e-14) over ``2 pi n_orbits`` radians, together with the coordinate time
    ``dt/dphi = E r^2 / (c^2 L (1 - rs/r))``.

    Parameters
    ----------
    r0, v0 : array_like, shape (2,)
        Initial position (AU) and proper-time velocity (AU/day) in the orbital plane.
        ``L = |r0 x v0|`` and ``u'(phi0) = -(r0 . v0)/(|r0| L)``.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).
    n_orbits : float
        Number of azimuthal turns ``2 pi`` to integrate (a fraction is allowed).
    n_per_orbit : int
        Output samples per ``2 pi`` of azimuth (uniform grid).
    r_stop : float, optional
        Terminate when the orbit falls to this radius (a capture); default ``1.001 rs``.
        The returned arrays are then truncated to the samples reached.

    Returns
    -------
    phi, r, t : ndarray
        Azimuth (rad, unwrapped, starting at ``atan2(y0, x0)``), radius (AU) and coordinate
        time elapsed (days), all of the same length ``round(n_orbits * n_per_orbit) + 1``.

    Notes
    -----
    The Binet equation is even in ``phi``; for a retrograde start (``r0 x v0 < 0``) it is solved
    in ``|phi - phi0|`` and the returned ``phi`` decreases, so the sense of rotation is preserved.
    ``L = 0`` (radial infall) is rejected: use :func:`geodesic_orbit` for it.
    """
    if n_orbits <= 0.0:
        raise ValueError("n_orbits must be positive")
    if n_per_orbit < 1:
        raise ValueError("n_per_orbit must be at least 1")
    x, y, vx, vy, r = _planar_initial_data(r0, v0)
    E, L = conserved_quantities(r0, v0, gm, c)
    if L == 0.0:
        raise ValueError("L = 0: the Binet equation does not describe radial infall")
    r_stop = _resolve_r_stop(r_stop, r, gm, c)

    sense = 1.0 if L > 0.0 else -1.0
    L_abs = abs(L)
    rs = schwarzschild_radius(gm, c)
    phi0 = math.atan2(y, x)
    v_r = (x * vx + y * vy) / r
    u0 = 1.0 / r
    du0 = -v_r / L_abs
    inv_p = gm / L_abs**2  # Newtonian 1/p = gm/L^2
    three_mu = 3.0 * gm / c**2
    e_over_c2 = E / c**2
    u_stop = 1.0 / r_stop

    def rhs(_psi: float, state: np.ndarray) -> list[float]:
        u, du, _t = float(state[0]), float(state[1]), float(state[2])
        return [du, inv_p + three_mu * u * u - u, e_over_c2 / (L_abs * u * u * (1.0 - rs * u))]

    def capture(_psi: float, state: np.ndarray) -> float:
        return u_stop - float(state[0])

    capture.terminal = True
    capture.direction = -1

    psi_end = 2.0 * math.pi * n_orbits
    n_points = int(round(n_orbits * n_per_orbit)) + 1
    psi_eval = np.linspace(0.0, psi_end, n_points)
    sol = solve_ivp(
        rhs,
        (0.0, psi_end),
        [u0, du0, 0.0],
        method="DOP853",
        t_eval=psi_eval,
        rtol=_RTOL,
        atol=_ATOL,
        events=capture,
    )
    if not sol.success:
        raise RuntimeError(f"binet_orbit integration failed: {sol.message}")
    phi = phi0 + sense * sol.t
    r_of_phi = 1.0 / sol.y[0]
    t = sol.y[2]
    return phi, r_of_phi, t


def geodesic_orbit(
    r0,
    v0,
    gm: float,
    c: float = C_AU_DAY,
    tau_end: float = 1.0,
    n_samples: int = 20000,
    r_stop: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Timelike geodesic integrated in proper time with the exact radial equation.

    State ``(r, dr/dtau, phi, t)`` with

        d^2r/dtau^2 = -gm/r^2 + L^2/r^3 - 3 gm L^2/(c^2 r^4)   (= -dV_eff/dr),
        dphi/dtau   = L/r^2,
        dt/dtau     = E / (c^2 (1 - rs/r)),

    ``E`` and ``L`` fixed by the initial data (:func:`conserved_quantities`).  DOP853 at
    rtol 1e-12, atol 1e-14, sampled on a uniform proper-time grid.

    Parameters
    ----------
    r0, v0 : array_like, shape (2,)
        Initial position (AU) and proper-time velocity (AU/day) in the orbital plane.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).
    tau_end : float
        Proper time to integrate (days).
    n_samples : int
        Number of output samples on ``[0, tau_end]`` (inclusive).
    r_stop : float, optional
        Terminate when the orbit falls to this radius (a capture); default ``1.001 rs``.
        The returned arrays are then truncated to the samples reached.

    Returns
    -------
    tau, t, r, phi : ndarray
        Proper time, coordinate time (both days, from 0), radius (AU) and unwrapped azimuth
        (rad, signed by the sense of rotation), all of the same length.
    """
    if tau_end <= 0.0:
        raise ValueError("tau_end must be positive")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    x, y, vx, vy, r = _planar_initial_data(r0, v0)
    E, L = conserved_quantities(r0, v0, gm, c)
    r_stop = _resolve_r_stop(r_stop, r, gm, c)

    rs = schwarzschild_radius(gm, c)
    phi0 = math.atan2(y, x)
    v_r = (x * vx + y * vy) / r
    L2 = L * L
    three_gm_L2_over_c2 = 3.0 * gm * L2 / c**2
    e_over_c2 = E / c**2

    def rhs(_tau: float, state: np.ndarray) -> list[float]:
        rr, vr = float(state[0]), float(state[1])
        r2 = rr * rr
        acc = -gm / r2 + L2 / (r2 * rr) - three_gm_L2_over_c2 / (r2 * r2)
        return [vr, acc, L / r2, e_over_c2 / (1.0 - rs / rr)]

    def capture(_tau: float, state: np.ndarray) -> float:
        return float(state[0]) - r_stop

    capture.terminal = True
    capture.direction = -1

    tau_eval = np.linspace(0.0, tau_end, n_samples)
    sol = solve_ivp(
        rhs,
        (0.0, tau_end),
        [r, v_r, phi0, 0.0],
        method="DOP853",
        t_eval=tau_eval,
        rtol=_RTOL,
        atol=_ATOL,
        events=capture,
    )
    if not sol.success:
        raise RuntimeError(f"geodesic_orbit integration failed: {sol.message}")
    return sol.t, sol.y[3], sol.y[0], sol.y[2]


# --------------------------------------------------------------------------- perihelion advance
def periapsis_angles(phi, r) -> np.ndarray:
    """Azimuths of the local minima of ``r(phi)`` refined by parabolic interpolation.

    An interior sample ``i`` is a minimum when ``r[i] < r[i-1]`` and ``r[i] <= r[i+1]``; the
    vertex of the parabola through the three samples around it (any spacing) is returned.
    End points are never reported.

    Parameters
    ----------
    phi, r : array_like, shape (n,)
        Azimuth (monotonic) and radius along the orbit.

    Returns
    -------
    ndarray
        Periapsis azimuths in the order encountered (empty if there is no interior minimum).
    """
    phi = np.asarray(phi, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if phi.shape != r.shape or phi.ndim != 1:
        raise ValueError("phi and r must be 1-D arrays of the same length")
    if r.size < 3:
        return np.empty(0)
    idx = np.flatnonzero((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:])) + 1
    x0, x1, x2 = phi[idx - 1], phi[idx], phi[idx + 1]
    y0, y1, y2 = r[idx - 1], r[idx], r[idx + 1]
    d10, d12 = x1 - x0, x1 - x2
    num = d10**2 * (y1 - y2) - d12**2 * (y1 - y0)
    den = d10 * (y1 - y2) - d12 * (y1 - y0)
    return x1 - 0.5 * num / den


def precession_per_orbit_1pn(a: float, e: float, gm: float, c: float = C_AU_DAY) -> float:
    """First post-Newtonian perihelion advance per orbit ``6 pi gm / (c^2 a (1 - e^2))`` (rad)."""
    return 6.0 * math.pi * gm / (c**2 * a * (1.0 - e**2))


def precession_per_orbit_numeric(
    a: float,
    e: float,
    gm: float,
    c: float = C_AU_DAY,
    n_orbits: int = 50,
    n_per_orbit: int = 2000,
) -> float:
    """Mean perihelion advance per orbit (rad) measured on the exact Binet orbit.

    The orbit starts at the perihelion of the Kepler ellipse with elements ``(a, e)``,
    ``r_p = a (1 - e)``, ``v_p = sqrt(gm (1 + e) / (a (1 - e)))`` (Newtonian velocity, taken as
    the proper-time velocity).  ``a`` and ``e`` are therefore the *osculating Newtonian elements
    at perihelion*; the exact orbit's aphelion differs from ``a (1 + e)`` by a relative amount of
    order ``gm/(c^2 a (1 - e^2))``.  Successive perihelia are the minima of ``r(phi)`` refined by
    :func:`periapsis_angles` (the starting point is one by construction); the result is the mean
    azimuth between consecutive perihelia minus ``2 pi``.

    Parameters
    ----------
    a, e : float
        Semi-major axis (AU) and eccentricity (``0 < e < 1``) of the starting Kepler ellipse.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).
    n_orbits : int
        Azimuthal turns integrated; the number of perihelion passages found is about this.
    n_per_orbit : int
        Samples per turn passed to :func:`binet_orbit`.

    Returns
    -------
    float
        Advance per orbit in radians (compare :func:`precession_per_orbit_1pn`).
    """
    if not 0.0 < e < 1.0:
        raise ValueError("e must lie in (0, 1) for a perihelion to be defined")
    if a <= 0.0:
        raise ValueError("a must be positive")
    r_p = a * (1.0 - e)
    v_p = math.sqrt(gm * (1.0 + e) / (a * (1.0 - e)))
    phi, r, _t = binet_orbit([r_p, 0.0], [0.0, v_p], gm, c, n_orbits, n_per_orbit)
    peri = periapsis_angles(phi, r)
    if r[1] > r[0]:  # the start is a perihelion (dr/dphi = 0 and r increases afterwards)
        peri = np.concatenate(([phi[0]], peri))
    if peri.size < 2:
        raise ValueError("fewer than two perihelion passages found; increase n_orbits")
    return float((peri[-1] - peri[0]) / (peri.size - 1) - 2.0 * math.pi)
