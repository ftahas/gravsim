"""Null geodesics of the Schwarzschild metric: light deflection and the Shapiro delay.

``gm`` (AU^3/day^2) and ``c`` (AU/day) are explicit so that lessons can scale the speed of
light down; ``r`` is the Schwarzschild areal radius in AU, ``b`` the impact parameter in AU,
angles are in radians and times in days.  With the gravitational radius ``mu = gm/c^2`` the
whole problem is scale free: every result depends on the geometry only through ``mu/b``.

Physics
-------
A photon has no proper time, so the equatorial geodesic is parametrised by an affine parameter
``lambda``.  The Killing vectors of the metric (see :mod:`gravsim.metric.schwarzschild`) give

    E = (1 - rs/r) c^2 dt/dlambda,      L = r^2 dphi/dlambda,      rs = 2 gm/c^2 ,

and the null condition g(k, k) = 0 reduces the radial motion to

    (dr/dlambda)^2 = E^2/c^2 - (1 - rs/r) L^2/r^2 .

Only the ratio ``b = c L/E`` (the impact parameter, the perpendicular distance of the incoming
asymptote from the mass) matters.  With ``u = 1/r`` and ' = d/dphi this is the first integral

    (u')^2 = 1/b^2 - u^2 + rs u^3 ,

whose derivative is the null Binet equation (Weinberg 8.5; MTW 25.6; Hartle 9.4)

    u'' + u = 3 (gm/c^2) u^2 .

A ray from infinity has ``u(0) = 0`` and ``u'(0) = 1/b`` when ``phi`` is measured from the
incoming asymptote; without the right-hand side ``u = sin(phi)/b`` is the straight line
``y = b``.  The code integrates the dimensionless ``w = b u`` (so ``w'' + w = 3 eps w^2`` with
``eps = mu/b``), and in fact the deviation ``v = w - sin(phi)`` from the straight line, which
keeps the small deflection free of cancellation in ``phi_end - pi``.

Closest approach and capture.  The turning point ``r_0`` (``u' = 0``) solves
``b^2 = r_0^2 / (1 - rs/r_0)``; it exists only for ``b > b_crit = 3 sqrt(3) gm/c^2``.  Rays
with ``b <= b_crit`` spiral into the photon sphere ``r = 3 gm/c^2`` (the unstable circular
null orbit, where ``b_crit`` is attained) and are captured.  As ``b -> b_crit`` from above the
deflection diverges logarithmically (Darwin 1959; Bozza 2002):
``delta ~ -ln(b/b_crit - 1) - 0.4002``.

Deflection.  The total bending is ``delta = phi_end - pi`` where ``phi_end`` is the azimuth at
which ``u`` returns to zero.  Solving the Binet equation perturbatively in ``mu/b``,

    delta = 4 mu/b + (15 pi/4) (mu/b)^2 + (128/3) (mu/b)^3 + O((mu/b)^4)

(Keeton & Petters 2005, eq. 29; the coefficients differ if the closest approach ``r_0`` is used
instead of ``b``).  The leading term, ``4 gm/(c^2 b)``, is Einstein's 1915 value and gives
1.7512" at the solar limb (``b = R_sun``): the quantity measured by Dyson, Eddington & Davidson
in 1919.  In the PPN framework the weak-field deflection is ``(1 + gamma)/2 * 4 gm/(c^2 b)``
(Will 2014, eq. 68), so ``gamma = 1`` is GR and ``gamma = 0`` recovers the Newtonian
corpuscle value ``2 gm/(c^2 b)`` of Soldner (1801; also Einstein 1911), which treats the photon
as a Newtonian particle on a hyperbola at speed ``c``: half of GR, because only the ``g_tt``
part of the metric (the Newtonian potential) is counted and the curvature of space
(``g_rr``, weighted by ``gamma``) is missed.

Shapiro delay.  A radar signal grazing the Sun takes longer than the Euclidean distance
suggests (Shapiro 1964).  Integrating ``dt/dr`` along the null geodesic to first order in
``mu``, in *isotropic* (PPN) coordinates where the spatial metric is conformally flat and
"Euclidean distance" is meaningful, the one-way excess over ``R/c`` is (Will 2014, eq. 70)

    Delta t = (1 + gamma) (gm/c^3) ln[ (r_e + r_r + R) / (r_e + r_r - R) ] ,

with ``r_e, r_r`` the distances of emitter and receiver from the mass and ``R`` the Euclidean
distance between them.  The impact parameter enters only through the geometry: for a signal
whose straight path passes the mass at distance ``b`` (closest-approach point between the two
end points, as at superior conjunction) ``R = sqrt(r_e^2 - b^2) + sqrt(r_r^2 - b^2)`` and
``r_e + r_r - R = b^2 / (r_e + sqrt(r_e^2 - b^2)) + b^2 / (r_r + sqrt(r_r^2 - b^2))``.
For ``b << r_e, r_r`` this reduces to the familiar conjunction form

    Delta t = (1 + gamma) (gm/c^3) ln( 4 r_e r_r / b^2 ) ,

which for Earth (1 AU) to Mars (1.52 AU) with ``b = R_sun`` gives 124 microseconds one way,
about 250 microseconds for the round trip of the 1976--77 Viking experiment.

:func:`shapiro_delay_numeric` integrates the exact coordinate time
``dt/dphi = 1 / (c b u^2 (1 - rs u))`` along the geodesic between two radii in Schwarzschild
coordinates and subtracts the flat-space light time of the chord between the end points.  The
chord must be taken in isotropic coordinates, ``r = rho (1 + mu/(2 rho))^2`` (so
``rho ~ r - mu``): the Schwarzschild ``r`` of each end point exceeds its isotropic value by
``mu``, which would shift the chord by ``~2 mu`` and the "delay" by the non-logarithmic
``2 mu/c`` (8 percent of the Earth--Mars value) -- the well-known coordinate dependence of the
constant term, which ephemeris fits absorb (Will 2014, sec. 3.4.2).

``b`` means different things in the two functions.  In :func:`shapiro_delay` it is the closest
approach of the *straight chord* between the end points, the configuration the weak-field
formula is written for; in :func:`shapiro_delay_numeric` it is the *impact parameter* of the
ray, the perpendicular distance of its incoming asymptote from the mass.  A ray of impact
parameter ``b`` bent by ``delta`` reaches a receiver that sits ``delta`` further round the mass,
so the chord between its actual end points passes closer, at
``d ~ b - delta r_e r_r/(r_e + r_r)`` (``d/b - 1 = -1.1e-3`` for Earth--Mars at the limb).
``shapiro_delay_numeric(r_e, r_r, b)`` therefore differs from ``shapiro_delay(r_e, r_r, b)`` by
two terms, both second order in the deflection and both of size
``delta^2 r_e r_r/((r_e + r_r) c)``:

* end-point geometry: the weak-field delay of the actual chord, ``shapiro_delay(.., d)``,
  exceeds the value at ``b`` by ``(4 mu/c) ln(b/d) ~ +delta^2 r_e r_r/((r_e + r_r) c)``
  (``+1.8e-4`` of the Earth--Mars delay);
* Fermat's principle: between *fixed* end points the null geodesic is the path of least
  coordinate time, so the exact delay lies *below* the straight-chord value, by
  ``(1/2) delta^2 r_e r_r/((r_e + r_r) c)`` to leading order: the bent path, treated as two
  straight segments meeting with a kink ``delta``, is longer than the chord by exactly this
  amount, and the stationarity of the travel time turns that excess into a deficit of the same
  size.  It is the ``-(1 + gamma)^2 mu^2 R/(c r_e r_r (1 + n_e . n_r))`` term of the
  second-order time transfer function (Teyssandier & Le Poncin-Lafitte 2008; ``R`` the chord,
  ``n`` the unit vectors of the end points), ``-0.9e-4`` for Earth--Mars.

The net residual ``shapiro_delay_numeric/shapiro_delay - 1`` is thus
``~ +(1/2) delta^2 r_e r_r/((r_e + r_r) c)`` over the delay: ``+0.9e-4`` for Earth--Mars at
``b = R_sun`` (0.011 microseconds of 123.6), growing with the end-point distances
(``1.7e-3`` for two end points at 30 AU), and resolved by the integration only to a noise
floor of ``~2e-14`` days (``1e-5`` of the delay) at the default tolerances.

References
----------
* Weinberg, *Gravitation and Cosmology* (1972), secs. 8.5 (deflection) and 8.7 (time delay).
* Misner, Thorne & Wheeler, *Gravitation* (1973), secs. 25.6, 40.3 and 40.4.
* Hartle, *Gravity* (2003), sec. 9.4.
* Will, *Living Rev. Relativ.* 17, 4 (2014), sec. 3.4 (PPN light deflection and time delay).
* Keeton & Petters, *Phys. Rev. D* 72, 104006 (2005) (post-post-Newtonian deflection series).
* Bozza, *Phys. Rev. D* 66, 103001 (2002) (strong-field logarithmic limit).
* Shapiro, *Phys. Rev. Lett.* 13, 789 (1964); Reasenberg et al., *ApJ* 234, L219 (1979).
* Teyssandier & Le Poncin-Lafitte, *Class. Quantum Grav.* 25, 145020 (2008) (time transfer
  function to second order; the enhanced ``1/(1 + n_e . n_r)`` term).
* Dyson, Eddington & Davidson, *Phil. Trans. R. Soc. A* 220, 291 (1920).
* Soldner, *Berliner Astronomisches Jahrbuch* (1804, written 1801).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from gravsim.constants import ARCSEC_PER_RAD, C_AU_DAY, GM_SUN, R_SUN_AU
from gravsim.metric.schwarzschild import photon_sphere_radius, schwarzschild_radius

__all__ = [
    "critical_impact_parameter",
    "closest_approach_radius",
    "impact_parameter",
    "isotropic_radius",
    "deflection_angle_weak",
    "deflection_newtonian_corpuscle",
    "solar_limb_deflection_arcsec",
    "deflection_angle",
    "ray_path",
    "ray_bundle",
    "null_transit",
    "shapiro_delay",
    "shapiro_delay_conjunction",
    "shapiro_delay_numeric",
]

_RTOL = 1e-12
_ATOL = 1e-14
_PHI_MAX = 100.0 * math.pi  # azimuth budget: enough for b/b_crit - 1 ~ 1e-130
_R_MAX_FACTOR = 1000.0  # ray_path arms extend to max(_R_MAX_FACTOR * b, _R_MAX_MIN_AU)
_R_MAX_MIN_AU = 50.0
_BOZZA_B = -0.4002  # constant of the strong-field limit delta ~ -ln(b/b_crit - 1) + _BOZZA_B


# --------------------------------------------------------------------------- characteristic scales
def critical_impact_parameter(gm: float, c: float = C_AU_DAY) -> float:
    """Critical impact parameter ``b_crit = 3 sqrt(3) gm/c^2`` (AU).

    Rays with ``b <= b_crit`` are captured; ``b_crit`` is the impact parameter of the photon
    sphere ``r = 3 gm/c^2``, i.e. ``b = r / sqrt(1 - rs/r)`` evaluated there.
    """
    return 3.0 * math.sqrt(3.0) * gm / c**2


def impact_parameter(r0, gm: float, c: float = C_AU_DAY):
    """Impact parameter ``b = r_0 / sqrt(1 - rs/r_0)`` of the ray with closest approach ``r_0``.

    Vectorised over ``r0``; NaN for ``r_0 < 3 gm/c^2`` (no turning point exists inside the
    photon sphere), and a float for scalar input.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    rs = schwarzschild_radius(gm, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(r0 >= photon_sphere_radius(gm, c), r0 / np.sqrt(1.0 - rs / r0), np.nan)
    return float(b) if b.ndim == 0 else b


def closest_approach_radius(b, gm: float, c: float = C_AU_DAY):
    """Turning point ``r_0`` of the ray with impact parameter ``b`` (AU).

    Largest root of ``r_0^3 - b^2 r_0 + rs b^2 = 0`` (from ``b^2 = r_0^2/(1 - rs/r_0)``),
    in closed trigonometric form ``r_0 = (2 b/sqrt 3) cos[ arccos(-3 sqrt(3) mu/b) / 3 ]``.
    Tends to ``b`` for ``b >> mu`` and to the photon sphere ``3 mu`` at ``b = b_crit``;
    NaN below ``b_crit`` (the ray is captured).  Vectorised over ``b``; float for scalar input.
    """
    b = np.asarray(b, dtype=np.float64)
    b_crit = critical_impact_parameter(gm, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = np.where(b >= b_crit, -b_crit / b, np.nan)
        r0 = 2.0 * b / math.sqrt(3.0) * np.cos(np.arccos(arg) / 3.0)
    return float(r0) if r0.ndim == 0 else r0


def isotropic_radius(r, gm: float, c: float = C_AU_DAY):
    """Isotropic radial coordinate ``rho`` of the Schwarzschild areal radius ``r``.

    Inverse of ``r = rho (1 + mu/(2 rho))^2``: ``rho = (r - mu + sqrt(r (r - 2 mu))) / 2``
    with ``mu = gm/c^2``; ``rho ~ r - mu`` far from the mass.  Defined for ``r >= rs``;
    NaN inside the horizon.  Vectorised over ``r``; float for scalar input.
    """
    r = np.asarray(r, dtype=np.float64)
    mu = gm / c**2
    with np.errstate(invalid="ignore"):
        rho = 0.5 * (r - mu + np.sqrt(r * (r - 2.0 * mu)))
    return float(rho) if rho.ndim == 0 else rho


# --------------------------------------------------------------------------- analytic deflection
def deflection_angle_weak(b, gm: float, c: float = C_AU_DAY, gamma: float = 1.0):
    """Weak-field PPN light deflection ``(1 + gamma)/2 * 4 gm/(c^2 b)`` (rad).

    ``gamma = 1`` is GR (Einstein 1915, ``4 gm/(c^2 b)``); ``gamma = 0`` is the Newtonian
    corpuscle value.  Vectorised over ``b``; float for scalar input.
    """
    b = np.asarray(b, dtype=np.float64)
    delta = 0.5 * (1.0 + gamma) * 4.0 * gm / (c**2 * b)
    return float(delta) if delta.ndim == 0 else delta


def deflection_newtonian_corpuscle(b, gm: float, c: float = C_AU_DAY):
    """Soldner's (1801) Newtonian deflection ``2 gm/(c^2 b)`` (rad): half of GR.

    A particle of speed ``c`` on a Newtonian hyperbola with ``v^2 >> gm/b`` is deflected by
    ``2 gm/(c^2 b)``; it equals the PPN formula with ``gamma = 0`` because it misses the
    spatial curvature.  Vectorised over ``b``; float for scalar input.
    """
    b = np.asarray(b, dtype=np.float64)
    delta = 2.0 * gm / (c**2 * b)
    return float(delta) if delta.ndim == 0 else delta


def solar_limb_deflection_arcsec(gamma: float = 1.0, c: float = C_AU_DAY) -> float:
    """Weak-field deflection of a ray grazing the Sun (``b = R_sun``) in arcseconds.

    ``(1 + gamma)/2 * 4 GM_sun/(c^2 R_sun)``: 1.7512" for ``gamma = 1`` (the Eddington 1919
    value, ``constants.LIGHT_DEFLECTION_LIMB_ARCSEC``), 0.8756" for ``gamma = 0``.
    """
    return deflection_angle_weak(R_SUN_AU, GM_SUN, c, gamma) * ARCSEC_PER_RAD


# --------------------------------------------------------------------------- ray integration
def _ray_parameter(b: float, gm: float, c: float) -> float:
    """Validate ``b`` against capture and return ``eps = mu/b``."""
    b_crit = critical_impact_parameter(gm, c)
    if not b > b_crit:
        raise ValueError(
            f"impact parameter b = {b} must exceed the critical value "
            f"3 sqrt(3) gm/c^2 = {b_crit}: the ray is captured"
        )
    return gm / (c**2 * b)


def _integrate_ray(eps: float, w_mark: float | None = None, dense_output: bool = False):
    """Integrate the null Binet equation from the incoming asymptote until the ray escapes.

    State ``(v, v')`` with ``w = b u = sin(phi) + v``, ``v'' + v = 3 eps w^2`` and
    ``v(0) = v'(0) = 0``.
    Terminates at the first downward zero of ``w`` (the outgoing asymptote).  When ``w_mark``
    is given, the crossings of ``w = w_mark`` are recorded as a second, non-terminal event.
    """

    def rhs(phi: float, state: np.ndarray) -> list[float]:
        w = math.sin(phi) + float(state[0])
        return [float(state[1]), 3.0 * eps * w * w - float(state[0])]

    def escape(phi: float, state: np.ndarray) -> float:
        return math.sin(phi) + float(state[0])

    escape.terminal = True
    escape.direction = -1
    events = [escape]
    if w_mark is not None:

        def mark(phi: float, state: np.ndarray) -> float:
            return math.sin(phi) + float(state[0]) - w_mark

        events.append(mark)

    sol = solve_ivp(
        rhs,
        (0.0, _PHI_MAX),
        [0.0, 0.0],
        method="DOP853",
        rtol=_RTOL,
        atol=_ATOL,
        events=events,
        dense_output=dense_output,
    )
    if not sol.success:
        raise RuntimeError(f"null geodesic integration failed: {sol.message}")
    if sol.t_events[0].size != 1:
        raise RuntimeError(
            "the ray did not escape within the azimuth budget (b too close to b_crit)"
        )
    return sol


def deflection_angle(b: float, gm: float, c: float = C_AU_DAY) -> float:
    """Exact Schwarzschild light deflection (rad) for impact parameter ``b``.

    Integrates the null Binet equation ``u'' + u = 3 gm u^2/c^2`` from the incoming asymptote
    (``u(0) = 0``, ``u'(0) = 1/b``) with ``scipy.integrate.solve_ivp`` (DOP853, rtol 1e-12,
    atol 1e-14) until ``u`` returns to zero, the root being located on the dense output by a
    terminal event, and returns ``phi_end - pi``.  Internally the deviation from the straight
    line ``b u = sin(phi)`` is integrated so that the deflection is obtained without cancellation.

    Parameters
    ----------
    b : float
        Impact parameter (AU), must exceed :func:`critical_impact_parameter`.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day); only ``gm/(c^2 b)`` matters.

    Returns
    -------
    float
        Total deflection in radians (``4 gm/(c^2 b)`` to leading order; see the module
        docstring for the series and the strong-field divergence).

    Raises
    ------
    ValueError
        If ``b <= 3 sqrt(3) gm/c^2`` (the ray is captured by the photon sphere).
    """
    eps = _ray_parameter(float(b), gm, c)
    sol = _integrate_ray(eps)
    return float(sol.t_events[0][0]) - math.pi


def ray_path(
    b: float,
    gm: float,
    c: float = C_AU_DAY,
    n_points: int = 2000,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian trace ``(x, y)`` of the ray with impact parameter ``b`` for plotting.

    The mass sits at the origin.  The ray enters from the left along the incoming asymptote
    ``y = b`` (``x -> -inf``, travelling towards ``+x``), passes the mass at the turning point
    :func:`closest_approach_radius` and leaves along the outgoing asymptote, bent towards the
    mass by :func:`deflection_angle` ``delta``: direction ``(cos delta, -sin delta)`` with
    perpendicular distance ``b`` from the origin, i.e. the line ``x sin delta + y cos delta = b``.
    Both arms extend to the radius ``r_max``; the samples are uniform in azimuth between the
    two crossings of ``r_max``, so they are dense near the mass and sparse on the straight
    arms.  In polar terms the trace is ``(x, y) = (-r cos phi, r sin phi)`` with ``phi``
    measured from the incoming asymptote.

    Parameters
    ----------
    b : float
        Impact parameter (AU); ``|b|`` must exceed :func:`critical_impact_parameter`.  A
        negative ``b`` gives the mirror image passing below the mass (``y -> -y``).
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).
    n_points : int
        Number of samples along the ray.
    r_max : float, optional
        Radius at which both arms are cut; default ``max(1000 |b|, 50 AU)``.

    Returns
    -------
    x, y : ndarray, shape (n_points,)
        Coordinates in AU, in the order travelled.

    Raises
    ------
    ValueError
        If the ray is captured (``|b| <= 3 sqrt(3) gm/c^2``) or ``r_max`` is not beyond the
        closest approach.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    b = float(b)
    side = -1.0 if b < 0.0 else 1.0
    b_abs = abs(b)
    eps = _ray_parameter(b_abs, gm, c)
    r_max = max(_R_MAX_FACTOR * b_abs, _R_MAX_MIN_AU) if r_max is None else float(r_max)
    w_mark = b_abs / r_max
    if not w_mark < 1.0:
        raise ValueError(f"r_max = {r_max} must exceed the impact parameter {b_abs}")
    sol = _integrate_ray(eps, w_mark=w_mark, dense_output=True)
    crossings = sol.t_events[1]
    if crossings.size != 2:
        raise RuntimeError(f"expected the ray to cross r_max twice, found {crossings.size}")
    phi = np.linspace(crossings[0], crossings[1], n_points)
    w = np.sin(phi) + sol.sol(phi)[0]
    r = b_abs / w
    return -r * np.cos(phi), side * r * np.sin(phi)


def ray_bundle(
    b_values,
    gm: float,
    c: float = C_AU_DAY,
    n_points: int = 2000,
    r_max: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """:func:`ray_path` for each impact parameter in ``b_values`` (signed, see there).

    Returns a list of ``(x, y)`` pairs in the order given; a common ``r_max`` (default
    ``max(1000 max|b|, 50 AU)``) is used so that all arms are cut at the same radius.
    """
    b_values = [float(b) for b in np.asarray(b_values, dtype=np.float64).reshape(-1)]
    if r_max is None and b_values:
        r_max = max(_R_MAX_FACTOR * max(abs(b) for b in b_values), _R_MAX_MIN_AU)
    return [ray_path(b, gm, c, n_points, r_max) for b in b_values]


# --------------------------------------------------------------------------- Shapiro delay
def _chord(r_a: float, r_b: float, b: float) -> tuple[float, float]:
    """``(R, r_a + r_b - R)`` for the straight path with closest approach ``b`` between the ends.

    ``R = sqrt(r_a^2 - b^2) + sqrt(r_b^2 - b^2)``; the small difference is evaluated as
    ``b^2/(r_a + sqrt(r_a^2 - b^2)) + b^2/(r_b + sqrt(r_b^2 - b^2))`` to avoid cancellation.
    """
    x_a = math.sqrt(r_a * r_a - b * b)
    x_b = math.sqrt(r_b * r_b - b * b)
    return x_a + x_b, b * b / (r_a + x_a) + b * b / (r_b + x_b)


def _check_shapiro_geometry(r_emit: float, r_recv: float, b: float) -> None:
    if not b > 0.0:
        raise ValueError("b must be positive")
    if not (r_emit > b and r_recv > b):
        raise ValueError(
            f"emitter and receiver radii ({r_emit}, {r_recv}) must exceed the impact parameter {b}"
        )


def shapiro_delay(
    r_emit: float,
    r_recv: float,
    b: float,
    gm: float,
    c: float = C_AU_DAY,
    gamma: float = 1.0,
) -> float:
    """One-way weak-field Shapiro delay (days) of a signal passing the mass at distance ``b``.

    ``Delta t = (1 + gamma) gm/c^3 * ln[(r_e + r_r + R)/(r_e + r_r - R)]`` with ``R`` the
    Euclidean distance between emitter and receiver.  ``b`` enters only through the geometry:
    the straight path is taken to pass the mass at ``b`` *between* the two end points (the
    superior-conjunction configuration), so ``R = sqrt(r_e^2 - b^2) + sqrt(r_r^2 - b^2)``.
    Multiply by two for a radar round trip.

    Parameters
    ----------
    r_emit, r_recv : float
        Distances of emitter and receiver from the mass (AU), both larger than ``b``.
    b : float
        Closest approach of the straight chord between the end points to the mass (AU).  This
        is not the impact parameter of the bent ray joining them, which is larger by
        ``~ delta r_e r_r/(r_e + r_r)``; see :func:`shapiro_delay_numeric`.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).
    gamma : float
        PPN parameter; 1 for GR, 0 halves the delay.

    Returns
    -------
    float
        Excess coordinate time in days (``~1.4e-9`` days = 124 microseconds for Earth--Mars
        at superior conjunction with ``b = R_sun``).
    """
    _check_shapiro_geometry(r_emit, r_recv, b)
    big, small = _chord(r_emit, r_recv, b)
    return (1.0 + gamma) * gm / c**3 * math.log((r_emit + r_recv + big) / small)


def shapiro_delay_conjunction(
    r_emit: float,
    r_recv: float,
    b: float,
    gm: float,
    c: float = C_AU_DAY,
    gamma: float = 1.0,
) -> float:
    """Conjunction form of the one-way Shapiro delay, ``(1 + gamma) gm/c^3 ln(4 r_e r_r/b^2)``.

    The ``b << r_e, r_r`` limit of :func:`shapiro_delay` (same arguments and units); for the
    Sun the two agree to a part in a million.
    """
    _check_shapiro_geometry(r_emit, r_recv, b)
    return (1.0 + gamma) * gm / c**3 * math.log(4.0 * r_emit * r_recv / (b * b))


def null_transit(
    r_emit: float,
    r_recv: float,
    b: float,
    gm: float,
    c: float = C_AU_DAY,
) -> tuple[float, float]:
    """Azimuth swept and coordinate time elapsed along the ray between two radii.

    The ray with impact parameter ``b`` is followed inwards from ``r_emit``, through its
    closest approach, and outwards to ``r_recv``.  The state ``(u, u', t)`` obeys the null
    Binet equation and ``dt/dphi = 1/(c b u^2 (1 - rs u))``; the initial slope comes from the
    first integral ``u'^2 = 1/b^2 - u^2 + rs u^3`` and the arrival is located by a terminal
    event on ``u = 1/r_recv`` (DOP853, rtol 1e-12, atol 1e-14, in the dimensionless ``w = b u``
    and ``c t/b``).

    Parameters
    ----------
    r_emit, r_recv : float
        Schwarzschild radii (AU) of the end points, both larger than ``b``.
    b : float
        Impact parameter (AU), larger than :func:`critical_impact_parameter`.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).

    Returns
    -------
    delta_phi, t : float
        Angle (rad) between the end points as seen from the mass, and coordinate time (days).
    """
    _check_shapiro_geometry(r_emit, r_recv, b)
    eps = _ray_parameter(float(b), gm, c)
    w_e = b / r_emit
    w_r = b / r_recv
    dw_e = math.sqrt(1.0 - w_e * w_e + 2.0 * eps * w_e**3)  # inbound: w increases

    def rhs(_phi: float, state: np.ndarray) -> list[float]:
        w, dw = float(state[0]), float(state[1])
        return [dw, 3.0 * eps * w * w - w, 1.0 / (w * w * (1.0 - 2.0 * eps * w))]

    def arrive(_phi: float, state: np.ndarray) -> float:
        return float(state[0]) - w_r

    arrive.terminal = True
    arrive.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, _PHI_MAX),
        [w_e, dw_e, 0.0],
        method="DOP853",
        rtol=_RTOL,
        atol=_ATOL,
        events=arrive,
    )
    if not sol.success:
        raise RuntimeError(f"null transit integration failed: {sol.message}")
    if sol.t_events[0].size != 1:
        raise RuntimeError("the ray did not reach r_recv within the azimuth budget")
    delta_phi = float(sol.t_events[0][0])
    t = float(sol.y_events[0][0][2]) * b / c
    return delta_phi, t


def shapiro_delay_numeric(
    r_emit: float,
    r_recv: float,
    b: float,
    gm: float,
    c: float = C_AU_DAY,
) -> float:
    """One-way Shapiro delay (days) from the exact null geodesic.

    The coordinate time of :func:`null_transit` minus the flat-space light time ``R/c`` of
    the chord between the end points, the chord being measured in isotropic coordinates
    (:func:`isotropic_radius`), where the weak-field formula is defined; see the module
    docstring for why the Schwarzschild chord would be off by the constant ``2 gm/c^3``.
    Agrees with :func:`shapiro_delay` (``gamma = 1``) at the same ``b`` to ``~1e-4`` for
    Earth--Mars at the solar limb, but ``b`` is the impact parameter of the ray here and the
    closest approach of the chord there.  With ``delta`` the deflection and ``R = r_e r_r /
    (r_e + r_r)``, the residual ``~ +(1/2) delta^2 R/c`` is the sum of the end-point term
    ``+delta^2 R/c`` (the chord between the actual end points passes closer than ``b``) and the
    Fermat deficit ``-(1/2) delta^2 R/c`` (the geodesic is the path of least coordinate time
    between its end points, so the exact delay is *below* the straight-chord value); see the
    module docstring.  It grows with the end-point distances (``1.7e-3`` of the delay for two
    end points at 30 AU) and the integration resolves it only to ``~2e-14`` days (``1e-5`` of
    the delay).

    Parameters
    ----------
    r_emit, r_recv : float
        Schwarzschild radii (AU) of the end points, both larger than ``b``.
    b : float
        Impact parameter (AU) of the ray (perpendicular distance of its incoming asymptote from
        the mass), larger than :func:`critical_impact_parameter`.  The chord between the two
        end points passes closer, at ``~ b - delta r_e r_r/(r_e + r_r)``.
    gm, c : float
        Central GM (AU^3/day^2) and speed of light (AU/day).

    Returns
    -------
    float
        Excess coordinate time in days.
    """
    delta_phi, t = null_transit(r_emit, r_recv, b, gm, c)
    rho_e = isotropic_radius(r_emit, gm, c)
    rho_r = isotropic_radius(r_recv, gm, c)
    chord = math.sqrt(rho_e * rho_e + rho_r * rho_r - 2.0 * rho_e * rho_r * math.cos(delta_phi))
    return t - chord / c
