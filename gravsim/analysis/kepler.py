"""Analytic two-body (Kepler) solution: Kepler's equation, elements <-> state, propagation.

Physics
-------
The relative motion of two point masses with gravitational parameter ``gm = G (M + m)``
(AU^3/day^2) is a conic with the focus at the origin.  For bound orbits (``0 <= e < 1``):

* semi-latus rectum      ``p = a (1 - e^2)``,  radius  ``r = p / (1 + e cos nu)``
* vis-viva               ``v^2 = gm (2/r - 1/a)``  ->  ``a = 1 / (2/r - v^2/gm)``
* mean motion, period    ``n = sqrt(gm / a^3)``,  ``P = 2 pi / n``             (Kepler III)
* Kepler's equation      ``M = E - e sin E``,  with  ``M = n (t - t_peri)``
* true <-> eccentric     ``tan(nu/2) = sqrt((1 + e)/(1 - e)) tan(E/2)``

Perifocal frame (``P`` toward perihelion, ``Q = W x P`` in the plane, ``W = h/|h|``):

    r = r (cos nu P + sin nu Q),      v = sqrt(gm/p) (-sin nu P + (e + cos nu) Q)

and the PQW axes are rotated into the reference frame with the 3-1-3 Euler sequence
``R_z(raan) R_x(inc) R_z(argp)`` (Bate, Mueller & White 1971, sec. 2.6; Vallado 2013, alg. 10).

Elements are recovered from ``(r, v)`` through ``h = r x v``, ``n = z x h`` and the eccentricity
vector ``e = (v x h)/gm - r/|r|`` (see ``gravsim.analysis.elements``).  Degenerate cases follow
the conventions documented there: equatorial orbits take the node along +x, circular orbits take
the perihelion at the node (``argp = 0``).  The well-conditioned combinations
``varpi = raan + argp`` and ``argp + nu`` therefore survive the round trip even as ``inc -> 0``
or ``e -> 0``, while the individual angles are intrinsically ill-conditioned there.

Kepler's equation is solved with Halley's method (cubic convergence) from Danby's starting guess
``E0 = M + 0.85 e sign(sin M)`` (Danby 1988, *Fundamentals of Celestial Mechanics*, sec. 6.6),
which converges for every ``0 <= e < 1`` and every ``M``.  Each sample stops when its correction
falls below ``tol`` or when its residual ``E - e sin E - M`` reaches the rounding floor of that
cancellation (a few ulps of ``max(|E|, |M|)``).  The second rule is what terminates near-parabolic
orbits close to perihelion (``e -> 1``, ``M -> 0``): there ``dE/dM = 1/(1 - e cos E)`` is huge,
the Halley step at the floor is ``noise / (1 - e cos E)`` (far above any absolute ``tol``) and
``E`` is intrinsically determined only to ``~ eps |E| / (1 - e cos E)``.

Angles are radians.  Outputs of ``state_to_elements`` are wrapped to ``(-pi, pi]``; the solution
``E`` of Kepler's equation keeps the branch of its ``M`` (``E - M`` stays ``O(e)``).  All routines
broadcast over leading axes (inputs ``(..., 3)`` give outputs ``(...)``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from gravsim.analysis.elements import (
    _dot,
    _norm,
    _unit,
    lrl_vector,
    node_vector,
    specific_h,
    wrap_angle,
)

__all__ = [
    "solve_kepler",
    "elements_to_state",
    "state_to_elements",
    "propagate",
    "mean_motion",
]

ECC_TOL = 1e-12
"""Eccentricity below which an orbit is treated as circular (perihelion taken at the node)."""

_TWO_PI = 2.0 * np.pi
_EPS = np.finfo(np.float64).eps

_RESIDUAL_ULPS = 4.0
"""Kepler residuals within this many ulps of ``max(|E|, |M|)`` are rounding noise."""

_STALL_TOL = 1e-9
"""A Halley correction that stops shrinking while already below this is bouncing on that floor."""


def _scalarize(x: np.ndarray) -> np.ndarray:
    """Return 0-d arrays as numpy scalars, leave other arrays untouched."""
    x = np.asarray(x)
    return x[()] if x.ndim == 0 else x


def _check_elliptic(a: np.ndarray, e: np.ndarray) -> None:
    if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("only bound (elliptic) orbits are supported: need a > 0")
    if np.any(e < 0.0) or np.any(e >= 1.0):
        raise ValueError("only bound (elliptic) orbits are supported: need 0 <= e < 1")


def mean_motion(a: ArrayLike, gm: ArrayLike) -> np.ndarray:
    """Mean motion ``n = sqrt(gm / a^3)`` in rad/day."""
    a = np.asarray(a, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)
    return np.sqrt(gm / a**3)


# --------------------------------------------------------------------------- Kepler's equation
def solve_kepler(
    M: ArrayLike, e: ArrayLike, tol: float = 1e-14, max_iter: int = 50
) -> np.ndarray:
    """Solve Kepler's equation ``M = E - e sin E`` for the eccentric anomaly ``E``.

    Halley iteration from Danby's starting guess; converges in a handful of iterations for any
    ``0 <= e < 1`` and any ``M``.  ``M`` is first reduced to ``[-pi, pi]`` and the multiple of
    ``2 pi`` is restored on ``E``, so ``E - M = e sin E`` holds on the same branch.

    Every sample stops on its own, either when its correction ``|dE|`` drops below ``tol`` or
    when its residual ``f = E - e sin E - M`` reaches the rounding floor of that cancellation
    (``_RESIDUAL_ULPS`` ulps of ``max(|E|, |M|)``), past which the Halley step ``-f / f'`` is
    noise.  The second rule matters for near-parabolic orbits close to perihelion (``e -> 1``,
    ``M -> 0``): ``f' = 1 - e cos E`` is tiny there, the noise step exceeds any absolute ``tol``,
    and ``E`` itself is only determined to ``~ eps |E| / f'`` (the problem's conditioning).

    Parameters
    ----------
    M : array_like
        Mean anomaly [rad], any finite value.
    e : array_like
        Eccentricity, ``0 <= e < 1``; broadcasts against ``M``.
    tol : float, optional
        Convergence threshold on the last correction ``|dE|`` (default 1e-14).
    max_iter : int, optional
        Maximum number of iterations (default 50; the worst corner, ``e = 1 - 1e-15`` with
        ``|M| ~ 1e-300``, needs about 30).

    Returns
    -------
    ndarray
        Eccentric anomaly ``E`` [rad], broadcast shape of ``M`` and ``e``.

    Raises
    ------
    ValueError
        If ``e`` is outside ``[0, 1)`` or any input is not finite.
    RuntimeError
        If some sample has not converged after ``max_iter`` iterations (not expected for valid
        inputs).
    """
    M = np.asarray(M, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    if np.any(e < 0.0) or np.any(e >= 1.0):
        raise ValueError("solve_kepler requires 0 <= e < 1")
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(e))):
        raise ValueError("solve_kepler requires finite M and e")
    M, e = np.broadcast_arrays(M, e)
    k = np.round(M / _TWO_PI)
    Mr = M - k * _TWO_PI  # in [-pi, pi]
    E = Mr + 0.85 * e * np.sign(np.sin(Mr))
    active = np.ones(E.shape, dtype=bool)
    d_prev = np.full(E.shape, np.inf)
    for _ in range(max_iter):
        s, c = np.sin(E), np.cos(E)
        f = E - e * s - Mr
        fp = 1.0 - e * c
        fpp = e * s
        d_newton = -f / fp
        d_halley = -f / (fp + 0.5 * d_newton * fpp)
        d_abs = np.abs(d_halley)
        # f is the cancellation of E against e sin E: below a few ulps of the operands it is
        # rounding noise, and so is the step it implies (noise / fp, large when fp ~ 1 - e).
        at_floor = np.abs(f) <= _RESIDUAL_ULPS * _EPS * np.maximum(np.abs(E), np.abs(Mr))
        # Safety net for a libm whose sin is noisier than assumed above: a tiny correction that
        # has stopped shrinking is bouncing on that floor as well.
        stalled = (d_abs >= d_prev) & (d_prev < _STALL_TOL)
        noise = at_floor | stalled
        E = np.where(active & ~noise, E + d_halley, E)
        active &= ~noise & (d_abs > tol)
        if not np.any(active):
            break
        d_prev = np.where(active, d_abs, np.inf)
    else:
        raise RuntimeError("solve_kepler did not converge")
    return _scalarize(E + k * _TWO_PI)


def _true_from_eccentric(E: np.ndarray, e: np.ndarray) -> np.ndarray:
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E), np.sqrt(1.0 - e) * np.cos(0.5 * E))


def _eccentric_from_true(nu: np.ndarray, e: np.ndarray) -> np.ndarray:
    return 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(0.5 * nu), np.sqrt(1.0 + e) * np.cos(0.5 * nu)
    )


# --------------------------------------------------------------------------- elements -> state
def elements_to_state(
    a: ArrayLike,
    e: ArrayLike,
    inc: ArrayLike,
    raan: ArrayLike,
    argp: ArrayLike,
    nu: ArrayLike,
    gm: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity from classical orbital elements.

    Parameters
    ----------
    a : array_like
        Semi-major axis [AU], ``a > 0``.
    e : array_like
        Eccentricity, ``0 <= e < 1``.
    inc, raan, argp, nu : array_like
        Inclination, longitude of ascending node, argument of perihelion, true anomaly [rad].
    gm : array_like
        Gravitational parameter G (M + m) [AU^3/day^2].

    Returns
    -------
    r, v : ndarray, shape (..., 3)
        Relative position [AU] and velocity [AU/day] in the frame the elements refer to.
        All element arguments broadcast together to the leading shape ``(...)``.
    """
    a, e, inc, raan, argp, nu, gm = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (a, e, inc, raan, argp, nu, gm))
    )
    _check_elliptic(a, e)
    p = a * (1.0 - e * e)
    cnu, snu = np.cos(nu), np.sin(nu)
    rmag = p / (1.0 + e * cnu)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)
    r = rmag[..., None] * (cnu[..., None] * P + snu[..., None] * Q)
    v = np.sqrt(gm / p)[..., None] * (-snu[..., None] * P + (e + cnu)[..., None] * Q)
    return r, v


# --------------------------------------------------------------------------- state -> elements
def state_to_elements(r: ArrayLike, v: ArrayLike, gm: ArrayLike) -> dict[str, np.ndarray]:
    """Classical osculating elements from a relative state (elliptic orbits only).

    Parameters
    ----------
    r, v : array_like, shape (..., 3)
        Relative position [AU] and velocity [AU/day].
    gm : array_like, scalar or shape (...)
        Gravitational parameter G (M + m) [AU^3/day^2].

    Returns
    -------
    dict[str, ndarray]
        Keys ``a`` [AU], ``e``, ``inc``, ``raan``, ``argp``, ``nu``, ``M`` [rad], ``period``
        [day] and ``varpi = raan + argp`` [rad]; every array has the leading shape ``(...)``.
        Angles are wrapped to ``(-pi, pi]``.  ``inc`` is in ``[0, pi]``.

    Notes
    -----
    Equatorial orbits (``inc = 0`` or ``pi``) take the node along +x (``raan = 0``); circular
    orbits (``e < ECC_TOL``) take the perihelion at the node (``argp = 0``, ``nu`` = argument of
    latitude).  The combinations ``raan + argp`` and ``argp + nu`` are always well conditioned.

    Raises
    ------
    ValueError
        If any sample is unbound (``a <= 0`` or ``e >= 1``).
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)

    rmag = _norm(r)
    v2 = _dot(v, v)
    h = specific_h(r, v)
    h_hat = _unit(h)
    n_hat = _unit(node_vector(h))
    evec = lrl_vector(r, v, gm)
    e = _norm(evec)
    a = 1.0 / (2.0 / rmag - v2 / gm)
    _check_elliptic(a, e)

    inc = np.arctan2(np.hypot(h[..., 0], h[..., 1]), h[..., 2])
    raan = wrap_angle(np.arctan2(n_hat[..., 1], n_hat[..., 0]))

    circular = e < ECC_TOL
    e_safe = np.where(circular, 1.0, e)
    p_hat = np.where(circular[..., None], n_hat, evec / e_safe[..., None])
    argp = wrap_angle(np.arctan2(_dot(np.cross(n_hat, p_hat), h_hat), _dot(n_hat, p_hat)))
    r_hat = r / rmag[..., None]
    nu = wrap_angle(np.arctan2(_dot(np.cross(p_hat, r_hat), h_hat), _dot(p_hat, r_hat)))

    E = _eccentric_from_true(nu, e)
    M = wrap_angle(E - e * np.sin(E))
    period = _TWO_PI / mean_motion(a, gm)
    varpi = wrap_angle(raan + argp)
    out = {
        "a": a,
        "e": e,
        "inc": inc,
        "raan": raan,
        "argp": argp,
        "nu": nu,
        "M": M,
        "period": period,
        "varpi": varpi,
    }
    return {k: _scalarize(val) for k, val in out.items()}


# --------------------------------------------------------------------------- propagation
def propagate(
    r0: ArrayLike, v0: ArrayLike, gm: ArrayLike, dt: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic two-body propagation of an elliptic orbit by ``dt``.

    The state is converted to elements, the mean anomaly advanced by ``n dt``, Kepler's equation
    solved, and the state rebuilt in the same frame.

    Parameters
    ----------
    r0, v0 : array_like, shape (..., 3)
        Initial relative position [AU] and velocity [AU/day].
    gm : array_like, scalar or shape (...)
        Gravitational parameter G (M + m) [AU^3/day^2].
    dt : array_like
        Time offset(s) [day]; broadcasts against the leading shape of ``r0`` (e.g. a single
        state with ``dt`` of shape ``(T,)`` gives ``(T, 3)`` outputs).

    Returns
    -------
    r, v : ndarray, shape (..., 3)
        Propagated relative position and velocity.
    """
    el = state_to_elements(r0, v0, gm)
    dt = np.asarray(dt, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)
    M = el["M"] + mean_motion(el["a"], gm) * dt
    E = solve_kepler(M, el["e"])
    nu = _true_from_eccentric(E, el["e"])
    return elements_to_state(el["a"], el["e"], el["inc"], el["raan"], el["argp"], nu, gm)
