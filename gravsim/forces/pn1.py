"""First post-Newtonian (1PN) correction for test bodies around one central mass.

Physics
-------
In harmonic coordinates the 1PN equation of motion of a test body around a static mass
``GM_c``, parametrised by the PPN parameters ``beta`` (nonlinearity of the superposition of
gravity) and ``gamma`` (space curvature per unit rest mass), is

    a = -GM_c r/r^3  +  (GM_c / (c^2 r^3)) * [ (2(beta+gamma) GM_c/r - gamma v^2) r
                                               + 2(1+gamma) (r.v) v ] ,

with ``r = pos_i - pos_central``, ``v = vel_i - vel_central`` and ``r = |r|``. This module
returns ONLY the bracketed correction (``make_pn1``); the Newtonian term comes from
``gravsim.forces.newton`` and ``make_newton_pn1`` provides the sum. For General Relativity
``beta = gamma = 1`` and the correction is the familiar Schwarzschild 1PN term

    a_1PN = (GM_c / (c^2 r^3)) * [ (4 GM_c/r - v^2) r + 4 (r.v) v ] .

Secular consequence: the perihelion advances by

    dphi = (2 + 2 gamma - beta)/3 * 6 pi GM_c / (c^2 a (1 - e^2))   per orbit,

i.e. 42.98"/century for Mercury in GR. The relative size of the correction is
``~ GM_c/(c^2 r) ~ v^2/c^2 ~ 1e-7`` for Mercury, so by default it is invisible to the eye;
pass a reduced ``c`` to exaggerate it (the expansion stays valid while ``v/c << 1``).

Conventions
-----------
* The correction is applied to every body except ``central`` (whose row is zero), so with the
  default solar system every planet gets the Sun's 1PN term and the Sun itself is unaffected.
  Mutual 1PN interactions between planets and the back-reaction on the Sun are *not*
  included; that is the Einstein-Infeld-Hoffmann law (``gravsim.forces.eih``, Phase 5).
* ``central`` is an integer row index (resolve names with ``SystemState.index("Sun")``).
* Units: AU, day, ``GM`` in AU^3/day^2; ``c`` in AU/day (default ``constants.C_AU_DAY``).

References
----------
* Soffel, *Relativity in Astrometry, Celestial Mechanics and Geodesy* (1989), Sec. 4.2.
* Will, "The Confrontation between General Relativity and Experiment", Living Rev. Rel. 17, 4
  (2014), Eq. (56) (PPN n-body equations, test-body limit).
* Moyer, *Formulation for Observed and Computed Values of DSN Data Types* (2003), Eq. (4-26).
* Tamayo et al., MNRAS 491, 2885 (2020): the ``gr`` effect of REBOUNDx uses the same form.
"""

from __future__ import annotations

import operator

import numpy as np

from gravsim.constants import C_AU_DAY
from gravsim.forces.newton import AccelFn, accel_newton, check_inputs

__all__ = ["make_pn1", "make_newton_pn1"]


def _check_central(central: int) -> int:
    """Validate that ``central`` is an integer index (names cannot be resolved here)."""
    if isinstance(central, str):
        raise TypeError(
            f"central must be an integer row index, got {central!r}; "
            "resolve names with SystemState.index(name)"
        )
    return operator.index(central)


def make_pn1(
    c: float = C_AU_DAY, central: int = 0, beta: float = 1.0, gamma: float = 1.0
) -> AccelFn:
    """Build the 1PN test-particle correction (correction only, no Newtonian term).

    Parameters
    ----------
    c : float, optional
        Speed of light [AU/day]. Default ``constants.C_AU_DAY``; lower it to exaggerate GR.
    central : int, optional
        Row index of the central mass whose field is treated relativistically (default 0).
    beta, gamma : float, optional
        PPN parameters; ``beta = gamma = 1`` is General Relativity.

    Returns
    -------
    AccelFn
        ``accel(pos, vel, gm, t) -> acc`` of shape (N, 3) with
        ``acc[i] = gm_c/(c^2 r^3) [ (2(beta+gamma) gm_c/r - gamma v^2) r + 2(1+gamma)(r.v) v ]``
        for ``i != central`` and ``acc[central] = 0``. ``t`` is ignored. Raises
        ``ValueError`` unless the shapes are (N, 3), (N, 3), (N,) (``newton.check_inputs``)
        and ``IndexError`` if ``central`` is out of range for N.
    """
    central = _check_central(central)
    c = float(c)
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")
    inv_c2 = 1.0 / (c * c)
    k_r = 2.0 * (float(beta) + float(gamma))  # coefficient of gm_c/r in the r term
    k_v = 2.0 * (1.0 + float(gamma))  # coefficient of (r.v) in the v term
    gamma = float(gamma)

    def accel_pn1(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0) -> np.ndarray:
        pos, vel, gm = check_inputs(pos, vel, gm)
        n = pos.shape[0]
        ic = central if central >= 0 else central + n
        if not 0 <= ic < n:
            raise IndexError(f"central index {central} out of range for {n} bodies")
        gm_c = gm[ic]

        r = pos - pos[ic]  # (N, 3) relative position
        v = vel - vel[ic]  # (N, 3) relative velocity
        r2 = np.einsum("ij,ij->i", r, r)
        r2[ic] = 1.0  # placeholder; the central row is zeroed below
        rmag = np.sqrt(r2)
        v2 = np.einsum("ij,ij->i", v, v)
        rv = np.einsum("ij,ij->i", r, v)

        pref = gm_c * inv_c2 / (r2 * rmag)  # gm_c / (c^2 r^3)
        coef_r = k_r * gm_c / rmag - gamma * v2  # 2(beta+gamma) gm_c/r - gamma v^2
        coef_v = k_v * rv  # 2(1+gamma) (r.v)
        acc = pref[:, None] * (coef_r[:, None] * r + coef_v[:, None] * v)
        acc[ic, :] = 0.0
        return np.ascontiguousarray(acc)

    accel_pn1.__name__ = "accel_pn1"
    accel_pn1.__qualname__ = "make_pn1.<locals>.accel_pn1"
    accel_pn1.params = {  # type: ignore[attr-defined]
        "law": "pn1",
        "c": c,
        "central": central,
        "beta": float(beta),
        "gamma": gamma,
    }
    return accel_pn1


def make_newton_pn1(
    c: float = C_AU_DAY, central: int = 0, beta: float = 1.0, gamma: float = 1.0
) -> AccelFn:
    """Build Newtonian N-body gravity plus the 1PN test-particle correction of ``central``.

    Convenience wrapper equal to ``forces.combine(accel_newton, make_pn1(c, central, beta,
    gamma))``. Parameters and the ``AccelFn`` signature are those of ``make_pn1``.
    """
    pn1 = make_pn1(c=c, central=central, beta=beta, gamma=gamma)

    def accel_newton_pn1(
        pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        return accel_newton(pos, vel, gm, t) + pn1(pos, vel, gm, t)

    accel_newton_pn1.params = {  # type: ignore[attr-defined]
        **pn1.params,  # type: ignore[attr-defined]
        "law": "newton+pn1",
    }
    return accel_newton_pn1
