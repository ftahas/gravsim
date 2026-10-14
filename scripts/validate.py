#!/usr/bin/env python
"""End-to-end physics validation of gravsim: Newton vs General Relativity in the solar system.

Run with the project venv::

    .venv/bin/python scripts/validate.py            # full century runs, ~2 min, figures in figures/
    .venv/bin/python scripts/validate.py --years 10 # quick look (looser physics, same code paths)

Physics being checked (ROADMAP section 4, CONTRIBUTING.md "Testing")
---------------------------------------------------------------
Every check measures the secular drift of Mercury's longitude of perihelion ``varpi`` (the
direction of the Laplace-Runge-Lenz vector, ``gravsim.analysis``) in a run produced by the
package's own force laws and integrators, and compares it with a reference:

1. **Noise floor.**  Sun + Mercury, Newtonian gravity, DOP853 (rtol 1e-12, atol 1e-15): the
   fitted rate must be ~0 (< 0.5 arcsec/century).  Fixed-step Leapfrog and Yoshida4 at a stated
   step are run too, to exhibit their spurious precession ``O((2 pi dt / P)^order)`` per orbit.
2. **1PN Mercury.**  Same, Newton + 1PN (``forces.pn1``, central = Sun).  Differencing the two
   ``varpi`` series (``analysis.compare.perihelion_rate_difference``) cancels the periodic part
   and the shared numerical error and leaves the GR advance
   ``dvarpi = 6 pi GM / (c^2 a (1 - e^2))`` per orbit = 42.98 arcsec/century.
3. **Exact Schwarzschild vs 1PN.**  The Binet orbit ``u'' + u = GM/L^2 + 3 GM u^2/c^2``
   (``metric.schwarzschild``) must agree with the 1PN formula to < 0.01 arcsec/century for
   Mercury (the 2PN remainder is of order ``(GM/(c^2 a))^2`` ~ 1e-15 relative).
4. **Exaggerated c.**  ``c -> c/100`` multiplies the advance by 1e4 (0.29 deg/orbit) while
   ``v/c`` ~ 0.013 keeps 1PN valid; the fitted rate must match the analytic formula with the
   same ``c`` to 1 percent and the orbit becomes a visible rosette.
5. **N-body Newton.**  Sun + 8 planets (Earth = EMB, outer planets = system barycenters) from
   the cached J2000 Horizons snapshot: planetary perturbations alone advance Mercury's
   perihelion by 531.6 arcsec/century (Clemence 1947: 531.63; Park et al. 2017).
6. **N-body Newton + 1PN.**  574.6 arcsec/century; the run-vs-run difference is again 42.98,
   and the differences for Venus, EMB and Mars are 8.62, 3.84 and 1.35 arcsec/century.
7. **Horizons.**  The cached DE441-based 100-year series (which contains GR) gives Mercury's
   total rate directly, and the Newton run drifts away from it by thousands of km while the
   1PN run stays within a few hundred km: Le Verrier's anomaly seen "from data".
8. **Energy.**  ``|dE/E|`` of the N-body DOP853 run < 1e-9 over a century; of a Yoshida4
   run at dt = 0.5 day < 1e-8 over ten years.
9. **Light deflection.**  ``4 GM / (c^2 R_sun) = 1.751 arcsec`` from constants alone.
10. **N-body EIH (Phase 5).**  Sun + 8 planets under the Einstein-Infeld-Hoffmann equations
    (``forces.eih``, every body relativistic, Newtonian term included): the run-vs-run
    differences against Newton are again 42.98 / 8.62 / 3.84 / 1.35 arcsec/century for
    Mercury / Venus / EMB / Mars, and the absolute Mercury fit is 574.6 (same bias as check 6).
11. **EIH vs Sun-only 1PN.**  The secular difference of Mercury's ``varpi`` between the EIH run
    and the Newton + ``pn1`` run is reported and must be below 0.02 arcsec/century: the Sun's
    barycentric motion and the planets' potentials change the 1PN advance only periodically.
12. **REBOUNDx oracle.**  The same nine bodies integrated for ten years by REBOUND (IAS15) with
    the REBOUNDx ``gr_full`` force; the maximum position difference against our EIH run must be
    below 10 km (skipped with a note when rebound/reboundx are not installed).
13. **PPN scaling (Phase 5).**  Sun + Mercury under ``make_eih(beta, gamma)`` for
    ``(beta, gamma) = (0, 1), (1, 0), (2, 1)`` and GR, ten years each: the differenced rate over
    ``(2 + 2 gamma - beta)/3 * 42.98`` must be 1 to 1 percent.
14. **Light (Phase 6).**  The exact null geodesic (``metric.null.deflection_angle``) grazing
    the Sun is bent by 1.751 arcsec (+- 0.001); Soldner's Newtonian corpuscle by half of it.
15. **Shapiro delay.**  One-way delay of an Earth (1 AU) to Mars (1.524 AU) signal grazing the
    Sun at superior conjunction, ``(1 + gamma) GM/c^3 ln[(r_e + r_r + R)/(r_e + r_r - R)]`` =
    124 microseconds (report; ~250 for the Viking round trip), and the exact null-geodesic
    transit time minus the flat light time agrees with it to 1 percent (measured ~1e-4, the
    second-order end-point and Fermat terms; see ``gravsim.metric.null``).
16. **Solar oblateness (Phase 8).**  Sun + Mercury with the quadrupole field of the Sun
    (``forces.extras.make_j2``, ``J2 = 2.2e-7``, axis ``SUN_POLE_ECLIPTIC``): the run-vs-run
    difference against Newton over the span is +0.028 +- 0.005 arcsec/century, the Kaula
    rates ``dOmega/dt = -(3/2) n J2 (R/p)^2 cos i`` and ``domega/dt = (3/4) n J2 (R/p)^2
    (5 cos^2 i - 1)`` (node and argument in the solar-equator frame, ``i`` = 3.38 deg) are
    recovered to 3 percent with ``J2 x 1000`` over ten years (measured 1e-4).
17. **Frame dragging.**  Same with ``make_lense_thirring`` (``J x 1e5``, ten years): the node
    advances at ``2 GJ / (c^2 a^3 (1 - e^2)^1.5)`` and the argument regresses three times
    faster (``-6 GJ cos i / ...``), each to 3 percent (measured 1e-4); the real-size
    difference against Newton over the span is reported (-0.002 arcsec/century).
18. **The Moon (Phase 8).**  Sun + planets + Earth + Moon (``load_j2000(with_moon=True)``) on
    the daily grid of the cached Horizons Moon series, ten years: sidereal month 27.32 +- 0.05 d
    (2 pi over the fitted mean motion of the mean longitude), perigee advance 40.7 +- 1.5 and
    node regression -19.3 +- 1.0 deg/yr (the Sun's tide; the same fits of the Horizons series
    over the same span agree to 0.3 deg/yr), the Moon within 500 km of Horizons over two years
    (Newton + 1PN; reported for Newton alone), the drift of the EMB point mass away from the real
    Earth-Moon barycentre (reported; ``2 eps n a t`` ~ 110 km/yr with
    ``eps = (3/4) w_E w_M (a_M/a_E)^2``), Mercury's GR-minus-Newton rate with the Moon present
    equal to the EMB value within 0.1 (measured 1e-7) and the Newtonian shift of its rate by the
    Moon (-0.0025, reported).

Conventions: AU, day, GM in AU^3/day^2, barycentric J2000-ecliptic frame, time in days
since J2000.0 TDB; rates in arcsec per Julian century (36525 d).  The script never touches
the network: initial conditions and the reference series come from ``gravsim.data``.

Caveat on absolute N-body rates.  An ordinary least-squares slope of the osculating ``varpi``
series over a finite span is biased by the long-period planetary terms (and by the missing
asteroids, Moon and Pluto, ~0.1 arcsec/century); the 100-year fit of the *Horizons* series by the
same method gives ~572 rather than the 574.1 observed, and our Newton run ~529 rather than
531.6.  The *differences* between runs on the same grid are free of this bias, which is why the
GR checks are differential.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gravsim.analysis import (
    element_series,
    fit_secular_rate,
    gr_precession_arcsec_per_century,
    inclination_to_axis,
    perihelion_longitude_series,
    perihelion_passages,
    perihelion_rate_difference,
    rate_arcsec_per_century,
    relative_energy_error,
    separation,
    state_to_elements,
    wrap_angle,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import (
    ARCSEC_PER_RAD,
    AU_KM,
    C_AU_DAY,
    DAY_S,
    DEG_PER_RAD,
    GJ_SUN,
    GM_SUN,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
    LIGHT_DEFLECTION_LIMB_ARCSEC,
    MERCURY_PRECESSION_GR_ARCSEC_CY,
    MERCURY_PRECESSION_J2_ARCSEC_CY,
    MERCURY_PRECESSION_LT_ARCSEC_CY,
    MERCURY_PRECESSION_NEWTON_ARCSEC_CY,
    MERCURY_PRECESSION_OBSERVED_ARCSEC_CY,
    MOON_NODE_RATE_DEG_YR,
    MOON_PERIGEE_RATE_DEG_YR,
    PLANETS,
    R_SUN_AU,
    SUN_J2,
    SUN_POLE_ECLIPTIC,
)
from gravsim.data.horizons import load_j2000, load_moon_reference_series, load_reference_series
from gravsim.forces import (
    accel_newton,
    combine,
    make_eih,
    make_j2,
    make_lense_thirring,
    make_newton_pn1,
)
from gravsim.integrators import integrate, integrate_scipy
from gravsim.metric import null
from gravsim.metric.schwarzschild import precession_per_orbit_1pn, precession_per_orbit_numeric

# Reference values of the run-vs-run 1PN advance for the inner planets [arcsec/century].
GR_RATES_ARCSEC_CY: dict[str, float] = {
    "Mercury": MERCURY_PRECESSION_GR_ARCSEC_CY,
    "Venus": 8.62,
    "EMB": 3.84,
    "Mars": 1.35,
}
NBODY_PN1_MERCURY_ARCSEC_CY = 574.6  # Newton N-body + 1PN
RTOL, ATOL = 1e-12, 1e-15  # production DOP853 tolerances
SAMPLE_STEP_DAYS = 1.0  # stored sample spacing of the DOP853 runs
FIXED_DT_DAYS = 0.5  # step of the fixed-step comparison runs
EXAGGERATION = 100.0  # c -> c / EXAGGERATION for the rosette
EXAGGERATED_ORBITS = 50
ROSETTE_ORBITS = 200  # figure-only run: 0.29 deg/orbit x 200 = 57 deg of visible advance
ROSETTE_EVERY = 20  # draw every 20th orbit of that run
MIN_ORBITS_FOR_RATE = 3.0  # planets with fewer orbits in the span are left off the rate plot
REPORT_ONLY = math.inf  # tolerance of a reported, not asserted, number
EIH_VS_PN1_TOL_ARCSEC_CY = 0.02  # secular EIH-minus-pn1 difference for Mercury (check 11)
ORACLE_YEARS = 10.0  # span of the REBOUNDx gr_full comparison (check 12)
ORACLE_SAMPLE_DAYS = 10.0  # epochs compared with the oracle
ORACLE_TOL_KM = 10.0
PPN_CASES: tuple[tuple[float, float], ...] = ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (2.0, 1.0))
PPN_YEARS = 10.0  # span of the two-body PPN runs (check 13)
SHAPIRO_R_EMIT_AU, SHAPIRO_R_RECV_AU = 1.0, 1.5237  # Earth and Mars at their mean distances
MICROSECOND_DAYS = 1e-6 / DAY_S
RAY_BUNDLE_R_SUN: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0)  # |b| of the drawn rays
RAY_AMPLIFICATION = 1e4  # the deviation from the straight line is drawn x 1e4
# Light-deflection measurements scaled to the solar limb [arcsec]: Dyson, Eddington & Davidson
# 1920 (Sobral, Principe; probable errors) and the 2004 VLBI value (Shapiro et al. 2004,
# gamma = 0.9998 +- 0.0004 -> (1 + gamma)/2 of the GR value).
DEFLECTION_MEASUREMENTS: tuple[tuple[str, float, float], ...] = (
    ("1919 Sobral", 1.98, 0.12),
    ("1919 Príncipe", 1.61, 0.30),
    ("2004 VLBI", 0.9999 * LIGHT_DEFLECTION_LIMB_ARCSEC, 0.0002 * LIGHT_DEFLECTION_LIMB_ARCSEC),
)
# Phase 8: the competing effects (checks 16-18).
J2_AMPLIFICATION = 1000.0  # J2 x 1000 for the Kaula-rate check
LT_AMPLIFICATION = 1e5  # solar spin x 1e5 for the Lense-Thirring-rate check
EXTRAS_YEARS = 10.0  # span of the amplified two-body runs
EXTRAS_RATE_TOL = 0.03  # relative tolerance on the node and argument rates vs the formulas
J2_TOL_ARCSEC_CY = 0.005  # real-size J2 advance of Mercury (0.028)
MOON_YEARS = 10.0  # span of the Earth + Moon runs (the cached Horizons Moon series)
MOON_SEPARATION_YEARS = 2.0  # window of the Moon-vs-Horizons separation check
MOON_SEPARATION_TOL_KM = 500.0
SIDEREAL_MONTH_DAYS = 27.321662
DEG_PER_YEAR = DEG_PER_RAD * JULIAN_YEAR_DAYS  # rad/day -> degrees per Julian year
ASTEROID_RING_GM = 1.2e-9 * GM_SUN  # main belt as a ring (Pitjeva & Pitjev 2018)
ASTEROID_RING_AU = 2.7


# ----------------------------------------------------------------------------- results
@dataclass(frozen=True)
class CheckResult:
    """One validation number: ``passed`` is ``|value - expected| <= tolerance``."""

    name: str
    value: float
    expected: float
    tolerance: float
    note: str = ""

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.value) and abs(self.value - self.expected) <= self.tolerance)


def format_table(results: Sequence[CheckResult]) -> str:
    """Plain-text table of the results (name, value, expected, tolerance, pass)."""
    rows = [("check", "value", "expected", "tolerance", "pass")]
    for r in results:
        tol = "report" if math.isinf(r.tolerance) else f"{r.tolerance:.4g}"
        rows.append(
            (r.name, f"{r.value:.6g}", f"{r.expected:.6g}", tol, "ok" if r.passed else "FAIL")
        )
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    lines = []
    for k, row in enumerate(rows):
        cells = [row[0].ljust(widths[0])] + [
            c.rjust(w) for c, w in zip(row[1:], widths[1:], strict=True)
        ]
        lines.append("  ".join(cells))
        if k == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- runs
def two_body_state() -> SystemState:
    """Sun + Mercury from the J2000 snapshot, moved to their own barycenter."""
    return load_j2000().subset(("Sun", "Mercury")).to_barycentric()


def daily_grid(t0: float, years: float, step: float = SAMPLE_STEP_DAYS) -> np.ndarray:
    """Output times ``t0, t0 + step, ...`` covering ``years`` Julian years."""
    n = int(round(years * JULIAN_YEAR_DAYS / step))
    return t0 + step * np.arange(n + 1)


def run_dop853(state: SystemState, accel: Callable, t_eval: np.ndarray) -> Trajectory:
    """Production run: DOP853 at rtol 1e-12, atol 1e-15, samples at ``t_eval``."""
    return integrate_scipy(state, accel, t_eval, method="DOP853", rtol=RTOL, atol=ATOL)


def osculating_elements(state: SystemState, body: str, central: str = "Sun") -> dict[str, float]:
    """Osculating Kepler elements of ``body`` about ``central`` (``gm = gm_c + gm_body``)."""
    r, v = state.relative_to(central)
    i = state.index(body)
    gm = float(state.gm[state.index(central)] + state.gm[i])
    el = state_to_elements(r[i], v[i], gm)
    return {k: float(el[k]) for k in ("a", "e", "period")} | {"gm": gm}


def dense_rate(traj: Trajectory, body: str, central: str = "Sun") -> tuple[float, float]:
    """Secular rate of the osculating ``varpi`` series [arcsec/century] and its OLS error."""
    t, varpi = perihelion_longitude_series(traj, body, central)
    fit = fit_secular_rate(t, varpi)
    return float(fit.rate_arcsec_per_century), float(fit.rate_err_arcsec_per_century)


def passage_rate(traj: Trajectory, body: str, central: str = "Sun") -> tuple[float, float]:
    """Secular rate of ``varpi`` sampled at perihelion passages [arcsec/century]."""
    t_p, varpi_p = perihelion_passages(traj, body, central)
    fit = fit_secular_rate(t_p, varpi_p)
    return float(fit.rate_arcsec_per_century), float(fit.rate_err_arcsec_per_century)


def on_common_grid(traj: Trajectory, t_ref: np.ndarray) -> Trajectory:
    """Sub-sample ``traj`` at the times ``t_ref`` (which must all be on its grid)."""
    idx = np.searchsorted(traj.t, t_ref)
    if idx.max() >= traj.t.size or not np.allclose(traj.t[idx], t_ref, rtol=0.0, atol=1e-9):
        raise ValueError("reference epochs are not on the trajectory grid")
    return Trajectory(
        traj.t[idx], traj.pos[idx], traj.vel[idx], traj.gm, traj.names, dict(traj.meta)
    )


def truncate(traj: Trajectory, t_max: float) -> Trajectory:
    """Samples of ``traj`` with ``t <= t_max``."""
    m = traj.t <= t_max + 1e-9
    return Trajectory(traj.t[m], traj.pos[m], traj.vel[m], traj.gm, traj.names, dict(traj.meta))


def rate_per_orbit_to_per_century(rad_per_orbit: float, period_days: float) -> float:
    """Convert an advance in rad/orbit to arcsec per Julian century."""
    return rad_per_orbit * ARCSEC_PER_RAD * JULIAN_CENTURY_DAYS / period_days


def equator_frame(pole: Sequence[float] = SUN_POLE_ECLIPTIC) -> np.ndarray:
    """Rotation matrix (rows ``x, y, z``) of the frame whose ``z`` axis is ``pole``.

    ``x`` points to the ascending node of the equator on the ecliptic (``z_ecl x pole``), so
    the node and argument of an orbit in the rotated frame are the ones the Kaula and
    Lense-Thirring formulas refer to; coordinates transform as ``r_eq = R @ r``.
    """
    z = np.asarray(pole, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def rotate_trajectory(traj: Trajectory, rotation: np.ndarray) -> Trajectory:
    """The same trajectory with every position and velocity rotated by ``rotation``."""
    return Trajectory(
        traj.t, traj.pos @ rotation.T, traj.vel @ rotation.T, traj.gm, traj.names, dict(traj.meta)
    )


def equatorial_rate_differences(
    traj: Trajectory, newton: Trajectory, body: str = "Mercury"
) -> dict[str, tuple[np.ndarray, np.ndarray, float, float]]:
    """Differenced node and argument of ``body`` in the solar-equator frame, with their fits.

    Returns ``{"raan": (t, d_angle, rate, err), "argp": ...}`` with the wrapped difference
    ``angle(traj) - angle(newton)`` [rad] and the fitted rate in arcsec per Julian century.
    """
    rotation = equator_frame()
    traj_eq, newton_eq = rotate_trajectory(traj, rotation), rotate_trajectory(newton, rotation)
    out = {}
    for key in ("raan", "argp"):
        t, x_run = element_series(traj_eq, body, key)
        _, x_newton = element_series(newton_eq, body, key)
        diff = wrap_angle(x_run - x_newton)
        fit = fit_secular_rate(t, diff)
        out[key] = (
            t,
            diff,
            float(fit.rate_arcsec_per_century),
            float(fit.rate_err_arcsec_per_century),
        )
    return out


def kaula_rates(a: float, e: float, inc: float, gm: float, j2: float) -> tuple[float, float]:
    """First-order ``(dOmega/dt, domega/dt)`` [rad/day] of the node and argument from ``j2``."""
    n_mean = math.sqrt(gm / a**3)
    k = n_mean * j2 * (R_SUN_AU / (a * (1.0 - e * e))) ** 2
    return -1.5 * k * math.cos(inc), 0.75 * k * (5.0 * math.cos(inc) ** 2 - 1.0)


def lense_thirring_rates(a: float, e: float, inc: float, gj: float) -> tuple[float, float]:
    """``(dOmega/dt, domega/dt)`` [rad/day] of the node and argument from a spin ``GJ``."""
    k = gj / (C_AU_DAY**2 * a**3 * (1.0 - e * e) ** 1.5)
    return 2.0 * k, -6.0 * k * math.cos(inc)


def ring_precession_rate(gm_ring: float, a_ring: float, a: float, gm: float = GM_SUN) -> float:
    """Secular advance ``(3/4) n (gm_ring/gm) (a/a_ring)^3`` [arcsec/cy] due to an outer ring."""
    n_mean = math.sqrt(gm / a**3)
    return float(rate_arcsec_per_century(0.75 * n_mean * (gm_ring / gm) * (a / a_ring) ** 3))


def lunar_fits(traj: Trajectory) -> dict[str, float]:
    """Sidereal month [d], perigee and node rates [deg/yr] of the Moon about the Earth."""
    r, v = traj.relative("Moon", "Earth")
    gm = float(traj.gm[traj.index("Earth")] + traj.gm[traj.index("Moon")])
    el = state_to_elements(r, v, gm)
    n_sidereal = fit_secular_rate(traj.t, el["varpi"] + el["M"]).rate
    fit_varpi = fit_secular_rate(traj.t, el["varpi"])
    fit_raan = fit_secular_rate(traj.t, el["raan"])
    return {
        "month": 2.0 * math.pi / n_sidereal,
        "perigee": fit_varpi.rate * DEG_PER_YEAR,
        "perigee_err": fit_varpi.rate_err * DEG_PER_YEAR,
        "node": fit_raan.rate * DEG_PER_YEAR,
        "node_err": fit_raan.rate_err * DEG_PER_YEAR,
        "a_km": float(np.mean(el["a"])) * AU_KM,
        "e": float(np.mean(el["e"])),
        "inc_deg": float(np.mean(el["inc"])) * DEG_PER_RAD,
    }


def earth_moon_barycentre(traj: Trajectory) -> np.ndarray:
    """Mass-weighted Earth + Moon position along a trajectory, shape (T, 3)."""
    i_e, i_m = traj.index("Earth"), traj.index("Moon")
    w = traj.gm[[i_e, i_m]] / traj.gm[[i_e, i_m]].sum()
    return w[0] * traj.pos[:, i_e] + w[1] * traj.pos[:, i_m]


def predicted_emb_drift_km(state: SystemState, span_days: float) -> tuple[float, float]:
    """``(eps, 2 eps n a span)`` [-, km]: the along-track drift of the EMB point mass."""
    i_e, i_m = state.index("Earth"), state.index("Moon")
    w = state.gm[[i_e, i_m]] / state.gm[[i_e, i_m]].sum()
    r, v = state.relative_to("Sun")
    a_e = float(state_to_elements(r[i_e], v[i_e], GM_SUN + state.gm[i_e])["a"])
    r_m, v_m = state.relative_to("Earth")
    a_m = float(state_to_elements(r_m[i_m], v_m[i_m], state.gm[i_e] + state.gm[i_m])["a"])
    eps = 0.75 * w[0] * w[1] * (a_m / a_e) ** 2
    n_e = math.sqrt((GM_SUN + state.gm[i_e]) / a_e**3)
    return float(eps), float(2.0 * eps * n_e * span_days * a_e * AU_KM)


# ----------------------------------------------------------------------------- the checks
@dataclass
class Runs:
    """Trajectories shared between checks and figures (filled by ``run_checks``)."""

    two_body_newton: Trajectory | None = None
    two_body_pn1: Trajectory | None = None
    exaggerated: Trajectory | None = None
    nbody_newton: Trajectory | None = None
    nbody_pn1: Trajectory | None = None
    yoshida4: Trajectory | None = None
    horizons: Trajectory | None = None
    planet_rates: dict[str, tuple[float, float]] | None = None  # 1PN-minus-Newton, N-body
    nbody_eih: Trajectory | None = None
    eih_planet_rates: dict[str, tuple[float, float]] | None = None  # EIH-minus-Newton, N-body
    ppn_newton: Trajectory | None = None  # Sun + Mercury, Newton, PPN_YEARS
    ppn_two_body: dict[tuple[float, float], Trajectory] | None = None  # (beta, gamma) -> EIH run
    extras_newton: Trajectory | None = None  # Sun + Mercury, Newton, EXTRAS_YEARS
    extras_amplified: dict[str, Trajectory] | None = None  # "j2" (x1000), "lt" (x1e5) runs
    extras_rates: dict[str, float] | None = None  # real-size J2 and LT advances of Mercury
    moon_newton: Trajectory | None = None  # Sun + planets + Earth + Moon, Newton, MOON_YEARS
    moon_gr: Trajectory | None = None  # same with the Sun's 1PN term
    moon_horizons: Trajectory | None = None  # the cached Moon series over the same span
    moon_fits: dict[str, float] | None = None  # lunar_fits of the Newton run
    moon_shift: float = math.nan  # Newton(Earth + Moon) - Newton(EMB), Mercury [arcsec/cy]


def check_two_body(years: float, runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Checks 1 and 2: noise floor and the 1PN advance in the Sun-Mercury problem."""
    state = two_body_state()
    el = osculating_elements(state, "Mercury")
    t_eval = daily_grid(state.t, years)
    sun = state.index("Sun")

    tic = time.perf_counter()
    newton = run_dop853(state, accel_newton, t_eval)
    pn1 = run_dop853(state, make_newton_pn1(c=C_AU_DAY, central=sun), t_eval)
    log(
        f"two-body DOP853 runs ({years:g} yr): {time.perf_counter() - tic:.1f} s, "
        f"nfev {newton.meta['nfev']} / {pn1.meta['nfev']}"
    )
    runs.two_body_newton, runs.two_body_pn1 = newton, pn1

    out: list[CheckResult] = []
    floor, floor_err = dense_rate(newton, "Mercury")
    out.append(
        CheckResult(
            "2body_newton_noise_floor_dop853",
            floor,
            0.0,
            0.5,
            f"arcsec/cy, OLS err {floor_err:.2g}, {years:g} yr",
        )
    )
    floor_p, _ = passage_rate(newton, "Mercury")
    out.append(
        CheckResult(
            "2body_newton_noise_floor_dop853_passages",
            floor_p,
            0.0,
            0.5,
            "arcsec/cy, from perihelion passages",
        )
    )

    # Fixed-step integrators over ten years: their spurious precession is the lesson's point.
    t_fixed = state.t + 10.0 * JULIAN_YEAR_DAYS
    every = int(round(SAMPLE_STEP_DAYS / FIXED_DT_DAYS))
    for name in ("leapfrog", "yoshida4"):
        traj = integrate(state, accel_newton, t_fixed, FIXED_DT_DAYS, name, save_every=every)
        rate, _ = dense_rate(traj, "Mercury")
        out.append(
            CheckResult(
                f"2body_newton_spurious_precession_{name}_dt{FIXED_DT_DAYS:g}d",
                rate,
                0.0,
                REPORT_ONLY,
                f"arcsec/cy, 10 yr, dt = {traj.meta['dt']:.3g} d (report only)",
            )
        )

    gr_rate, gr_err = perihelion_rate_difference(pn1, newton, "Mercury")
    out.append(
        CheckResult(
            "2body_pn1_minus_newton_mercury",
            gr_rate,
            MERCURY_PRECESSION_GR_ARCSEC_CY,
            0.10,
            f"arcsec/cy, OLS err {gr_err:.2g}",
        )
    )
    pn1_abs, _ = dense_rate(pn1, "Mercury")
    out.append(
        CheckResult(
            "2body_pn1_mercury_absolute_fit",
            pn1_abs,
            MERCURY_PRECESSION_GR_ARCSEC_CY,
            0.10,
            "arcsec/cy, undifferenced",
        )
    )
    analytic = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"], c=C_AU_DAY))
    out.append(
        CheckResult(
            "2body_pn1_analytic_osculating_elements",
            analytic,
            MERCURY_PRECESSION_GR_ARCSEC_CY,
            0.10,
            f"arcsec/cy, a={el['a']:.5f} AU e={el['e']:.5f}",
        )
    )
    out.append(
        CheckResult("2body_pn1_fit_minus_analytic", gr_rate - analytic, 0.0, 0.10, "arcsec/cy")
    )
    return out


def check_schwarzschild(log: Callable[[str], None]) -> list[CheckResult]:
    """Check 3: exact Schwarzschild (Binet) orbit against the 1PN formula for Mercury."""
    el = osculating_elements(two_body_state(), "Mercury")
    tic = time.perf_counter()
    exact = precession_per_orbit_numeric(el["a"], el["e"], el["gm"], C_AU_DAY, n_orbits=50)
    log(f"Schwarzschild Binet orbit (50 orbits): {time.perf_counter() - tic:.1f} s")
    pn1 = precession_per_orbit_1pn(el["a"], el["e"], el["gm"], C_AU_DAY)
    exact_cy = rate_per_orbit_to_per_century(exact, el["period"])
    pn1_cy = rate_per_orbit_to_per_century(pn1, el["period"])
    return [
        CheckResult(
            "schwarzschild_exact_mercury",
            exact_cy,
            MERCURY_PRECESSION_GR_ARCSEC_CY,
            0.10,
            "arcsec/cy, Binet orbit",
        ),
        CheckResult("schwarzschild_exact_minus_1pn", exact_cy - pn1_cy, 0.0, 0.01, "arcsec/cy"),
    ]


def check_exaggerated_c(runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Check 4: c/100 rosette, fitted rate against the analytic 1PN rate with the same c."""
    state = two_body_state()
    el = osculating_elements(state, "Mercury")
    c = C_AU_DAY / EXAGGERATION
    t_end = EXAGGERATED_ORBITS * el["period"]
    t_eval = state.t + np.linspace(0.0, t_end, int(round(t_end / 0.25)) + 1)
    tic = time.perf_counter()
    pn1 = run_dop853(state, make_newton_pn1(c=c, central=state.index("Sun")), t_eval)
    newton = run_dop853(state, accel_newton, t_eval)
    log(f"exaggerated-c runs ({EXAGGERATED_ORBITS} orbits): {time.perf_counter() - tic:.1f} s")
    runs.exaggerated = pn1
    predicted = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"], c=c))
    fitted, _ = dense_rate(pn1, "Mercury")
    differenced, _ = perihelion_rate_difference(pn1, newton, "Mercury")
    passages, _ = passage_rate(pn1, "Mercury")
    v_over_c = float(np.linalg.norm(state.relative_to("Sun")[1][1]) / c)
    per_orbit_deg = predicted / (JULIAN_CENTURY_DAYS / el["period"]) / 3600.0
    return [
        CheckResult(
            "exaggerated_c_fit_over_analytic",
            fitted / predicted,
            1.0,
            0.01,
            f"ratio; analytic {predicted:.0f} arcsec/cy = {per_orbit_deg:.3f} deg/orbit, "
            f"v/c = {v_over_c:.3f}",
        ),
        CheckResult(
            "exaggerated_c_differenced_over_analytic",
            differenced / predicted,
            1.0,
            0.01,
            "ratio, 1PN minus Newton on the same grid",
        ),
        CheckResult(
            "exaggerated_c_passages_over_analytic",
            passages / predicted,
            1.0,
            0.01,
            "ratio, from perihelion passages",
        ),
    ]


def check_nbody(years: float, runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Checks 5, 6 and 8: Sun + 8 planets, Newton and Newton + 1PN, DOP853 and Yoshida4."""
    state = load_j2000()
    sun = state.index("Sun")
    t_eval = daily_grid(state.t, years)
    tic = time.perf_counter()
    newton = run_dop853(state, accel_newton, t_eval)
    log(
        f"N-body Newton DOP853 ({years:g} yr): {time.perf_counter() - tic:.1f} s, "
        f"nfev {newton.meta['nfev']}"
    )
    tic = time.perf_counter()
    pn1 = run_dop853(state, make_newton_pn1(c=C_AU_DAY, central=sun), t_eval)
    log(
        f"N-body Newton+1PN DOP853 ({years:g} yr): {time.perf_counter() - tic:.1f} s, "
        f"nfev {pn1.meta['nfev']}"
    )
    runs.nbody_newton, runs.nbody_pn1 = newton, pn1

    out: list[CheckResult] = []
    rate_n, err_n = dense_rate(newton, "Mercury")
    out.append(
        CheckResult(
            "nbody_newton_mercury_dense_fit",
            rate_n,
            MERCURY_PRECESSION_NEWTON_ARCSEC_CY,
            3.0,
            f"arcsec/cy, OLS err {err_n:.2g}",
        )
    )
    rate_np, _ = passage_rate(newton, "Mercury")
    out.append(
        CheckResult(
            "nbody_newton_mercury_passages_fit",
            rate_np,
            MERCURY_PRECESSION_NEWTON_ARCSEC_CY,
            3.0,
            "arcsec/cy",
        )
    )
    rate_g, err_g = dense_rate(pn1, "Mercury")
    out.append(
        CheckResult(
            "nbody_pn1_mercury_dense_fit",
            rate_g,
            NBODY_PN1_MERCURY_ARCSEC_CY,
            3.0,
            f"arcsec/cy, OLS err {err_g:.2g}",
        )
    )
    rate_gp, _ = passage_rate(pn1, "Mercury")
    out.append(
        CheckResult(
            "nbody_pn1_mercury_passages_fit", rate_gp, NBODY_PN1_MERCURY_ARCSEC_CY, 3.0, "arcsec/cy"
        )
    )

    runs.planet_rates = {}
    for body in PLANETS:
        rate, err = perihelion_rate_difference(pn1, newton, body)
        runs.planet_rates[body] = (rate, err)
        if body in GR_RATES_ARCSEC_CY:
            out.append(
                CheckResult(
                    f"nbody_pn1_minus_newton_{body.lower()}",
                    rate,
                    GR_RATES_ARCSEC_CY[body],
                    0.10,
                    f"arcsec/cy, OLS err {err:.2g}",
                )
            )

    # check 8: energy
    de_dop = float(np.max(np.abs(relative_energy_error(newton))))
    out.append(
        CheckResult("energy_rel_error_nbody_dop853", de_dop, 0.0, 1e-9, f"max |dE/E|, {years:g} yr")
    )
    tic = time.perf_counter()
    every = int(round(SAMPLE_STEP_DAYS / FIXED_DT_DAYS))
    yoshida = integrate(
        state,
        accel_newton,
        state.t + 10.0 * JULIAN_YEAR_DAYS,
        FIXED_DT_DAYS,
        "yoshida4",
        save_every=every,
    )
    log(f"N-body Yoshida4 dt={FIXED_DT_DAYS} d (10 yr): {time.perf_counter() - tic:.1f} s")
    runs.yoshida4 = yoshida
    de_y = float(np.max(np.abs(relative_energy_error(yoshida))))
    out.append(
        CheckResult("energy_rel_error_nbody_yoshida4_dt0.5d_10yr", de_y, 0.0, 1e-8, "max |dE/E|")
    )
    return out


def check_horizons(years: float, runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Check 7: Mercury's rate in the Horizons series and run-vs-Horizons separations."""
    if runs.nbody_newton is None or runs.nbody_pn1 is None:
        raise RuntimeError("check_horizons needs the N-body runs of check_nbody")
    ref = load_reference_series()
    runs.horizons = ref
    out: list[CheckResult] = []
    rate_h, err_h = dense_rate(ref, "Mercury")
    out.append(
        CheckResult(
            "horizons_mercury_dense_fit_100yr",
            rate_h,
            MERCURY_PRECESSION_OBSERVED_ARCSEC_CY,
            5.0,
            f"arcsec/cy, OLS err {err_h:.2g}, 20-day samples",
        )
    )
    rate_hp, _ = passage_rate(ref, "Mercury")
    out.append(
        CheckResult(
            "horizons_mercury_passages_fit_100yr",
            rate_hp,
            MERCURY_PRECESSION_OBSERVED_ARCSEC_CY,
            5.0,
            "arcsec/cy",
        )
    )

    t_max = runs.nbody_newton.t[-1]
    ref_span = truncate(ref, t_max)
    newton = on_common_grid(runs.nbody_newton, ref_span.t)
    pn1 = on_common_grid(runs.nbody_pn1, ref_span.t)
    sep_n = separation(newton, ref_span, "Mercury", t_atol=1e-9) * AU_KM
    sep_g = separation(pn1, ref_span, "Mercury", t_atol=1e-9) * AU_KM
    log(f"Horizons comparison over {years:g} yr at {ref_span.t.size} common epochs")
    out.append(
        CheckResult(
            "horizons_vs_newton_mercury_mean_km",
            float(sep_n.mean()),
            0.0,
            REPORT_ONLY,
            "km (report only)",
        )
    )
    out.append(
        CheckResult(
            "horizons_vs_newton_mercury_final_km",
            float(sep_n[-1]),
            0.0,
            REPORT_ONLY,
            "km (report only)",
        )
    )
    out.append(
        CheckResult(
            "horizons_vs_pn1_mercury_mean_km",
            float(sep_g.mean()),
            0.0,
            REPORT_ONLY,
            "km (report only)",
        )
    )
    out.append(
        CheckResult(
            "horizons_vs_pn1_mercury_final_km",
            float(sep_g[-1]),
            0.0,
            REPORT_ONLY,
            "km (report only)",
        )
    )
    out.append(
        CheckResult(
            "horizons_pn1_over_newton_final_separation",
            float(sep_g[-1] / sep_n[-1]),
            0.0,
            1.0,
            "ratio < 1: 1PN closer to Horizons than Newton",
        )
    )
    return out


def check_eih(years: float, runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Checks 10 and 11: the EIH N-body run against Newton and against Newton + pn1."""
    if runs.nbody_newton is None or runs.nbody_pn1 is None:
        raise RuntimeError("check_eih needs the N-body runs of check_nbody")
    state = load_j2000()
    newton, pn1 = runs.nbody_newton, runs.nbody_pn1
    tic = time.perf_counter()
    eih = run_dop853(state, make_eih(c=C_AU_DAY), newton.t)
    log(
        f"N-body EIH DOP853 ({years:g} yr): {time.perf_counter() - tic:.1f} s, "
        f"nfev {eih.meta['nfev']}"
    )
    runs.nbody_eih = eih

    out: list[CheckResult] = []
    rate_e, err_e = dense_rate(eih, "Mercury")
    out.append(
        CheckResult(
            "nbody_eih_mercury_dense_fit",
            rate_e,
            NBODY_PN1_MERCURY_ARCSEC_CY,
            3.0,
            f"arcsec/cy, OLS err {err_e:.2g}",
        )
    )
    runs.eih_planet_rates = {}
    for body in PLANETS:
        rate, err = perihelion_rate_difference(eih, newton, body)
        runs.eih_planet_rates[body] = (rate, err)
        if body in GR_RATES_ARCSEC_CY:
            out.append(
                CheckResult(
                    f"nbody_eih_minus_newton_{body.lower()}",
                    rate,
                    GR_RATES_ARCSEC_CY[body],
                    0.10,
                    f"arcsec/cy, OLS err {err:.2g}",
                )
            )
    rate_d, err_d = perihelion_rate_difference(eih, pn1, "Mercury")
    out.append(
        CheckResult(
            "nbody_eih_minus_pn1_mercury",
            rate_d,
            0.0,
            EIH_VS_PN1_TOL_ARCSEC_CY,
            f"arcsec/cy, OLS err {err_d:.2g}: Sun-only 1PN already has the secular term",
        )
    )
    sep = separation(eih, pn1, "Mercury") * AU_KM
    out.append(
        CheckResult(
            "nbody_eih_vs_pn1_mercury_max_km",
            float(sep.max()),
            0.0,
            REPORT_ONLY,
            f"km over {years:g} yr (report only)",
        )
    )
    return out


def rebound_gr_full_positions(state: SystemState, t_eval: np.ndarray, c: float) -> np.ndarray:
    """Positions (T, N, 3) of ``state`` integrated by REBOUND (IAS15) + REBOUNDx ``gr_full``.

    Raises ``ImportError`` when rebound or reboundx is not installed.  ``gr_full`` implements
    the same EIH equations (``beta = gamma = 1``) in the barycentric frame, which is the frame
    of the cached snapshot.
    """
    import rebound
    import reboundx

    sim = rebound.Simulation()
    sim.G = 1.0
    sim.t = float(state.t)
    sim.integrator = "ias15"
    for i in range(state.n):
        x, y, z = (float(q) for q in state.pos[i])
        vx, vy, vz = (float(q) for q in state.vel[i])
        sim.add(m=float(state.gm[i]), x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    rebx = reboundx.Extras(sim)
    gr_full = rebx.load_force("gr_full")
    rebx.add_force(gr_full)
    gr_full.params["c"] = float(c)
    out = [np.array([[p.x, p.y, p.z] for p in sim.particles])]
    for t in t_eval[1:]:
        sim.integrate(float(t))
        out.append(np.array([[p.x, p.y, p.z] for p in sim.particles]))
    del rebx, gr_full  # must outlive the integration
    return np.array(out)


def check_rebound_oracle(runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Check 12: ten years of REBOUNDx ``gr_full`` against the EIH run (skipped without rebound)."""
    if runs.nbody_eih is None:
        raise RuntimeError("check_rebound_oracle needs the EIH run of check_eih")
    eih = runs.nbody_eih
    every = int(round(ORACLE_SAMPLE_DAYS / SAMPLE_STEP_DAYS))
    t_max = eih.t[0] + ORACLE_YEARS * JULIAN_YEAR_DAYS
    idx = np.arange(0, eih.t.size, every)
    idx = idx[eih.t[idx] <= t_max + 1e-9]
    state = load_j2000()
    try:
        tic = time.perf_counter()
        pos_rb = rebound_gr_full_positions(state, eih.t[idx], C_AU_DAY)
        log(
            f"REBOUNDx gr_full oracle ({ORACLE_YEARS:g} yr, IAS15): "
            f"{time.perf_counter() - tic:.1f} s"
        )
    except ImportError as exc:
        log(f"REBOUNDx oracle skipped: {exc}")
        return []
    sep_km = np.linalg.norm(eih.pos[idx] - pos_rb, axis=2) * AU_KM  # (T, N)
    worst = sep_km.max(axis=0)
    years = (eih.t[idx][-1] - eih.t[0]) / JULIAN_YEAR_DAYS
    merc = state.index("Mercury")
    return [
        CheckResult(
            "rebound_gr_full_vs_eih_max_km",
            float(worst.max()),
            0.0,
            ORACLE_TOL_KM,
            f"km, max over {years:g} yr and all bodies ({state.names[int(worst.argmax())]}); "
            f"Mercury {worst[merc]:.3g} km",
        )
    ]


def check_ppn(runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Check 13: Sun + Mercury under ``make_eih(beta, gamma)``, rate over the PPN prediction."""
    state = two_body_state()
    el = osculating_elements(state, "Mercury")
    t_eval = daily_grid(state.t, PPN_YEARS)
    tic = time.perf_counter()
    newton = run_dop853(state, accel_newton, t_eval)
    runs.ppn_newton = newton
    runs.ppn_two_body = {}
    out: list[CheckResult] = []
    for beta, gamma in PPN_CASES:
        traj = run_dop853(state, make_eih(c=C_AU_DAY, beta=beta, gamma=gamma), t_eval)
        runs.ppn_two_body[(beta, gamma)] = traj
        rate, err = perihelion_rate_difference(traj, newton, "Mercury")
        factor = (2.0 + 2.0 * gamma - beta) / 3.0
        predicted = factor * MERCURY_PRECESSION_GR_ARCSEC_CY
        analytic = float(
            gr_precession_arcsec_per_century(
                el["a"], el["e"], el["gm"], c=C_AU_DAY, beta=beta, gamma=gamma
            )
        )
        out.append(
            CheckResult(
                f"ppn_two_body_fit_over_prediction_beta{beta:g}_gamma{gamma:g}",
                rate / predicted,
                1.0,
                0.01,
                f"ratio; fit {rate:.3f} +- {err:.3f} arcsec/cy, (2+2g-b)/3 = {factor:.4f}, "
                f"prediction {predicted:.3f} (analytic {analytic:.3f})",
            )
        )
    log(
        f"PPN two-body runs ({len(PPN_CASES)} laws, {PPN_YEARS:g} yr): "
        f"{time.perf_counter() - tic:.1f} s"
    )
    return out


def check_light() -> list[CheckResult]:
    """Checks 9, 14 and 15: light deflection (constants, exact geodesic, Soldner) and Shapiro."""
    delta = 4.0 * GM_SUN / (C_AU_DAY**2 * R_SUN_AU) * ARCSEC_PER_RAD
    exact = null.deflection_angle(R_SUN_AU, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD
    soldner = null.deflection_newtonian_corpuscle(R_SUN_AU, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD
    r_e, r_r = SHAPIRO_R_EMIT_AU, SHAPIRO_R_RECV_AU
    weak = null.shapiro_delay(r_e, r_r, R_SUN_AU, GM_SUN, C_AU_DAY)
    numeric = null.shapiro_delay_numeric(r_e, r_r, R_SUN_AU, GM_SUN, C_AU_DAY)
    return [
        CheckResult(
            "light_deflection_limb_arcsec", delta, LIGHT_DEFLECTION_LIMB_ARCSEC, 0.001, "arcsec"
        ),
        CheckResult(
            "light_deflection_numeric_limb_arcsec",
            exact,
            LIGHT_DEFLECTION_LIMB_ARCSEC,
            0.001,
            "arcsec, exact null geodesic (metric.null.deflection_angle)",
        ),
        CheckResult(
            "light_deflection_soldner_limb_arcsec",
            soldner,
            0.5 * LIGHT_DEFLECTION_LIMB_ARCSEC,
            0.001,
            "arcsec, Newtonian corpuscle 2 GM/(c^2 R_sun) = half of GR",
        ),
        CheckResult(
            "shapiro_one_way_earth_mars_limb_us",
            weak / MICROSECOND_DAYS,
            124.0,
            REPORT_ONLY,
            f"microseconds, r_e = {r_e} AU, r_r = {r_r} AU, b = R_sun; round trip "
            f"{2.0 * weak / MICROSECOND_DAYS:.1f} (report only)",
        ),
        CheckResult(
            "shapiro_numeric_over_weak_field",
            numeric / weak,
            1.0,
            0.01,
            "ratio, exact null geodesic over the weak-field formula at the same b",
        ),
    ]


def check_extras(years: float, runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Checks 16 and 17: solar J2 and Lense-Thirring, real size over the span and amplified."""
    if runs.two_body_newton is None:
        raise RuntimeError("check_extras needs the two-body Newton run of check_two_body")
    state = two_body_state()
    sun = state.index("Sun")
    newton = runs.two_body_newton
    out: list[CheckResult] = []

    # real sizes over the full span, differenced against the Newton run on the same grid
    tic = time.perf_counter()
    j2_real = run_dop853(state, combine(accel_newton, make_j2(central=sun)), newton.t)
    lt_real = run_dop853(state, combine(accel_newton, make_lense_thirring(central=sun)), newton.t)
    log(f"two-body J2 and Lense-Thirring runs ({years:g} yr): {time.perf_counter() - tic:.1f} s")
    rate_j2, err_j2 = perihelion_rate_difference(j2_real, newton, "Mercury")
    rate_lt, err_lt = perihelion_rate_difference(lt_real, newton, "Mercury")
    runs.extras_rates = {"j2": rate_j2, "lt": rate_lt}
    out.append(
        CheckResult(
            "extras_j2_real_minus_newton_mercury",
            rate_j2,
            MERCURY_PRECESSION_J2_ARCSEC_CY,
            J2_TOL_ARCSEC_CY,
            f"arcsec/cy, J2 = {SUN_J2:g}, {years:g} yr, OLS err {err_j2:.2g}",
        )
    )
    out.append(
        CheckResult(
            "extras_lense_thirring_real_minus_newton_mercury",
            rate_lt,
            MERCURY_PRECESSION_LT_ARCSEC_CY,
            REPORT_ONLY,
            f"arcsec/cy, GJ = {GJ_SUN:.3g} AU^5/day^3, {years:g} yr, OLS err {err_lt:.2g} "
            "(report only)",
        )
    )

    # amplified effects over EXTRAS_YEARS: node and argument rates in the solar-equator frame
    t_end = state.t + EXTRAS_YEARS * JULIAN_YEAR_DAYS
    if newton.t[-1] >= t_end - 1e-9:
        newton10 = truncate(newton, t_end)
    else:
        newton10 = run_dop853(state, accel_newton, daily_grid(state.t, EXTRAS_YEARS))
    runs.extras_newton = newton10
    tic = time.perf_counter()
    j2_big = run_dop853(
        state,
        combine(accel_newton, make_j2(j2=J2_AMPLIFICATION * SUN_J2, central=sun)),
        newton10.t,
    )
    lt_big = run_dop853(
        state,
        combine(accel_newton, make_lense_thirring(central=sun, scale=LT_AMPLIFICATION)),
        newton10.t,
    )
    log(
        f"amplified J2 and Lense-Thirring runs ({EXTRAS_YEARS:g} yr): "
        f"{time.perf_counter() - tic:.1f} s"
    )
    runs.extras_amplified = {"j2": j2_big, "lt": lt_big}
    el = osculating_elements(state, "Mercury")
    r, v = state.relative_to("Sun")
    merc = state.index("Mercury")
    inc = float(inclination_to_axis(r[merc], v[merc], SUN_POLE_ECLIPTIC))
    predictions = {
        "j2": kaula_rates(el["a"], el["e"], inc, el["gm"], J2_AMPLIFICATION * SUN_J2),
        "lt": lense_thirring_rates(el["a"], el["e"], inc, LT_AMPLIFICATION * GJ_SUN),
    }
    labels = {"j2": ("j2_x1000", "Kaula"), "lt": ("lense_thirring_x1e5", "LT formula")}
    for key, traj in runs.extras_amplified.items():
        fits = equatorial_rate_differences(traj, newton10)
        tag, formula = labels[key]
        for element, name, predicted in zip(
            ("raan", "argp"), ("node", "argument"), predictions[key], strict=True
        ):
            _, _, rate, err = fits[element]
            predicted_cy = float(rate_arcsec_per_century(predicted))
            out.append(
                CheckResult(
                    f"extras_{tag}_{name}_over_{formula.split()[0].lower()}",
                    rate / predicted_cy,
                    1.0,
                    EXTRAS_RATE_TOL,
                    f"ratio; fit {rate:.4g} +- {err:.2g} vs {formula} {predicted_cy:.4g} "
                    f"arcsec/cy, i = {math.degrees(inc):.2f} deg to the solar equator",
                )
            )
    return out


def check_moon(years: float, runs: Runs, log: Callable[[str], None]) -> list[CheckResult]:
    """Check 18: Sun + planets + Earth + Moon against lunar theory, Horizons and the EMB runs."""
    ref = load_moon_reference_series()
    span = min(years, MOON_YEARS)
    n = int(round(span * JULIAN_YEAR_DAYS))
    t_eval = ref.t[: n + 1]
    horizons = Trajectory(t_eval, ref.pos[: n + 1], ref.vel[: n + 1], ref.gm, ref.names)
    split, emb = load_j2000(with_moon=True), load_j2000()
    gr_law = make_newton_pn1(c=C_AU_DAY, central=split.index("Sun"))
    tic = time.perf_counter()
    split_newton = run_dop853(split, accel_newton, t_eval)
    split_gr = run_dop853(split, gr_law, t_eval)
    emb_newton = run_dop853(emb, accel_newton, t_eval)
    emb_gr = run_dop853(emb, gr_law, t_eval)
    log(
        f"Earth + Moon runs ({split.n} bodies, {span:g} yr, Newton and 1PN, split and EMB): "
        f"{time.perf_counter() - tic:.1f} s"
    )
    runs.moon_newton, runs.moon_gr, runs.moon_horizons = split_newton, split_gr, horizons

    out: list[CheckResult] = []
    fit = lunar_fits(split_newton)
    ref_fit = lunar_fits(horizons)
    runs.moon_fits = fit
    out.append(
        CheckResult(
            "moon_sidereal_month_days",
            fit["month"],
            SIDEREAL_MONTH_DAYS,
            0.05,
            f"days, {span:g} yr; Horizons same span {ref_fit['month']:.5f}; "
            f"a = {fit['a_km']:.0f} km, e = {fit['e']:.4f}, i = {fit['inc_deg']:.2f} deg",
        )
    )
    out.append(
        CheckResult(
            "moon_perigee_rate_deg_yr",
            fit["perigee"],
            MOON_PERIGEE_RATE_DEG_YR,
            1.5,
            f"deg/yr, OLS err {fit['perigee_err']:.2g}; Horizons same span "
            f"{ref_fit['perigee']:+.3f}",
        )
    )
    out.append(
        CheckResult(
            "moon_node_rate_deg_yr",
            fit["node"],
            MOON_NODE_RATE_DEG_YR,
            1.0,
            f"deg/yr, OLS err {fit['node_err']:.2g}; Horizons same span {ref_fit['node']:+.3f}",
        )
    )
    out.append(
        CheckResult(
            "moon_perigee_rate_minus_horizons_same_span",
            fit["perigee"] - ref_fit["perigee"],
            0.0,
            0.3,
            "deg/yr: the short-span bias is the same in the run and in the ephemeris",
        )
    )
    out.append(
        CheckResult(
            "moon_node_rate_minus_horizons_same_span",
            fit["node"] - ref_fit["node"],
            0.0,
            0.3,
            "deg/yr",
        )
    )

    # the Moon against Horizons over the first MOON_SEPARATION_YEARS
    t_sep = split.t + min(span, MOON_SEPARATION_YEARS) * JULIAN_YEAR_DAYS
    sep_newton = separation(truncate(split_newton, t_sep), truncate(horizons, t_sep), "Moon")
    sep_gr = separation(truncate(split_gr, t_sep), truncate(horizons, t_sep), "Moon")
    window = min(span, MOON_SEPARATION_YEARS)
    out.append(
        CheckResult(
            "moon_vs_horizons_max_km_pn1",
            float(sep_gr.max() * AU_KM),
            0.0,
            MOON_SEPARATION_TOL_KM,
            f"km, Newton + 1PN, max over {window:g} yr",
        )
    )
    out.append(
        CheckResult(
            "moon_vs_horizons_max_km_newton",
            float(sep_newton.max() * AU_KM),
            0.0,
            REPORT_ONLY,
            f"km, Newton alone, max over {window:g} yr (report only)",
        )
    )

    # the EMB approximation: the real barycentre vs Horizons and vs the EMB run
    bc_split = earth_moon_barycentre(split_gr)
    bc_horizons = earth_moon_barycentre(horizons)
    emb_pos = emb_gr.body("EMB")[0]
    vs_horizons = np.linalg.norm(bc_split - bc_horizons, axis=1) * AU_KM
    drift = np.linalg.norm(bc_split - emb_pos, axis=1) * AU_KM
    eps, predicted = predicted_emb_drift_km(split, span * JULIAN_YEAR_DAYS)
    out.append(
        CheckResult(
            "emb_split_barycentre_vs_horizons_max_km",
            float(vs_horizons.max()),
            0.0,
            REPORT_ONLY,
            f"km over {span:g} yr, Newton + 1PN (report only)",
        )
    )
    out.append(
        CheckResult(
            "emb_point_mass_vs_split_barycentre_final_km",
            float(drift[-1]),
            predicted,
            REPORT_ONLY,
            f"km after {span:g} yr; predicted 2 eps n a t with eps = {eps:.2e} (report only)",
        )
    )

    # Mercury: the GR signal does not care, the Newtonian rate shifts by -0.0025
    rate_split, err_split = perihelion_rate_difference(split_gr, split_newton, "Mercury")
    rate_emb, _ = perihelion_rate_difference(emb_gr, emb_newton, "Mercury")
    shift, shift_err = perihelion_rate_difference(split_newton, emb_newton, "Mercury")
    runs.moon_shift = shift
    out.append(
        CheckResult(
            "mercury_pn1_minus_newton_with_moon_minus_emb_value",
            rate_split - rate_emb,
            0.0,
            0.1,
            f"arcsec/cy over {span:g} yr: {rate_split:.4f} +- {err_split:.2g} (Earth + Moon) vs "
            f"{rate_emb:.4f} (EMB)",
        )
    )
    out.append(
        CheckResult(
            "mercury_newton_rate_shift_moon_vs_emb",
            shift,
            -0.0025,
            REPORT_ONLY,
            f"arcsec/cy, OLS err {shift_err:.2g} (report only)",
        )
    )
    return out


def run_checks(
    years: float = 100.0, log: Callable[[str], None] = lambda s: None
) -> tuple[list[CheckResult], Runs]:
    """Run every check; returns the results and the trajectories for the figures."""
    runs = Runs()
    results: list[CheckResult] = []
    results += check_two_body(years, runs, log)
    results += check_schwarzschild(log)
    results += check_exaggerated_c(runs, log)
    results += check_nbody(years, runs, log)
    results += check_horizons(years, runs, log)
    results += check_eih(years, runs, log)
    results += check_rebound_oracle(runs, log)
    results += check_ppn(runs, log)
    results += check_light()
    results += check_extras(years, runs, log)
    results += check_moon(years, runs, log)
    return results, runs


# ----------------------------------------------------------------------------- figures
def save_figures(runs: Runs, figdir: Path, log: Callable[[str], None]) -> list[Path]:
    """Draw the headline figures with ``gravsim.viz`` and save them as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gravsim.viz import (
        PALETTE,
        body_colors,
        figure_grid,
        plot_energy_error,
        plot_orbits_3d,
        plot_precession,
        plot_separation,
    )
    from gravsim.viz.plots import INK as PALETTE_INK

    PALETTE_BLUE, PALETTE_ORANGE = PALETTE[0], PALETTE[1]

    figdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    years = (runs.nbody_newton.t[-1] - runs.nbody_newton.t[0]) / JULIAN_YEAR_DAYS

    # 1. Mercury's perihelion longitude, Newton vs Newton + 1PN (N-body), and their difference.
    newton, pn1 = runs.nbody_newton, runs.nbody_pn1
    t, w_n = perihelion_longitude_series(newton, "Mercury")
    _, w_g = perihelion_longitude_series(pn1, "Mercury")
    fit_n, fit_g = fit_secular_rate(t, w_n), fit_secular_rate(t, w_g)
    diff = wrap_angle(w_g - w_n)
    fit_d = fit_secular_rate(t, diff)
    fig, axes = figure_grid(2, panel=(5.5, 4.0))
    plot_precession(t, w_n, fit_n.rate, ax=axes[0], label="Newton, Sun + 8 planets")
    plot_precession(t, w_g, fit_g.rate, ax=axes[0], label="Newton + 1PN")
    axes[0].set_title(f"Mercury: longitude of perihelion, {years:.0f} yr")
    plot_precession(t, diff, fit_d.rate, ax=axes[1], label="(Newton + 1PN) minus Newton")
    axes[1].set_title("Run-vs-run difference: the GR advance alone")
    paths.append(_save(fig, figdir / "mercury_perihelion_newton_vs_gr.png"))

    # 2. Exaggerated-c rosette (3-D, a longer figure-only run so the ellipse visibly turns)
    #    next to the varpi(t) fit of the 50-orbit check run.
    state2 = two_body_state()
    el_m = osculating_elements(state2, "Mercury")
    c_exag = C_AU_DAY / EXAGGERATION
    t_ros = state2.t + np.arange(0.0, ROSETTE_ORBITS * el_m["period"], 0.25)
    tic = time.perf_counter()
    rosette = run_dop853(state2, make_newton_pn1(c=c_exag, central=state2.index("Sun")), t_ros)
    log(f"rosette run ({ROSETTE_ORBITS} orbits, figure only): {time.perf_counter() - tic:.1f} s")
    fig = plt.figure(figsize=(11.0, 4.6), layout="constrained")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    # Individual orbits spaced ROSETTE_EVERY apart, coloured by time, so the turning of the
    # ellipse (0.29 deg per orbit) is visible instead of 200 superposed loops.
    r_rel, _ = rosette.relative("Mercury", "Sun")
    t_rel = rosette.t - rosette.t[0]
    shown = list(range(0, ROSETTE_ORBITS, ROSETTE_EVERY))
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(shown):
        m = (t_rel >= k * el_m["period"]) & (t_rel <= (k + 1) * el_m["period"])
        label = f"orbit {k + 1}" if k in (shown[0], shown[-1]) else None
        ax3.plot(
            r_rel[m, 0],
            r_rel[m, 1],
            r_rel[m, 2],
            color=cmap(i / max(1, len(shown) - 1)),
            lw=0.9,
            label=label,
        )
    ax3.plot([0.0], [0.0], [0.0], marker="*", color="black", ms=9, ls="none", label="Sun")
    lim = float(np.abs(r_rel).max()) * 1.05
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_zlim(-lim, lim)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("x [AU]")
    ax3.set_ylabel("y [AU]")
    ax3.set_zlabel("z [AU]")
    ax3.view_init(elev=50.0, azim=-60.0)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_title(
        f"Mercury with c/{EXAGGERATION:g}: every {ROSETTE_EVERY}th of {ROSETTE_ORBITS} orbits"
    )
    ax2 = fig.add_subplot(1, 2, 2)
    exag = runs.exaggerated
    t_e, w_e = perihelion_longitude_series(exag, "Mercury")
    fit_e = fit_secular_rate(t_e, w_e)
    plot_precession(t_e, w_e, fit_e.rate, ax=ax2, units="deg", label=f"1PN, c/{EXAGGERATION:g}")
    analytic = gr_precession_arcsec_per_century(el_m["a"], el_m["e"], el_m["gm"], c=c_exag)
    ax2.set_title(
        f"Perihelion advance, {EXAGGERATED_ORBITS} orbits "
        f"(analytic {float(analytic) / 3600.0:.1f} deg/century)"
    )
    paths.append(_save(fig, figdir / "exaggerated_c_rosette.png"))

    # 3. The solar system in 3-D: all planets and the inner four.
    fig, axes = figure_grid(2, projection="3d", panel=(5.5, 5.0))
    plot_orbits_3d(newton, ax=axes[0])
    axes[0].set_title(f"Sun + 8 planets, Newton, {years:.0f} yr (DOP853)")
    plot_orbits_3d(newton, list(PLANETS[:4]), ax=axes[1])
    axes[1].set_title("Inner planets")
    paths.append(_save(fig, figdir / "solar_system_3d.png"))

    # 4. Energy error of the production and the symplectic runs.
    fig, axes = figure_grid(1, panel=(6.0, 4.0))
    ax = axes[0]
    plot_energy_error(
        newton.t,
        relative_energy_error(newton),
        ax=ax,
        t_units="years",
        label=f"DOP853 rtol 1e-12, Sun + 8 planets, {years:.0f} yr",
    )
    yosh = runs.yoshida4
    plot_energy_error(
        yosh.t,
        relative_energy_error(yosh),
        ax=ax,
        t_units="years",
        label=f"Yoshida4 dt = {yosh.meta['dt']:g} d, 10 yr",
    )
    ax.set_title("Relative energy error")
    paths.append(_save(fig, figdir / "energy_error.png"))

    # 5. GR advance per orbit against a(1-e^2), with the analytic 1/p law.  Only planets
    #    that complete at least MIN_ORBITS_FOR_RATE orbits in the span are shown: for the
    #    others the differenced osculating series is dominated by its periodic part.
    state = load_j2000()
    fig, axes = figure_grid(1, panel=(6.0, 4.2))
    ax = axes[0]
    colors = body_colors(PLANETS)
    p_min, p_max = math.inf, 0.0
    span = newton.t[-1] - newton.t[0]
    for body, color in zip(PLANETS, colors, strict=True):
        el = osculating_elements(state, body)
        if span / el["period"] < MIN_ORBITS_FOR_RATE:
            continue
        p = el["a"] * (1.0 - el["e"] ** 2)
        p_min, p_max = min(p_min, p), max(p_max, p)
        rate, err = runs.planet_rates[body]
        per_orbit = rate * el["period"] / JULIAN_CENTURY_DAYS
        per_orbit_err = err * el["period"] / JULIAN_CENTURY_DAYS
        ax.errorbar(
            p, per_orbit, yerr=per_orbit_err, fmt="o", color=color, ms=5, capsize=2, label=body
        )
    p_grid = np.geomspace(p_min / 1.5, p_max * 1.5, 200)
    ax.plot(
        p_grid,
        6.0 * math.pi * GM_SUN / (C_AU_DAY**2 * p_grid) * ARCSEC_PER_RAD,
        "-",
        color="0.25",
        lw=1.0,
        label=r"$6\pi GM_\odot / (c^2 a(1-e^2))$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a(1-e^2)$ [AU]")
    ax.set_ylabel("perihelion advance per orbit [arcsec]")
    ax.set_title(f"1PN advance from the {years:.0f}-yr N-body runs (1PN minus Newton)")
    ax.set_xlim(p_min / 1.5, p_max * 1.5)
    ax.legend(fontsize=8)
    paths.append(_save(fig, figdir / "precession_vs_a.png"))

    # 6. Separation from Horizons (bonus): Newton drifts, 1PN does not.
    ref = truncate(runs.horizons, newton.t[-1])
    fig, axes = figure_grid(1, panel=(6.0, 4.0))
    ax = axes[0]
    for traj, label in ((newton, "Newton"), (pn1, "Newton + 1PN")):
        sub = on_common_grid(traj, ref.t)
        plot_separation(sub.t, separation(sub, ref, "Mercury", t_atol=1e-9), ax=ax, label=label)
    ax.set_yscale("log")
    ax.set_title("Mercury: distance from the Horizons ephemeris")
    paths.append(_save(fig, figdir / "mercury_vs_horizons.png"))

    # 7. (Phase 5) Mercury's separation from Horizons for Newton, Newton + 1PN and EIH on one
    #    plot, next to the km-level difference between the two GR laws.
    if runs.nbody_eih is not None:
        eih = runs.nbody_eih
        fig, axes = figure_grid(2, panel=(5.5, 4.0))
        ax = axes[0]
        for traj, label in ((newton, "Newton"), (pn1, "Newton + 1PN (Sun only)"), (eih, "EIH")):
            sub = on_common_grid(traj, ref.t)
            plot_separation(
                sub.t, separation(sub, ref, "Mercury", t_atol=1e-9), ax=ax, label=label
            )
        ax.set_yscale("log")
        ax.set_title(f"Mercury: distance from Horizons, {years:.0f} yr")
        ax = axes[1]
        for body in ("Mercury", "Sun"):
            plot_separation(eih.t, separation(eih, pn1, body), ax=ax, label=body)
        ax.set_title("EIH minus (Newton + 1PN): the two GR laws differ by km")
        paths.append(_save(fig, figdir / "eih_vs_horizons.png"))

    # 8. (Phase 5) PPN grid: the differenced perihelion longitude of Mercury for each
    #    (beta, gamma) with its fit and the (2 + 2 gamma - beta)/3 prediction.
    if runs.ppn_two_body and runs.ppn_newton is not None:
        cases = list(runs.ppn_two_body)
        fig, axes = figure_grid(len(cases), ncols=2, panel=(5.5, 3.6))
        t_n, w_n = perihelion_longitude_series(runs.ppn_newton, "Mercury")
        for ax, (beta, gamma) in zip(axes, cases, strict=True):
            _, w_p = perihelion_longitude_series(runs.ppn_two_body[(beta, gamma)], "Mercury")
            diff = wrap_angle(w_p - w_n)
            fit = fit_secular_rate(t_n, diff)
            factor = (2.0 + 2.0 * gamma - beta) / 3.0
            plot_precession(
                t_n, diff, fit.rate, ax=ax, label=f"EIH(β={beta:g}, γ={gamma:g}) − Newton"
            )
            predicted = factor * MERCURY_PRECESSION_GR_ARCSEC_CY
            t_yr = (t_n - t_n[0]) / JULIAN_YEAR_DAYS
            ax.plot(
                t_yr,
                predicted / JULIAN_CENTURY_DAYS * (t_n - t_n[0]),
                ":",
                color="0.4",
                lw=1.2,
                label=f"(2+2γ−β)/3 × 42.98 = {predicted:.2f} arcsec/century",
            )
            ax.legend(fontsize=7, loc="upper left")
            ax.set_title(f"β = {beta:g}, γ = {gamma:g}: factor {factor:.3f}", fontsize=10)
        fig.suptitle("Mercury's perihelion under the PPN-EIH equations (Sun + Mercury, 10 yr)")
        paths.append(_save(fig, figdir / "ppn_precession_grid.png"))

    # 9. (Phase 6) Light bending: a bundle of rays past the Sun (deviation amplified) and the
    #    deflection against the impact parameter with the 1919 and VLBI measurements.
    fig, axes = figure_grid(2, panel=(5.5, 4.4))
    ax = axes[0]
    b_values = np.concatenate([-np.array(RAY_BUNDLE_R_SUN), RAY_BUNDLE_R_SUN]) * R_SUN_AU
    r_max = 2.4 * max(RAY_BUNDLE_R_SUN) * R_SUN_AU
    rays = null.ray_bundle(b_values, GM_SUN, C_AU_DAY, 600, r_max)
    for b, (x, y) in zip(b_values, rays, strict=True):
        y_amp = b + RAY_AMPLIFICATION * (y - b)
        delta_arcsec = null.deflection_angle(abs(b), GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD
        ax.plot(
            x / R_SUN_AU,
            y_amp / R_SUN_AU,
            color=PALETTE_INK if abs(b) <= R_SUN_AU else PALETTE_BLUE,
            lw=1.2,
            label=f"b = {abs(b) / R_SUN_AU:g} R☉: {delta_arcsec:.3f}″" if b > 0 else None,
        )
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, color="#eda100", alpha=0.9, label="Sun"))
    lim = r_max / R_SUN_AU * 0.75
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [R☉]")
    ax.set_ylabel("y [R☉]")
    ax.set_title(f"Rays past the Sun, deviation from straight × {RAY_AMPLIFICATION:g}")
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    b_grid = np.geomspace(1.0, 100.0, 40) * R_SUN_AU
    exact = np.array([null.deflection_angle(b, GM_SUN, C_AU_DAY) for b in b_grid])
    ax.plot(
        b_grid / R_SUN_AU, exact * ARCSEC_PER_RAD, color=PALETTE_BLUE, lw=2.0,
        label="exact null geodesic (GR)",
    )
    ax.plot(
        b_grid / R_SUN_AU,
        null.deflection_angle_weak(b_grid, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD,
        "--", color=PALETTE_INK, lw=1.0, label="4GM/(c²b), Einstein 1915",
    )
    ax.plot(
        b_grid / R_SUN_AU,
        null.deflection_newtonian_corpuscle(b_grid, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD,
        "--", color=PALETTE_ORANGE, lw=1.0, label="2GM/(c²b), Soldner 1801",
    )
    for k, (label, value, err) in enumerate(DEFLECTION_MEASUREMENTS):
        ax.errorbar(
            1.0 + 0.03 * k, value, yerr=err, fmt="o", ms=4, capsize=2, color=f"C{k + 2}",
            label=f"{label}: {value:.3g} ± {err:.2g}″ (scaled to the limb)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("impact parameter b [R☉]")
    ax.set_ylabel("deflection [arcsec]")
    ax.set_title("Deflection versus impact parameter")
    ax.legend(fontsize=7)
    paths.append(_save(fig, figdir / "light_bending_rays.png"))

    # 10. (Phase 8) Mercury's precession budget: every contribution on a log scale, the
    #     modelled ones with the values measured by this script.
    if runs.extras_rates is not None and runs.planet_rates is not None:
        asteroid = ring_precession_rate(ASTEROID_RING_GM, ASTEROID_RING_AU, el_m["a"])
        budget = [
            ("planets (Newtonian)", MERCURY_PRECESSION_NEWTON_ARCSEC_CY, True),
            ("General Relativity (1PN − Newton)", runs.planet_rates["Mercury"][0], True),
            ("solar J₂ = 2.2e−7", runs.extras_rates["j2"], True),
            ("Earth + Moon as bodies vs EMB", runs.moon_shift, True),
            ("Lense–Thirring (solar spin)", runs.extras_rates["lt"], True),
            ("asteroid belt (ring, analytic)", asteroid, False),
        ]
        budget = [b for b in budget if np.isfinite(b[1])]
        fig, axes = figure_grid(1, panel=(7.5, 4.2))
        ax = axes[0]
        names = [b[0] for b in budget][::-1]
        rates = np.array([b[1] for b in budget])[::-1]
        colors = [
            (PALETTE_BLUE if b[1] >= 0.0 else PALETTE_ORANGE) if b[2] else "0.6"
            for b in budget
        ][::-1]
        bars = ax.barh(names, np.abs(rates), color=colors)
        for bar, rate in zip(bars, rates, strict=True):
            ax.text(
                bar.get_width() * 1.15,
                bar.get_y() + bar.get_height() / 2.0,
                f"{rate:+.4g}″/cy",
                va="center",
                fontsize=8,
            )
        ax.axvline(0.01, color=PALETTE_INK, linestyle=":", linewidth=1.0)
        ax.text(0.0105, len(budget) - 0.6, "ranging precision ~0.01″/cy", fontsize=7, rotation=90,
                va="top", color=PALETTE_INK)
        ax.set_xscale("log")
        ax.set_xlim(5e-4, 5e3)
        ax.set_xlabel("|rate| [arcsec per century]")
        ax.set_title(
            "Mercury's perihelion budget (orange = regression, grey = not modelled)", fontsize=10
        )
        paths.append(_save(fig, figdir / "precession_budget.png"))

    # 11. (Phase 8) The Moon about the Earth in 3-D with its perigees marked (the line of
    #     apsides turns by 40.7 deg/yr) next to the fitted perigee and node longitudes.
    if runs.moon_newton is not None and runs.moon_fits is not None:
        moon = runs.moon_newton
        t_rel = moon.t - moon.t[0]
        shown = t_rel <= 2.0 * JULIAN_YEAR_DAYS + 1e-9  # two years: 27 orbits, 81 deg of advance
        r_geo, _ = moon.relative("Moon", "Earth")
        r_km = r_geo[shown] * AU_KM
        dist = np.linalg.norm(r_km, axis=1)
        minima = np.flatnonzero((dist[1:-1] < dist[:-2]) & (dist[1:-1] < dist[2:])) + 1
        fig = plt.figure(figsize=(11.0, 4.8), layout="constrained")
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        cmap = plt.get_cmap("viridis")
        t_shown = t_rel[shown]
        for k in range(len(t_shown) - 1):
            ax3.plot(
                r_km[k : k + 2, 0], r_km[k : k + 2, 1], r_km[k : k + 2, 2],
                color=cmap(t_shown[k] / t_shown[-1]), lw=0.8,
            )
        ax3.plot(
            r_km[minima, 0], r_km[minima, 1], r_km[minima, 2], "o", color=PALETTE_ORANGE,
            ms=4, label=f"perigees ({minima.size}), advancing {runs.moon_fits['perigee']:.1f}°/yr",
        )
        ax3.plot(
            [0.0], [0.0], [0.0], marker="o", color=PALETTE_BLUE, ms=8, ls="none", label="Earth"
        )
        lim = float(np.abs(r_km).max()) * 1.05
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(-lim, lim)
        ax3.set_box_aspect((1, 1, 1))
        ax3.set_xlabel("x [km]")
        ax3.set_ylabel("y [km]")
        ax3.set_zlabel("z [km]")
        ax3.view_init(elev=35.0, azim=-60.0)
        ax3.legend(loc="upper left", fontsize=8)
        ax3.set_title("The Moon about the Earth, 2 yr (ten-body Newton run)", fontsize=10)
        ax2 = fig.add_subplot(1, 2, 2)
        r_all, v_all = moon.relative("Moon", "Earth")
        gm_em = float(moon.gm[moon.index("Earth")] + moon.gm[moon.index("Moon")])
        el_moon = state_to_elements(r_all, v_all, gm_em)
        for key, label, rate_deg in (
            ("varpi", "longitude of perigee ϖ", runs.moon_fits["perigee"]),
            ("raan", "longitude of the node Ω", runs.moon_fits["node"]),
        ):
            fit = fit_secular_rate(moon.t, el_moon[key])
            plot_precession(
                moon.t, el_moon[key], fit.rate, ax=ax2, units="deg",
                label=f"{label}: {rate_deg:+.2f}°/yr",
            )
        ax2.set_ylabel("Δ angle [deg]")
        ax2.set_title(
            f"Lunar perigee and node, {(moon.t[-1] - moon.t[0]) / JULIAN_YEAR_DAYS:.0f} yr "
            f"(secular {MOON_PERIGEE_RATE_DEG_YR:+.2f} / {MOON_NODE_RATE_DEG_YR:+.2f}°/yr)",
            fontsize=10,
        )
        paths.append(_save(fig, figdir / "moon_orbit.png"))

    # 12. (Phase 8) The amplified solar effects: differenced node and argument of Mercury in the
    #     solar-equator frame for J2 x 1000 and J x 1e5 with the fits and the formulas.
    if runs.extras_amplified is not None and runs.extras_newton is not None:
        state2 = two_body_state()
        r2, v2 = state2.relative_to("Sun")
        merc = state2.index("Mercury")
        inc = float(inclination_to_axis(r2[merc], v2[merc], SUN_POLE_ECLIPTIC))
        predictions = {
            "j2": kaula_rates(el_m["a"], el_m["e"], inc, el_m["gm"], J2_AMPLIFICATION * SUN_J2),
            "lt": lense_thirring_rates(el_m["a"], el_m["e"], inc, LT_AMPLIFICATION * GJ_SUN),
        }
        titles = {
            "j2": f"Solar oblateness, J₂ × {J2_AMPLIFICATION:g} (Kaula)",
            "lt": f"Frame dragging, J × {LT_AMPLIFICATION:g} (Lense–Thirring)",
        }
        fig, axes = figure_grid(2, panel=(5.5, 4.0))
        for ax, key in zip(axes, ("lt", "j2"), strict=True):
            fits = equatorial_rate_differences(runs.extras_amplified[key], runs.extras_newton)
            for element, symbol, predicted in zip(
                ("raan", "argp"), ("ΔΩ (node)", "Δω (argument)"), predictions[key], strict=True
            ):
                t, diff, rate, _ = fits[element]
                plot_precession(
                    t, diff, rate / (ARCSEC_PER_RAD * JULIAN_CENTURY_DAYS), ax=ax,
                    label=f"{symbol}: fit {rate:+.4g}″/cy",
                )
                t_yr = (t - t[0]) / JULIAN_YEAR_DAYS
                ax.plot(
                    t_yr, predicted * ARCSEC_PER_RAD * (t - t[0]), ":", color="0.4", lw=1.2,
                    label=f"formula {float(rate_arcsec_per_century(predicted)):+.4g}″/cy",
                )
            ax.legend(fontsize=7, loc="center left")
            ax.set_ylabel("Δ angle [arcsec]")
            ax.set_title(titles[key], fontsize=10)
        fig.suptitle(
            "Sun + Mercury: (Newton + extra) − Newton in the solar-equator frame, "
            f"{EXTRAS_YEARS:g} yr"
        )
        paths.append(_save(fig, figdir / "lense_thirring_amplified.png"))

    log(f"saved {len(paths)} figures to {figdir}")
    return paths


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--years",
        type=float,
        default=100.0,
        help="span of the century-class runs in Julian years (default 100)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("figures"),
        help="directory for the PNG figures (default figures/)",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip the figures")
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the results table as JSON to this path"
    )
    parser.add_argument("--quiet", action="store_true", help="no progress messages")
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        if not args.quiet:
            print(f"[validate] {msg}", file=sys.stderr, flush=True)

    tic = time.perf_counter()
    results, runs = run_checks(args.years, log)
    figures: list[Path] = []
    if not args.no_figures:
        figures = save_figures(runs, args.figures, log)
    log(f"total {time.perf_counter() - tic:.0f} s")

    print(format_table(results))
    n_fail = sum(not r.passed for r in results)
    print(
        f"\n{len(results) - n_fail}/{len(results)} checks passed"
        + ("" if n_fail == 0 else f", {n_fail} FAILED")
    )
    if figures:
        print("figures: " + ", ".join(str(p) for p in figures))
    if args.json is not None:
        doc = [asdict(r) | {"pass": r.passed} for r in results]
        for row in doc:
            if math.isinf(row["tolerance"]):
                row["tolerance"] = None
        args.json.write_text(
            json.dumps(
                {"years": args.years, "results": doc, "figures": [str(p) for p in figures]},
                indent=2,
            )
            + "\n"
        )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
