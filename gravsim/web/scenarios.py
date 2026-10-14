"""Scenario builders for the Three.js front end: the physics runs here, the browser only draws.

Every builder is a pure function of its parameters returning
``{"meta": {...}, "arrays": {name: float32/int32 ndarray}}``; ``build_scenario`` validates the
parameters against the declared ranges, calls the builder and completes ``meta`` (id, title,
params, ``n_samples`` and the array documentation ``meta["arrays"]`` = name -> {shape, dtype,
description}).  No FastAPI, no streamlit, no I/O besides the shipped ephemeris cache.

Physics
-------
``solar_system`` and ``earth_moon`` integrate the J2000 snapshot (``data.load_j2000``) twice on a
common output grid with DOP853 (rtol 1e-12, atol 1e-15): Newtonian gravity
(``forces.accel_newton``) and the full Einstein-Infeld-Hoffmann 1PN N-body law
(``forces.make_eih``).  Comparing the two runs sample by sample cancels the planetary wobble of
the osculating elements and most of the numerical error (``analysis.compare``), leaving the
secular GR advance of each perihelion,

    dvarpi/dt = 6 pi gm / (c^2 a (1 - e^2))  per orbit  ->  42.98 "/century for Mercury,

which is also given analytically (``analysis.precession.gr_precession_arcsec_per_century``).
The Sun's barycentric wobble (dominated by Jupiter, ~0.005 AU ~ 1 R_sun) is reported so the
front end can offer a heliocentric/barycentric toggle.  For ``earth_moon`` the lunar orbit about
the Earth is reduced to osculating elements (``analysis.kepler.state_to_elements`` with
``gm = GM_Earth + GM_Moon``) and the secular rates of the longitude of perigee ``varpi`` and of
the node ``Omega`` are fitted (solar perturbations: +40.69 and -19.34 deg/yr; Brouwer & Clemence
1961, ch. 12); the perigee passages give the anomalistic month (27.555 d) and the swept
geocentric longitude the sidereal month (27.322 d).

``rosette`` keeps only Sun + Mercury (barycentric) and replaces EIH by Newton + the 1PN
test-particle correction (``forces.make_newton_pn1``) with the speed of light scaled down to
``c_scale * C_AU_DAY``.  The 1PN advance scales as ``1/c^2``, so ``c_scale = 0.01`` multiplies
Mercury's 5.0e-7 rad/orbit by 1e4 and the orbit becomes a visible rosette; the fitted advance per
orbit is compared with ``metric.schwarzschild.precession_per_orbit_1pn`` at the same ``c``.  The
expansion needs ``v/c << 1``: Mercury's ``v/c ~ 1.6e-4`` becomes 0.16 at ``c_scale = 0.001``.

``black_hole`` is an exact timelike geodesic of the Schwarzschild metric in geometric units
(``GM = c = 1``; lengths in GM/c^2, times in GM/c^3), integrated in proper time by
``metric.schwarzschild.geodesic_orbit`` with the exact effective potential

    V_eff(r) = -1/r + L^2/(2 r^2) - L^2/r^3 .

The orbit is launched tangentially at the apoapsis ``r0``; the periapsis ``peri`` fixes ``L``
through ``V_eff(r0) = V_eff(peri)`` (both are turning points of the same energy level),

    L^2 = (1/peri - 1/r0) / [ f(peri) - f(r0) ],   f(r) = 1/(2 r^2) - 1/r^3 ,

and ``E`` follows from the 4-velocity normalisation.  A bound orbit exists only outside the
separatrix ``p = 6 + 2 e`` (Cutler, Kennefick & Poisson 1994), i.e. for
``peri > 4 r0 / (r0 - 2)``; below it the level lies above the centrifugal barrier and the
particle never reaches ``peri`` (it turns around outside the unstable circular orbit), so the
requested periapsis is clamped to the separatrix value times ``1 + 1 %`` and both numbers are
reported.  Close to the separatrix the particle whirls several times near the unstable circular
orbit before zooming back out: the zoom-whirl orbit (Glampedakis & Kennefick 2002).  ISCO,
photon sphere and horizon sit at 6, 3 and 2 GM/c^2.

``light_rays`` traces null geodesics (``metric.null.ray_path``, the null Binet equation
``u'' + u = 3 (GM/c^2) u^2``) for a bundle of impact parameters.  With ``compact = True`` the mass
is a black hole in units ``GM/c^2 = 1`` and ``b`` runs geometrically from ``1.05 b_crit`` to 12
(``b_crit = 3 sqrt 3`` is the capture threshold, where the deflection diverges logarithmically).
With ``compact = False`` the mass is the Sun, ``b`` runs from 1 to 5 R_sun and, because the
real bending is only 1.75" at the limb, the deviation of each ray from its incoming straight line
``y = b`` is multiplied by a stated factor (``meta["exaggeration"]``) for display; the reported
deflections are the true ones (``4 GM/(c^2 b)`` to leading order).

References
----------
* Will, *Living Rev. Relativ.* 17, 4 (2014), sec. 3.4 (PPN equations of motion, deflection).
* Cutler, Kennefick & Poisson, *Phys. Rev. D* 50, 3816 (1994) (separatrix ``p = 6 + 2e``).
* Glampedakis & Kennefick, *Phys. Rev. D* 66, 044002 (2002) (zoom-whirl orbits).
* Brouwer & Clemence, *Methods of Celestial Mechanics* (1961), ch. 12 (lunar theory).
* Keeton & Petters, *Phys. Rev. D* 72, 104006 (2005) (deflection series in ``GM/(c^2 b)``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from gravsim.analysis.compare import perihelion_rate_difference, separation
from gravsim.analysis.kepler import state_to_elements
from gravsim.analysis.precession import (
    fit_secular_rate,
    gr_precession_arcsec_per_century,
    perihelion_passages,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import (
    ARCSEC_PER_RAD,
    AU_KM,
    C_AU_DAY,
    DEG_PER_RAD,
    GM_SUN,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
    MOON_NODE_RATE_DEG_YR,
    MOON_PERIGEE_RATE_DEG_YR,
    R_SUN_AU,
)
from gravsim.data.horizons import load_j2000
from gravsim.forces import accel_newton, make_eih, make_newton_pn1
from gravsim.integrators import integrate_scipy
from gravsim.metric import null
from gravsim.metric import schwarzschild as sch

__all__ = [
    "Param",
    "Scenario",
    "SCENARIOS",
    "list_scenarios",
    "validate_params",
    "default_params",
    "build_scenario",
    "build_solar_system",
    "build_rosette",
    "build_black_hole",
    "build_light_rays",
    "build_earth_moon",
    "separatrix_periapsis",
    "zoom_whirl_angular_momentum",
]

RTOL, ATOL = 1e-12, 1e-15  # production DOP853 tolerances (CONTRIBUTING.md)
FRAME = "barycentric, J2000 ecliptic"
EPOCH = "J2000.0 TDB (JD 2451545.0)"
MOON_SIDEREAL_MONTH_DAYS = 27.321662  # reference values for the earth_moon numbers
MOON_ANOMALISTIC_MONTH_DAYS = 27.554550
EARTH_MOON_MEAN_DISTANCE_KM = 384_400.0

# black_hole tuning
_BH_SEPARATRIX_MARGIN = 0.01  # relative distance of the clamped periapsis outside the separatrix
_BH_SAMPLES_PER_PERIOD = 1200  # output samples per estimated radial period
_BH_TAU_FACTOR = 2.5  # first guess of tau_end in units of n_orbits * T_est (doubled on demand)
_BH_MAX_EXTENSIONS = 5

# light_rays tuning
_RAYS_POINTS = 600  # samples per ray polyline
_RAYS_COMPACT_B_MIN_FACTOR = 1.05  # first impact parameter in units of b_crit
_RAYS_COMPACT_B_MAX = 12.0  # last impact parameter in GM/c^2
_RAYS_COMPACT_R_MAX = 30.0  # the arms are cut at this radius (GM/c^2)
_RAYS_SUN_B_RANGE = (1.0, 5.0)  # impact parameters in solar radii
_RAYS_SUN_R_MAX = 10.0  # cut radius in solar radii
_RAYS_SUN_EXAGGERATION = 20_000.0  # the limb deflection of 8.5e-6 rad is drawn as 0.17 rad


# --------------------------------------------------------------------------- parameter model
@dataclass(frozen=True)
class Param:
    """One scenario parameter: type (``"float"``, ``"int"`` or ``"bool"``), default and range."""

    name: str
    type: str
    default: float | int | bool
    description: str
    min: float | None = None
    max: float | None = None

    def to_dict(self) -> dict:
        out: dict = {"type": self.type, "default": self.default, "description": self.description}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        return out

    def coerce(self, value: object) -> float | int | bool:
        """Convert a raw value (number, bool or query string) and check the range."""
        if self.type == "bool":
            return _coerce_bool(self.name, value)
        if isinstance(value, bool):
            raise ValueError(f"parameter {self.name!r} must be a number, got {value!r}")
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameter {self.name!r} must be a number, got {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"parameter {self.name!r} must be finite, got {value!r}")
        if self.min is not None and number < self.min:
            raise ValueError(f"parameter {self.name!r} = {number} is below the minimum {self.min}")
        if self.max is not None and number > self.max:
            raise ValueError(f"parameter {self.name!r} = {number} is above the maximum {self.max}")
        if self.type == "int":
            if number != int(number):
                raise ValueError(f"parameter {self.name!r} must be an integer, got {value!r}")
            return int(number)
        return number


def _coerce_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"parameter {name!r} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class Scenario:
    """A scenario: identity, documentation, declared parameters and the builder."""

    id: str
    title: str
    blurb: str
    params: tuple[Param, ...]
    build: Callable[..., dict]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "blurb": self.blurb,
            "params": {p.name: p.to_dict() for p in self.params},
        }


# --------------------------------------------------------------------------- shared helpers
def _time_grid(t0: float, span_days: float, step_days: float) -> np.ndarray:
    """``t0, t0 + step, ...`` with ``round(span/step)`` steps; the last one lands near ``span``."""
    n = int(round(span_days / step_days))
    return t0 + step_days * np.arange(n + 1, dtype=np.float64)


def _osculating(state: SystemState, body: str, central: str = "Sun") -> dict[str, float]:
    r, v = state.relative_to(central)
    i = state.index(body)
    gm = float(state.gm[state.index(central)] + state.gm[i])
    el = state_to_elements(r[i], v[i], gm)
    return {k: float(el[k]) for k in ("a", "e", "inc", "period")} | {"gm": gm}


def _nearest_indices(t: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the sample of the sorted grid ``t`` closest to each of ``times`` (int32)."""
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return np.empty(0, dtype=np.int32)
    hi = np.clip(np.searchsorted(t, times), 1, t.size - 1)
    lo = hi - 1
    idx = np.where(np.abs(t[hi] - times) < np.abs(times - t[lo]), hi, lo)
    return idx.astype(np.int32)


def _perihelion_indices(traj: Trajectory, body: str, central: str) -> np.ndarray:
    t_peri, _ = perihelion_passages(traj, body, central)
    return _nearest_indices(traj.t, t_peri)


def _local_minima(x: np.ndarray) -> np.ndarray:
    """Interior indices ``i`` with ``x[i] < x[i-1]`` and ``x[i] <= x[i+1]`` (ties count once)."""
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])) + 1


def _model_meta(traj: Trajectory, law, note: str) -> dict:
    params = dict(getattr(law, "params", None) or {"law": getattr(law, "__name__", str(law))})
    return {
        "law": str(params.get("law")),
        "law_params": {k: v for k, v in params.items() if k not in ("law", "terms")},
        "integrator": f"{traj.meta.get('integrator')} {traj.meta.get('method')}",
        "rtol": traj.meta.get("rtol"),
        "atol": traj.meta.get("atol"),
        "nfev": traj.meta.get("nfev"),
        "note": note,
    }


def _nbody_arrays(newton: Trajectory, gr: Trajectory) -> tuple[dict, dict]:
    """``t``, ``pos_newton``, ``pos_gr`` and ``delta_gr`` with their documentation."""
    n = newton.n
    arrays = {
        "t": newton.t.astype(np.float32),
        "pos_newton": newton.pos.astype(np.float32),
        "pos_gr": gr.pos.astype(np.float32),
        "delta_gr": (gr.pos - newton.pos).astype(np.float32),
    }
    docs = {
        "t": "(T,) sample times, days since J2000.0 TDB",
        "pos_newton": f"(T, {n}, 3) barycentric positions of the Newtonian run, AU",
        "pos_gr": f"(T, {n}, 3) barycentric positions of the GR run, AU",
        "delta_gr": (
            f"(T, {n}, 3) pos_gr - pos_newton computed in float64 before the float32 cast "
            "(AU); use it for amplified differences, the float32 positions alone resolve "
            "only ~5 km at 0.4 AU"
        ),
    }
    return arrays, docs


def _rate_entry(gr: Trajectory, newton: Trajectory, state: SystemState, body: str) -> dict:
    rate, err = perihelion_rate_difference(gr, newton, body)
    el = _osculating(state, body)
    analytic = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"]))
    return {
        "gr_minus_newton_arcsec_cy": rate,
        "err_arcsec_cy": err,
        "analytic_1pn_arcsec_cy": analytic,
        "a_au": el["a"],
        "e": el["e"],
        "period_days": el["period"],
    }


# --------------------------------------------------------------------------- solar_system
def build_solar_system(years: float = 100.0, step_days: float = 2.0) -> dict:
    """Sun + 8 planets, Newton vs EIH, with Mercury's perihelia and the GR rates.

    Parameters
    ----------
    years : float
        Span in Julian years from J2000.0.
    step_days : float
        Output sample spacing [day] (the integrator steps adaptively underneath).

    Returns
    -------
    dict
        ``arrays``: ``t`` (T,), ``pos_newton``/``pos_gr``/``delta_gr`` (T, 9, 3),
        ``peri_idx_newton``/``peri_idx_gr`` (K,) int32 sample indices of Mercury's perihelia.
        ``meta``: names, units, frame, per-model notes, GR-minus-Newton perihelion rates of
        Mercury, Venus, EMB and Mars with the analytic 1PN values, the final Mercury
        separation (km) and the Sun's barycentric wobble (AU and solar radii).
    """
    state = load_j2000()
    t_eval = _time_grid(state.t, years * JULIAN_YEAR_DAYS, step_days)
    gr_law = make_eih()
    newton = integrate_scipy(state, accel_newton, t_eval, rtol=RTOL, atol=ATOL)
    gr = integrate_scipy(state, gr_law, t_eval, rtol=RTOL, atol=ATOL)

    arrays, docs = _nbody_arrays(newton, gr)
    arrays["peri_idx_newton"] = _perihelion_indices(newton, "Mercury", "Sun")
    arrays["peri_idx_gr"] = _perihelion_indices(gr, "Mercury", "Sun")
    docs["peri_idx_newton"] = "(K,) sample indices of Mercury's perihelion passages, Newtonian run"
    docs["peri_idx_gr"] = "(K,) sample indices of Mercury's perihelion passages, GR run"

    rates = {
        body: _rate_entry(gr, newton, state, body) for body in ("Mercury", "Venus", "EMB", "Mars")
    }
    sep_km = separation(newton, gr, "Mercury") * AU_KM
    sun = state.index("Sun")
    sun_r = np.linalg.norm(gr.pos[:, sun, :], axis=1)
    wobble_au = float(sun_r.max())
    meta = {
        "names": list(state.names),
        "gm": [float(g) for g in state.gm],
        "central": "Sun",
        "units": {"length": "AU", "time": "day"},
        "frame": FRAME,
        "epoch": EPOCH,
        "t0": float(state.t),
        "step_days": float(step_days),
        "span_days": float(t_eval[-1] - t_eval[0]),
        "models": {
            "newton": _model_meta(newton, accel_newton, "pairwise Newtonian gravity"),
            "gr": _model_meta(
                gr, gr_law, "Einstein-Infeld-Hoffmann 1PN N-body law, Newtonian term included"
            ),
        },
        "perihelion_body": "Mercury",
        "perihelion_rates": rates,
        "mercury": {
            "gr_minus_newton_arcsec_cy": rates["Mercury"]["gr_minus_newton_arcsec_cy"],
            "analytic_1pn_arcsec_cy": rates["Mercury"]["analytic_1pn_arcsec_cy"],
            "final_separation_km": float(sep_km[-1]),
            "max_separation_km": float(sep_km.max()),
            "n_perihelia_gr": int(arrays["peri_idx_gr"].size),
        },
        "sun_wobble": {
            "amplitude_au": wobble_au,
            "amplitude_r_sun": wobble_au / R_SUN_AU,
            "mean_au": float(sun_r.mean()),
            "r_sun_au": R_SUN_AU,
        },
        "notes": [
            "GR rates are run-vs-run differences (GR minus Newton on the same grid); absolute "
            "fits over a century are biased by long-period planetary terms.",
            "Positions are float32: the GR-Newton offset of Mercury is below the float32 "
            "resolution for the first decades, use delta_gr for amplified differences.",
        ],
        "arrays": docs,
    }
    return {"meta": meta, "arrays": arrays}


# --------------------------------------------------------------------------- rosette
def build_rosette(c_scale: float = 0.01, n_orbits: int = 50, step_days: float = 0.25) -> dict:
    """Sun + Mercury with the speed of light scaled down: the perihelion advance made visible.

    Parameters
    ----------
    c_scale : float
        ``c = c_scale * C_AU_DAY``; the 1PN advance per orbit grows as ``1/c_scale^2``.
    n_orbits : int
        Number of Mercury orbits (osculating period at J2000) to integrate.
    step_days : float
        Output sample spacing [day].

    Returns
    -------
    dict
        Same arrays as ``build_solar_system`` with N = 2; ``meta["advance"]`` holds the fitted
        advance per orbit (differenced varpi fit and perihelion passages) and the analytic 1PN
        value at the same ``c``, in rad, deg and arcsec/century.
    """
    state = load_j2000().subset(("Sun", "Mercury")).to_barycentric()
    el = _osculating(state, "Mercury")
    period = el["period"]
    c = C_AU_DAY * c_scale
    t_eval = _time_grid(state.t, n_orbits * period, step_days)
    gr_law = make_newton_pn1(c=c, central=state.index("Sun"))
    newton = integrate_scipy(state, accel_newton, t_eval, rtol=RTOL, atol=ATOL)
    gr = integrate_scipy(state, gr_law, t_eval, rtol=RTOL, atol=ATOL)

    arrays, docs = _nbody_arrays(newton, gr)
    arrays["peri_idx_newton"] = _perihelion_indices(newton, "Mercury", "Sun")
    arrays["peri_idx_gr"] = _perihelion_indices(gr, "Mercury", "Sun")
    docs["peri_idx_newton"] = "(K,) sample indices of Mercury's perihelion passages, Newtonian run"
    docs["peri_idx_gr"] = "(K,) sample indices of Mercury's perihelion passages, 1PN run"

    rate_cy, err_cy = perihelion_rate_difference(gr, newton, "Mercury")
    per_orbit_fit = rate_cy / ARCSEC_PER_RAD / JULIAN_CENTURY_DAYS * period
    per_orbit_1pn = sch.precession_per_orbit_1pn(el["a"], el["e"], el["gm"], c)
    analytic_cy = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"], c=c))
    per_orbit_passages = _passage_advance(gr, period) - _passage_advance(newton, period)
    r_rel, v_rel = state.relative_to("Sun")
    i_m = state.index("Mercury")
    r_peri = el["a"] * (1.0 - el["e"])
    v_peri = math.sqrt(el["gm"] * (1.0 + el["e"]) / r_peri)
    meta = {
        "names": list(state.names),
        "gm": [float(g) for g in state.gm],
        "central": "Sun",
        "units": {"length": "AU", "time": "day"},
        "frame": "Sun-Mercury barycentre, J2000 ecliptic",
        "epoch": EPOCH,
        "t0": float(state.t),
        "step_days": float(step_days),
        "span_days": float(t_eval[-1] - t_eval[0]),
        "c_scale": float(c_scale),
        "c_au_day": c,
        "c_real_au_day": C_AU_DAY,
        "orbit": {
            "a_au": el["a"],
            "e": el["e"],
            "period_days": period,
            "n_orbits": int(n_orbits),
            "v_over_c_at_perihelion": v_peri / c,
            "v_over_c_now": float(np.linalg.norm(v_rel[i_m])) / c,
        },
        "models": {
            "newton": _model_meta(newton, accel_newton, "two-body Newtonian gravity"),
            "gr": _model_meta(
                gr, gr_law, f"Newton + 1PN test-particle correction with c = {c_scale} c"
            ),
        },
        "perihelion_body": "Mercury",
        "advance": {
            "fit_rad_per_orbit": per_orbit_fit,
            "fit_err_rad_per_orbit": err_cy / ARCSEC_PER_RAD / JULIAN_CENTURY_DAYS * period,
            "passages_rad_per_orbit": per_orbit_passages,
            "analytic_1pn_rad_per_orbit": per_orbit_1pn,
            "fit_deg_per_orbit": per_orbit_fit * DEG_PER_RAD,
            "analytic_1pn_deg_per_orbit": per_orbit_1pn * DEG_PER_RAD,
            "fit_arcsec_cy": rate_cy,
            "analytic_1pn_arcsec_cy": analytic_cy,
            "fit_over_analytic": per_orbit_fit / per_orbit_1pn,
            "real_mercury_arcsec_cy": float(
                gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"])
            ),
        },
        "notes": [
            "The 1PN expansion needs v/c << 1; below c_scale ~ 0.01 the orbit is still integrable "
            "but the analytic rate is only indicative.",
        ],
        "arrays": docs,
    }
    return {"meta": meta, "arrays": arrays}


def _passage_advance(traj: Trajectory, period: float) -> float:
    """Secular rate of varpi at the perihelion passages, converted to rad per ``period``."""
    t_peri, varpi = perihelion_passages(traj, "Mercury", "Sun")
    if t_peri.size < 2:
        return float("nan")
    return float(fit_secular_rate(t_peri, varpi).rate * period)


# --------------------------------------------------------------------------- black_hole
def separatrix_periapsis(r0: float) -> float:
    """Smallest periapsis of a bound Schwarzschild orbit with apoapsis ``r0`` (GM/c^2 units).

    From the separatrix ``p = 6 + 2 e`` with ``p = 2 r0 rp/(r0 + rp)`` and
    ``e = (r0 - rp)/(r0 + rp)``: ``rp_min = 4 r0 / (r0 - 2)`` (4 for ``r0 -> inf``, 6 at the
    ISCO ``r0 = 6``).
    """
    if r0 <= 2.0:
        raise ValueError("the apoapsis must lie outside the horizon r = 2")
    return 4.0 * r0 / (r0 - 2.0)


def zoom_whirl_angular_momentum(r0: float, peri: float) -> float:
    """``L`` (units GM/c) of the geodesic with turning points ``r0`` and ``peri``.

    ``V_eff(r0) = V_eff(peri)`` for ``V_eff = -1/r + L^2/(2 r^2) - L^2/r^3`` gives
    ``L^2 = (1/peri - 1/r0) / [f(peri) - f(r0)]`` with ``f(r) = 1/(2 r^2) - 1/r^3``.  Both
    radii must exceed the photon sphere ``r = 3`` (``f`` is decreasing there) and
    ``peri < r0``.  Whether ``peri`` is actually reached is a separate question
    (:func:`separatrix_periapsis`).
    """
    if not 3.0 < peri < r0:
        raise ValueError(f"need 3 < peri < r0, got peri = {peri}, r0 = {r0}")

    def f(r: float) -> float:
        return 1.0 / (2.0 * r * r) - 1.0 / r**3

    L2 = (1.0 / peri - 1.0 / r0) / (f(peri) - f(r0))
    return math.sqrt(L2)


def build_black_hole(r0: float = 20.0, peri: float = 4.6, n_orbits: int = 6) -> dict:
    """Zoom-whirl timelike geodesic around a Schwarzschild black hole, ``GM = c = 1``.

    Parameters
    ----------
    r0 : float
        Apoapsis radius [GM/c^2]; the orbit starts there, moving tangentially.
    peri : float
        Requested periapsis [GM/c^2].  Clamped to ``(1 + 1 %) * separatrix_periapsis(r0)``
        when it is not reachable by a bound orbit; ``meta["periapsis"]`` records both.
    n_orbits : int
        Number of radial periods (apoapsis to apoapsis) to integrate.

    Returns
    -------
    dict
        ``arrays``: ``geodesic`` (T, 3) with ``x, y`` in GM/c^2 and ``z = 0``, ``t`` (T,)
        coordinate time in GM/c^3 (uniform in proper time, ``meta["tau_step"]``), ``peri_idx``
        and ``apo_idx`` (int32 sample indices of the periapsis/apoapsis passages).  ``meta``:
        ``E``, ``L``, measured ``r_peri``/``r_apo``, advance per radial period, characteristic
        radii (horizon 2, photon sphere 3, ISCO 6) and the circular-orbit radii for this ``L``.
    """
    rp_min = separatrix_periapsis(r0)
    peri_used = max(float(peri), rp_min * (1.0 + _BH_SEPARATRIX_MARGIN))
    clamped = peri_used != float(peri)
    L = zoom_whirl_angular_momentum(r0, peri_used)
    v_phi = L / r0  # L = x vy - y vx with x = r0, y = 0
    E, _ = sch.conserved_quantities([r0, 0.0], [0.0, v_phi], 1.0, 1.0)
    a_est = 0.5 * (r0 + peri_used)
    t_est = 2.0 * math.pi * a_est**1.5  # Newtonian radial period of the osculating ellipse

    tau_end = _BH_TAU_FACTOR * n_orbits * t_est
    for _ in range(_BH_MAX_EXTENSIONS):
        n_samples = int(round(_BH_SAMPLES_PER_PERIOD * tau_end / t_est)) + 1
        tau, t, r, phi = sch.geodesic_orbit(
            [r0, 0.0], [0.0, v_phi], 1.0, 1.0, tau_end=tau_end, n_samples=n_samples
        )
        if r.size < n_samples:  # capture event: cannot happen outside the separatrix
            raise RuntimeError("the geodesic was captured; the periapsis clamp failed")
        apo = _local_minima(-r)
        if apo.size >= n_orbits:
            break
        tau_end *= 2.0
    else:
        raise RuntimeError("could not complete the requested number of radial periods")

    cut = int(apo[n_orbits - 1]) + 1
    tau, t, r, phi = tau[:cut], t[:cut], r[:cut], phi[:cut]
    apo = apo[:n_orbits]
    peri_idx = _local_minima(r)
    geodesic = np.zeros((cut, 3), dtype=np.float32)
    geodesic[:, 0] = r * np.cos(phi)
    geodesic[:, 1] = r * np.sin(phi)
    advance = (float(phi[apo[-1]]) - float(phi[0])) / n_orbits - 2.0 * math.pi
    r_stable, r_unstable = sch.circular_orbit_radii(L, 1.0, 1.0)

    arrays = {
        "geodesic": geodesic,
        "t": t.astype(np.float32),
        "peri_idx": peri_idx.astype(np.int32),
        "apo_idx": apo.astype(np.int32),
    }
    meta = {
        "units": {"length": "GM/c^2", "time": "GM/c^3"},
        "frame": "Schwarzschild coordinates, equatorial plane, mass at the origin",
        "t0": 0.0,
        "tau_step": float(tau[1] - tau[0]) if cut > 1 else 0.0,
        "tau_end": float(tau[-1]),
        "E": float(E),
        "L": float(L),
        "r0": float(r0),
        "periapsis": {
            "requested": float(peri),
            "used": float(peri_used),
            "separatrix": float(rp_min),
            "clamped": bool(clamped),
        },
        "r_peri": float(r.min()),
        "r_apo": float(r.max()),
        "n_orbits": int(n_orbits),
        "radial_period_tau": float(tau[apo[-1]]) / n_orbits,
        "radial_period_t": float(t[apo[-1]]) / n_orbits,
        "advance_per_orbit_rad": advance,
        "advance_per_orbit_deg": advance * DEG_PER_RAD,
        "turns_per_orbit": (advance + 2.0 * math.pi) / (2.0 * math.pi),
        "advance_1pn_rad": sch.precession_per_orbit_1pn(
            a_est, (r0 - peri_used) / (r0 + peri_used), 1.0, 1.0
        ),
        "r_horizon": sch.schwarzschild_radius(1.0, 1.0),
        "r_photon_sphere": sch.photon_sphere_radius(1.0, 1.0),
        "r_isco": sch.isco_radius(1.0, 1.0),
        "r_circular_stable": float(r_stable),
        "r_circular_unstable": float(r_unstable),
        "notes": [
            "Exact Schwarzschild geodesic in proper time; samples are uniform in tau, t is the "
            "coordinate time of a distant observer.",
            "A bound orbit needs peri > 4 r0/(r0 - 2) (separatrix p = 6 + 2e); the requested "
            "periapsis is clamped just outside it when necessary.",
        ],
        "arrays": {
            "geodesic": "(T, 3) x, y of the geodesic in GM/c^2 (z = 0), uniform in proper time",
            "t": "(T,) Schwarzschild coordinate time in GM/c^3 at each sample",
            "peri_idx": "(K,) sample indices of the periapsis passages",
            "apo_idx": "(n_orbits,) sample indices of the apoapsis returns (sample 0 is one too)",
        },
    }
    return {"meta": meta, "arrays": arrays}


# --------------------------------------------------------------------------- light_rays
def build_light_rays(n_rays: int = 9, compact: bool = True) -> dict:
    """A bundle of null geodesics past a black hole (``compact``) or the Sun.

    Parameters
    ----------
    n_rays : int
        Number of rays.
    compact : bool
        ``True``: mass with ``GM/c^2 = 1``, impact parameters geometric from ``1.05 b_crit``
        to 12 (lengths in GM/c^2).  ``False``: the Sun, impact parameters from 1 to 5 R_sun,
        drawn with the deviation from the straight line exaggerated (lengths in R_sun).

    Returns
    -------
    dict
        ``arrays``: ``ray_0 ... ray_{k-1}`` (M, 3) polylines (``z = 0``) in the scenario's
        length unit and ``b_values`` (k,).  ``meta``: ``b_crit``, deflection of each ray in
        radians and arcseconds (true values), closest approach, exaggeration factor, radius of
        the drawn body and the cut radius of the arms.
    """
    if compact:
        gm, c = 1.0, 1.0
        unit, unit_au = "GM/c^2", GM_SUN / C_AU_DAY**2
        b_crit = null.critical_impact_parameter(gm, c)
        b_values = np.geomspace(_RAYS_COMPACT_B_MIN_FACTOR * b_crit, _RAYS_COMPACT_B_MAX, n_rays)
        r_max, scale, exaggeration = _RAYS_COMPACT_R_MAX, 1.0, 1.0
        body_radius = sch.schwarzschild_radius(gm, c)
        body = "Schwarzschild black hole"
    else:
        gm, c = GM_SUN, C_AU_DAY
        unit, unit_au = "R_sun", R_SUN_AU
        b_crit = null.critical_impact_parameter(gm, c)
        b_values = np.linspace(*_RAYS_SUN_B_RANGE, n_rays) * R_SUN_AU
        r_max, scale, exaggeration = _RAYS_SUN_R_MAX * R_SUN_AU, R_SUN_AU, _RAYS_SUN_EXAGGERATION
        body_radius = R_SUN_AU
        body = "Sun"

    arrays: dict[str, np.ndarray] = {}
    docs: dict[str, str] = {}
    deflection = []
    closest = []
    for k, b in enumerate(b_values):
        b = float(b)
        x, y = null.ray_path(b, gm, c, n_points=_RAYS_POINTS, r_max=r_max)
        y = b + exaggeration * (y - b)  # amplify the deviation from the incoming line y = b
        ray = np.zeros((x.size, 3), dtype=np.float32)
        ray[:, 0] = x / scale
        ray[:, 1] = y / scale
        arrays[f"ray_{k}"] = ray
        docs[f"ray_{k}"] = (
            f"(M, 3) polyline of the ray with b = {b / scale:.4g} {unit}, in the order travelled "
            f"(enters from -x along y = b); z = 0"
        )
        deflection.append(null.deflection_angle(b, gm, c))
        closest.append(null.closest_approach_radius(b, gm, c) / scale)
    arrays["b_values"] = (b_values / scale).astype(np.float32)
    docs["b_values"] = f"(k,) impact parameters of ray_0 ... ray_{{k-1}} in {unit}"

    deflection_arr = np.asarray(deflection)
    meta = {
        "units": {"length": unit},
        "length_unit_au": float(unit_au),
        "frame": "mass at the origin; rays travel towards +x, incoming asymptote y = b",
        "body": body,
        "compact": bool(compact),
        "n_rays": int(n_rays),
        "gm": float(gm),
        "c": float(c),
        "b_crit": float(b_crit / scale),
        "b_values": (b_values / scale).tolist(),
        "closest_approach": [float(v) for v in closest],
        "deflection_rad": deflection_arr.tolist(),
        "deflection_arcsec": (deflection_arr * ARCSEC_PER_RAD).tolist(),
        "deflection_weak_rad": [float(null.deflection_angle_weak(b, gm, c)) for b in b_values],
        "exaggeration": float(exaggeration),
        "body_radius": float(body_radius / scale),
        "r_horizon": float(sch.schwarzschild_radius(gm, c) / scale),
        "r_photon_sphere": float(sch.photon_sphere_radius(gm, c) / scale),
        "r_max": float(r_max / scale),
        "limb_deflection_arcsec": float(null.solar_limb_deflection_arcsec()),
        "notes": [
            "Deflections are exact null-geodesic values; the weak-field 4 GM/(c^2 b) is given "
            "for comparison.",
            (
                "The drawn deviation from the straight line y = b is exaggerated by the stated "
                "factor; the Sun bends light by only 1.75 arcsec at the limb."
                if not compact
                else "Rays with b <= b_crit = 3 sqrt(3) GM/c^2 would be captured; the bundle "
                "starts at 1.05 b_crit where the deflection is already several radians."
            ),
        ],
        "arrays": docs,
    }
    return {"meta": meta, "arrays": arrays}


# --------------------------------------------------------------------------- earth_moon
def build_earth_moon(years: float = 2.0, step_days: float = 0.1) -> dict:
    """Sun + planets with Earth and Moon separate, Newton vs EIH, plus the lunar orbit numbers.

    Parameters
    ----------
    years : float
        Span in Julian years from J2000.0.
    step_days : float
        Output sample spacing [day] (the Moon's month needs ~0.1 d for a smooth trail).

    Returns
    -------
    dict
        ``arrays``: ``t`` (T,), ``pos_newton``/``pos_gr``/``delta_gr`` (T, 10, 3).  ``meta``:
        sidereal and anomalistic months, perigee and node rates (deg/yr) with the reference
        values, mean/min/max Earth-Moon distance (km), Mercury GR rate.
    """
    state = load_j2000(with_moon=True)
    t_eval = _time_grid(state.t, years * JULIAN_YEAR_DAYS, step_days)
    gr_law = make_eih()
    newton = integrate_scipy(state, accel_newton, t_eval, rtol=RTOL, atol=ATOL)
    gr = integrate_scipy(state, gr_law, t_eval, rtol=RTOL, atol=ATOL)
    arrays, docs = _nbody_arrays(newton, gr)

    r_rel, v_rel = gr.relative("Moon", "Earth")
    gm_em = float(state.gm[state.index("Earth")] + state.gm[state.index("Moon")])
    el = state_to_elements(r_rel, v_rel, gm_em)
    span = float(gr.t[-1] - gr.t[0])
    # Mean motions from the slopes of the mean longitude (varpi + M) and the mean anomaly: the
    # true longitude would carry the 6-degree equation of centre over a fractional revolution.
    two_pi = 2.0 * math.pi
    n_sidereal = fit_secular_rate(gr.t, el["varpi"] + el["M"]).rate
    n_anomalistic = fit_secular_rate(gr.t, el["M"]).rate
    sidereal = two_pi / n_sidereal if n_sidereal > 0 else float("nan")
    anomalistic = two_pi / n_anomalistic if n_anomalistic > 0 else float("nan")
    t_perigee, _ = perihelion_passages(gr, "Moon", central="Earth")
    spacing = float(np.mean(np.diff(t_perigee))) if t_perigee.size >= 2 else float("nan")
    deg_yr = DEG_PER_RAD * JULIAN_YEAR_DAYS
    fit_varpi = fit_secular_rate(gr.t, el["varpi"])
    fit_raan = fit_secular_rate(gr.t, el["raan"])
    fit_argp = fit_secular_rate(gr.t, el["argp"])
    dist_km = np.linalg.norm(r_rel, axis=1) * AU_KM
    mercury_rate, mercury_err = perihelion_rate_difference(gr, newton, "Mercury")

    meta = {
        "names": list(state.names),
        "gm": [float(g) for g in state.gm],
        "central": "Sun",
        "units": {"length": "AU", "time": "day"},
        "frame": FRAME,
        "epoch": EPOCH,
        "t0": float(state.t),
        "step_days": float(step_days),
        "span_days": span,
        "models": {
            "newton": _model_meta(newton, accel_newton, "pairwise Newtonian gravity"),
            "gr": _model_meta(
                gr, gr_law, "Einstein-Infeld-Hoffmann 1PN N-body law, Newtonian term included"
            ),
        },
        "moon": {
            "sidereal_period_days": sidereal,
            "sidereal_period_reference_days": MOON_SIDEREAL_MONTH_DAYS,
            "anomalistic_period_days": anomalistic,
            "anomalistic_period_from_passages_days": spacing,
            "anomalistic_period_reference_days": MOON_ANOMALISTIC_MONTH_DAYS,
            "n_perigee_passages": int(t_perigee.size),
            "perigee_rate_deg_yr": fit_varpi.rate * deg_yr,
            "perigee_rate_err_deg_yr": fit_varpi.rate_err * deg_yr,
            "perigee_rate_reference_deg_yr": MOON_PERIGEE_RATE_DEG_YR,
            "node_rate_deg_yr": fit_raan.rate * deg_yr,
            "node_rate_err_deg_yr": fit_raan.rate_err * deg_yr,
            "node_rate_reference_deg_yr": MOON_NODE_RATE_DEG_YR,
            "argp_rate_deg_yr": fit_argp.rate * deg_yr,
            "mean_distance_km": float(dist_km.mean()),
            "min_distance_km": float(dist_km.min()),
            "max_distance_km": float(dist_km.max()),
            "mean_distance_reference_km": EARTH_MOON_MEAN_DISTANCE_KM,
            "a_mean_au": float(np.mean(el["a"])),
            "e_mean": float(np.mean(el["e"])),
            "inc_mean_deg": float(np.mean(el["inc"])) * DEG_PER_RAD,
            "gm_earth_moon": gm_em,
        },
        "mercury": {
            "gr_minus_newton_arcsec_cy": mercury_rate,
            "err_arcsec_cy": mercury_err,
        },
        "notes": [
            "Lunar elements are osculating geocentric elements (gm = GM_Earth + GM_Moon); the "
            "perigee and node rates are least-squares slopes over the span and carry the large "
            "periodic lunar inequalities (evection, variation) as noise for short spans.",
            "Months are 2 pi over the fitted mean motions of the mean longitude (sidereal) and "
            "of the mean anomaly (anomalistic); the raw perigee-passage spacing is also given.",
        ],
        "arrays": docs,
    }
    return {"meta": meta, "arrays": arrays}


# --------------------------------------------------------------------------- registry
SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in (
        Scenario(
            id="solar_system",
            title="Solar system: Newton vs General Relativity",
            blurb=(
                "Sun and eight planets integrated for a century with Newtonian gravity and "
                "with the Einstein-Infeld-Hoffmann equations; Mercury's perihelion creeps "
                "ahead by 43 arcsec per century in the GR run."
            ),
            params=(
                Param("years", "float", 100.0, "Span in Julian years from J2000.0", 1.0, 200.0),
                Param("step_days", "float", 2.0, "Sample spacing in days", 0.25, 10.0),
            ),
            build=build_solar_system,
        ),
        Scenario(
            id="rosette",
            title="Mercury's rosette at c/100",
            blurb=(
                "Sun + Mercury with the speed of light scaled down: the 1PN perihelion advance "
                "grows as 1/c^2 and the ellipse visibly turns into a rosette."
            ),
            params=(
                Param("c_scale", "float", 0.01, "Speed of light as a fraction of c", 0.001, 1.0),
                Param("n_orbits", "int", 50, "Number of Mercury orbits", 5, 200),
                Param("step_days", "float", 0.25, "Sample spacing in days", 0.05, 2.0),
            ),
            build=build_rosette,
        ),
        Scenario(
            id="black_hole",
            title="Zoom-whirl orbit around a black hole",
            blurb=(
                "Exact Schwarzschild geodesic launched at apoapsis r0 with the periapsis "
                "chosen just outside the separatrix: the particle whirls near the unstable "
                "circular orbit before zooming back out (GM = c = 1)."
            ),
            params=(
                Param("r0", "float", 20.0, "Apoapsis radius in GM/c^2", 6.5, 60.0),
                Param(
                    "peri",
                    "float",
                    4.6,
                    "Requested periapsis in GM/c^2 (clamped to 4 r0/(r0-2) if unreachable)",
                    3.5,
                    6.0,
                ),
                Param("n_orbits", "int", 6, "Radial periods to integrate", 1, 40),
            ),
            build=build_black_hole,
        ),
        Scenario(
            id="light_rays",
            title="Light bending: a bundle of null geodesics",
            blurb=(
                "Photon paths past a black hole (impact parameters from 1.05 b_crit to 12 GM/c^2) "
                "or past the Sun with the 1.75-arcsec limb deflection exaggerated for display."
            ),
            params=(
                Param("n_rays", "int", 9, "Number of rays in the bundle", 1, 25),
                Param("compact", "bool", True, "Black hole (true) or the Sun (false)"),
            ),
            build=build_light_rays,
        ),
        Scenario(
            id="earth_moon",
            title="Earth and Moon among the planets",
            blurb=(
                "The J2000 system with Earth and Moon as separate bodies, Newton vs EIH, with "
                "the lunar months and the perigee and node rates measured from the run."
            ),
            params=(
                Param("years", "float", 2.0, "Span in Julian years from J2000.0", 0.5, 10.0),
                Param("step_days", "float", 0.1, "Sample spacing in days", 0.05, 1.0),
            ),
            build=build_earth_moon,
        ),
    )
}


def list_scenarios() -> list[dict]:
    """Plain-JSON description of every scenario: id, title, blurb and params (default/range)."""
    return [s.to_dict() for s in SCENARIOS.values()]


def default_params(scenario_id: str) -> dict:
    """Default parameter values of ``scenario_id``."""
    return {p.name: p.default for p in _scenario(scenario_id).params}


def _scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        raise ValueError(
            f"unknown scenario {scenario_id!r}; have {sorted(SCENARIOS)}"
        ) from None


def validate_params(scenario_id: str, params: Mapping[str, object] | None = None) -> dict:
    """Coerce and range-check ``params`` against the scenario's declaration, filling defaults.

    Values may be numbers, booleans or query strings.  Raises ``ValueError`` for an unknown
    scenario, an unknown parameter name, a value of the wrong type or outside its range.
    Returns a new dict with every declared parameter in declaration order.
    """
    scenario = _scenario(scenario_id)
    params = dict(params or {})
    declared = {p.name: p for p in scenario.params}
    unknown = sorted(set(params) - set(declared))
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {unknown} for scenario {scenario_id!r}; "
            f"accepted: {list(declared)}"
        )
    out: dict = {}
    for name, param in declared.items():
        out[name] = param.coerce(params[name]) if name in params else param.default
    return out


def _as_storage(name: str, value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind == "f":
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.dtype.kind in "iu":
        return np.ascontiguousarray(arr, dtype=np.int32)
    raise TypeError(f"array {name!r} has unsupported dtype {arr.dtype}")


def build_scenario(scenario_id: str, **params) -> dict:
    """Validate ``params``, run the builder and complete the metadata.

    Parameters
    ----------
    scenario_id : str
        One of ``SCENARIOS``.
    **params
        Scenario parameters (numbers, booleans or query strings); missing ones take defaults.

    Returns
    -------
    dict
        ``{"meta": {...}, "arrays": {name: ndarray}}`` with float32/int32 arrays.  ``meta``
        contains ``id``, ``title``, ``blurb``, ``params`` (as used), ``n_samples`` (length of
        ``t`` when present) and ``arrays`` = name -> {"shape", "dtype", "description"} for
        every array returned.

    Raises
    ------
    ValueError
        Unknown scenario or invalid parameters (see ``validate_params``).
    """
    scenario = _scenario(scenario_id)
    used = validate_params(scenario_id, params)
    result = scenario.build(**used)
    arrays = {name: _as_storage(name, value) for name, value in result["arrays"].items()}
    meta = dict(result["meta"])
    descriptions = dict(meta.get("arrays", {}))
    if set(descriptions) != set(arrays):
        raise RuntimeError(
            f"scenario {scenario_id!r} documents {sorted(descriptions)} but returns "
            f"{sorted(arrays)}"
        )
    meta["arrays"] = {
        name: {
            "shape": [int(s) for s in arr.shape],
            "dtype": str(arr.dtype),
            "description": str(descriptions[name]),
        }
        for name, arr in arrays.items()
    }
    meta.update(
        id=scenario.id,
        title=scenario.title,
        blurb=scenario.blurb,
        params=used,
        n_samples=int(arrays["t"].shape[0]) if "t" in arrays else None,
    )
    return {"meta": meta, "arrays": arrays}
