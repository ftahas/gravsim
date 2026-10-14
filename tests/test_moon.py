"""Validation of the Earth + Moon system (Phase 8): Sun + planets + Earth + Moon under DOP853.

Fast variants integrate two years on the daily grid of the cached Horizons Moon series
(``data.horizons.load_moon_reference_series``); the ten-year variants are ``@pytest.mark.slow``.
Four runs share each grid: the split system (``load_j2000(with_moon=True)``) and the default
EMB system (``load_j2000()``), each with Newton and with Newton + the Sun's 1PN term
(``forces.make_newton_pn1``).  No network.

Physics being checked
---------------------
* The two cached snapshots are consistent: the mass-weighted Earth + Moon barycentre of the
  split snapshot coincides with the EMB entry of the default one (to a metre).
* The lunar orbit about the Earth (osculating geocentric elements, ``gm = GM_Earth + GM_Moon``):
  sidereal month 27.32 d (2 pi over the fitted mean motion of the mean longitude ``varpi + M``),
  perigee advance +40.7 deg/yr and node regression -19.3 deg/yr -- the Sun's tide, the largest
  precessions in the solar system and purely Newtonian.  Short spans are biased by the
  long-period lunar inequalities, so the ten-year fits are held to +-1.5 / +-1.0 deg/yr and the
  two-year ones to +-3 / +-2; the same fits on the Horizons series over the *same* span must
  agree with the run to 0.3 deg/yr whatever the span.
* The Moon's barycentric position against Horizons stays within 500 km over two years (the GR
  run: ~40 km; Newton alone: ~160 km, mostly the Earth's own 1PN advance).
* The EMB approximation is not exact: the barycentre of the Earth-Moon pair feels on average
  ``eps = (3/4) w_E w_M (a_M / a_E)^2 ~ 6e-8`` more solar attraction than a point mass (the
  time-averaged quadrupole of the dumbbell), which raises its mean motion by ``2 eps`` and makes
  the real barycentre run ahead of the point mass by ``2 eps n a t`` ~ 110 km per year.  The
  split run's barycentre matches Horizons' to < 100 km (0.9 km measured), while the EMB *run*
  drifts from both at the predicted rate (ratio measured/predicted within 0.7-1.4; 234 km in
  two years, 1170 km in ten).
* Mercury: the GR-minus-Newton rate with Earth and Moon separate equals the EMB value to
  0.3"/cy over two years (0.1 over ten; measured 1e-7), and the Newtonian rate itself shifts by
  -0.0025"/cy (reported; |shift| < 0.01).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from gravsim import constants as C
from gravsim.analysis import (
    fit_secular_rate,
    perihelion_rate_difference,
    separation,
    state_to_elements,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.data.horizons import load_j2000, load_moon_reference_series
from gravsim.forces import accel_newton, make_newton_pn1
from gravsim.integrators import integrate_scipy

DEG_PER_YEAR = C.DEG_PER_RAD * C.JULIAN_YEAR_DAYS
SIDEREAL_MONTH_DAYS = 27.321662


# --------------------------------------------------------------------------- helpers
@dataclass(frozen=True)
class MoonRuns:
    """The four runs on the Horizons daily grid plus the Horizons series over the same span."""

    years: float
    split: SystemState
    emb: SystemState
    split_newton: Trajectory
    split_gr: Trajectory
    emb_newton: Trajectory
    emb_gr: Trajectory
    horizons: Trajectory


def make_runs(years: float) -> MoonRuns:
    ref = load_moon_reference_series()
    n = int(round(years * C.JULIAN_YEAR_DAYS))
    t_eval = ref.t[: n + 1]
    horizons = Trajectory(t_eval, ref.pos[: n + 1], ref.vel[: n + 1], ref.gm, ref.names)
    split = load_j2000(with_moon=True)
    emb = load_j2000()
    gr_law = make_newton_pn1(central=0)  # the Sun is row 0 in both snapshots
    return MoonRuns(
        years=years,
        split=split,
        emb=emb,
        split_newton=integrate_scipy(split, accel_newton, t_eval),
        split_gr=integrate_scipy(split, gr_law, t_eval),
        emb_newton=integrate_scipy(emb, accel_newton, t_eval),
        emb_gr=integrate_scipy(emb, gr_law, t_eval),
        horizons=horizons,
    )


def weights(state: SystemState) -> np.ndarray:
    """Mass weights ``(w_E, w_M)`` of Earth and Moon."""
    gm = state.gm[[state.index("Earth"), state.index("Moon")]]
    return gm / gm.sum()


def barycentre(traj: Trajectory, w: np.ndarray) -> np.ndarray:
    """Mass-weighted Earth + Moon position along a trajectory, shape (T, 3)."""
    return w[0] * traj.body("Earth")[0] + w[1] * traj.body("Moon")[0]


def lunar_fits(traj: Trajectory) -> dict[str, float]:
    """Sidereal month [d] and perigee/node rates [deg/yr] from the geocentric elements."""
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
        "a_km": float(np.mean(el["a"])) * C.AU_KM,
        "inc_deg": float(np.mean(el["inc"])) * C.DEG_PER_RAD,
    }


def predicted_emb_drift_km(state: SystemState, span_days: float) -> tuple[float, float]:
    """``(eps, 2 eps n a span)`` [-, km]: the along-track drift of the pair's barycentre."""
    w = weights(state)
    r, v = state.relative_to("Sun")
    i_e = state.index("Earth")
    el_e = state_to_elements(r[i_e], v[i_e], C.GM_SUN + state.gm[i_e])
    r_m, v_m = state.relative_to("Earth")
    i_m = state.index("Moon")
    el_m = state_to_elements(r_m[i_m], v_m[i_m], state.gm[i_e] + state.gm[i_m])
    a_e, a_m = float(el_e["a"]), float(el_m["a"])
    eps = 0.75 * w[0] * w[1] * (a_m / a_e) ** 2
    n_e = math.sqrt((C.GM_SUN + state.gm[i_e]) / a_e**3)
    return float(eps), float(2.0 * eps * n_e * span_days * a_e * C.AU_KM)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def runs2() -> MoonRuns:
    return make_runs(2.0)


@pytest.fixture(scope="module")
def runs10() -> MoonRuns:
    return make_runs(10.0)


# --------------------------------------------------------------------------- snapshots
def test_split_snapshot_is_consistent_with_the_emb_snapshot():
    split, emb = load_j2000(with_moon=True), load_j2000()
    assert split.names == C.DEFAULT_BODIES_MOON and emb.names == C.DEFAULT_BODIES
    assert split.t == emb.t == 0.0
    w = weights(split)
    assert w[0] == pytest.approx(1.0 - 1.0 / 82.300569, rel=1e-6)  # Earth/Moon mass ratio 81.3006
    idx = [split.index("Earth"), split.index("Moon")]
    r_bc, v_bc = w @ split.pos[idx], w @ split.vel[idx]
    i_emb = emb.index("EMB")
    assert np.linalg.norm(r_bc - emb.pos[i_emb]) * C.AU_KM < 1e-3  # a metre
    assert np.linalg.norm(v_bc - emb.vel[i_emb]) * C.AU_KM * 1e3 / C.DAY_S < 1e-6  # mm/s
    assert split.gm[idx].sum() == pytest.approx(emb.gm[i_emb], rel=1e-15)
    # the other bodies are identical in the two snapshots
    for name in C.DEFAULT_BODIES:
        if name != "EMB":
            np.testing.assert_array_equal(split.pos[split.index(name)], emb.pos[emb.index(name)])


# --------------------------------------------------------------------------- the lunar orbit
def test_moon_sidereal_month_two_years(runs2):
    fit = lunar_fits(runs2.split_newton)
    assert fit["month"] == pytest.approx(SIDEREAL_MONTH_DAYS, abs=0.05)
    assert fit["a_km"] == pytest.approx(384_400.0, abs=3000.0)
    assert fit["inc_deg"] == pytest.approx(5.15, abs=0.3)
    # the Horizons series over the same span gives the same month to a few seconds
    assert lunar_fits(runs2.horizons)["month"] == pytest.approx(fit["month"], abs=1e-3)


def test_moon_perigee_and_node_rates_two_years(runs2):
    fit = lunar_fits(runs2.split_newton)
    ref = lunar_fits(runs2.horizons)
    report = (
        f"run: perigee {fit['perigee']:+.3f} +- {fit['perigee_err']:.3f}, node {fit['node']:+.3f} "
        f"+- {fit['node_err']:.3f} deg/yr; Horizons same span: {ref['perigee']:+.3f}, "
        f"{ref['node']:+.3f}; secular {C.MOON_PERIGEE_RATE_DEG_YR}, {C.MOON_NODE_RATE_DEG_YR}"
    )
    assert fit["perigee"] == pytest.approx(C.MOON_PERIGEE_RATE_DEG_YR, abs=3.0), report
    assert fit["node"] == pytest.approx(C.MOON_NODE_RATE_DEG_YR, abs=2.0), report
    # the biases of a short span are the same in the run and in the ephemeris
    assert fit["perigee"] == pytest.approx(ref["perigee"], abs=0.3), report
    assert fit["node"] == pytest.approx(ref["node"], abs=0.3), report
    # the GR run gives the same lunar rates (geodetic precession is 2"/yr, invisible here)
    fit_gr = lunar_fits(runs2.split_gr)
    assert fit_gr["perigee"] == pytest.approx(fit["perigee"], abs=0.05)
    assert fit_gr["node"] == pytest.approx(fit["node"], abs=0.01)


def test_moon_position_vs_horizons_two_years(runs2):
    sep_newton = separation(runs2.split_newton, runs2.horizons, "Moon") * C.AU_KM
    sep_gr = separation(runs2.split_gr, runs2.horizons, "Moon") * C.AU_KM
    geo = runs2.split_gr.relative("Moon", "Earth")[0] - runs2.horizons.relative("Moon", "Earth")[0]
    geo_km = np.linalg.norm(geo, axis=1) * C.AU_KM
    report = (
        f"Moon vs Horizons over {runs2.years:g} yr: max {sep_newton.max():.1f} km (Newton), "
        f"{sep_gr.max():.1f} km (Newton + 1PN), geocentric {geo_km.max():.1f} km"
    )
    assert sep_newton[0] == 0.0 and sep_gr[0] == 0.0
    assert sep_newton.max() < 500.0, report
    assert sep_gr.max() < 500.0, report
    assert sep_gr.max() < sep_newton.max(), report  # the ephemeris contains GR
    assert geo_km.max() < 200.0, report


# --------------------------------------------------------------------------- the EMB approximation
def test_emb_approximation_drift_two_years(runs2):
    w = weights(runs2.split)
    bc_split = barycentre(runs2.split_gr, w)
    bc_horizons = barycentre(runs2.horizons, w)
    emb_run = runs2.emb_gr.body("EMB")[0]
    vs_horizons = np.linalg.norm(bc_split - bc_horizons, axis=1) * C.AU_KM
    drift = np.linalg.norm(bc_split - emb_run, axis=1) * C.AU_KM
    emb_vs_horizons = np.linalg.norm(emb_run - bc_horizons, axis=1) * C.AU_KM
    eps, predicted = predicted_emb_drift_km(runs2.split, runs2.years * C.JULIAN_YEAR_DAYS)
    report = (
        f"over {runs2.years:g} yr: split barycentre vs Horizons max {vs_horizons.max():.2f} km; "
        f"split barycentre vs EMB run final {drift[-1]:.1f} km (max {drift.max():.1f}), "
        f"predicted 2 eps n a t = {predicted:.1f} km with eps = {eps:.2e}; "
        f"EMB run vs Horizons max {emb_vs_horizons.max():.1f} km"
    )
    assert vs_horizons.max() < 100.0, report  # the split run follows the ephemeris
    assert eps == pytest.approx(5.9e-8, rel=0.05)
    assert 0.7 < drift[-1] / predicted < 1.4, report  # the EMB run drifts at the predicted rate
    assert drift.max() < 500.0, report
    # the drift is along the track and grows (nearly) linearly with time
    v = runs2.emb_gr.body("EMB")[1]
    along = np.einsum("ij,ij->i", bc_split - emb_run, v) / np.linalg.norm(v, axis=1) * C.AU_KM
    assert along[-1] > 0.95 * drift[-1], report  # the real barycentre runs ahead
    half = drift.size // 2
    assert drift[half] == pytest.approx(0.5 * drift[-1], rel=0.2), report
    # the same holds for the Newtonian pair of runs
    bc_newton = barycentre(runs2.split_newton, w)
    drift_n = np.linalg.norm(bc_newton - runs2.emb_newton.body("EMB")[0], axis=1) * C.AU_KM
    assert drift_n[-1] == pytest.approx(drift[-1], rel=0.01)


# --------------------------------------------------------------------------- Mercury
def test_mercury_gr_rate_unchanged_by_the_moon_two_years(runs2):
    rate_split, err_split = perihelion_rate_difference(
        runs2.split_gr, runs2.split_newton, "Mercury"
    )
    rate_emb, err_emb = perihelion_rate_difference(runs2.emb_gr, runs2.emb_newton, "Mercury")
    shift, shift_err = perihelion_rate_difference(runs2.split_newton, runs2.emb_newton, "Mercury")
    report = (
        f"GR - Newton: split {rate_split:.4f} +- {err_split:.4f}, EMB {rate_emb:.4f} +- "
        f"{err_emb:.4f}; Newton(split) - Newton(EMB) {shift:+.5f} +- {shift_err:.5f} arcsec/cy"
    )
    assert rate_split == pytest.approx(rate_emb, abs=0.3), report
    assert rate_split == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.5), report
    assert abs(shift) < 0.01, report
    assert shift < 0.0, report  # the pair as two bodies slows Mercury's Newtonian advance a little
    # Mercury itself barely notices: a few metres over two years
    sep_km = separation(runs2.split_newton, runs2.emb_newton, "Mercury") * C.AU_KM
    assert sep_km.max() < 0.1, report


# ============================================================================= slow
@pytest.mark.slow
def test_moon_rates_ten_years(runs10):
    fit = lunar_fits(runs10.split_newton)
    ref = lunar_fits(runs10.horizons)
    report = (
        f"10 yr: month {fit['month']:.4f} d; perigee {fit['perigee']:+.3f} +- "
        f"{fit['perigee_err']:.3f}, node {fit['node']:+.3f} +- {fit['node_err']:.3f} deg/yr; "
        f"Horizons same span {ref['perigee']:+.3f}, {ref['node']:+.3f}"
    )
    assert fit["month"] == pytest.approx(SIDEREAL_MONTH_DAYS, abs=0.05), report
    assert fit["perigee"] == pytest.approx(C.MOON_PERIGEE_RATE_DEG_YR, abs=1.5), report
    assert fit["node"] == pytest.approx(C.MOON_NODE_RATE_DEG_YR, abs=1.0), report
    assert fit["perigee"] == pytest.approx(ref["perigee"], abs=0.3), report
    assert fit["node"] == pytest.approx(ref["node"], abs=0.3), report


@pytest.mark.slow
def test_moon_position_and_emb_drift_ten_years(runs10):
    sep_newton = separation(runs10.split_newton, runs10.horizons, "Moon") * C.AU_KM
    sep_gr = separation(runs10.split_gr, runs10.horizons, "Moon") * C.AU_KM
    w = weights(runs10.split)
    bc_split = barycentre(runs10.split_gr, w)
    drift = np.linalg.norm(bc_split - runs10.emb_gr.body("EMB")[0], axis=1) * C.AU_KM
    vs_horizons = np.linalg.norm(bc_split - barycentre(runs10.horizons, w), axis=1) * C.AU_KM
    _, predicted = predicted_emb_drift_km(runs10.split, runs10.years * C.JULIAN_YEAR_DAYS)
    report = (
        f"10 yr: Moon vs Horizons max {sep_newton.max():.0f} km (Newton), {sep_gr.max():.0f} km "
        f"(Newton + 1PN); EMB drift final {drift[-1]:.0f} km (predicted {predicted:.0f}); "
        f"split barycentre vs Horizons max {vs_horizons.max():.1f} km"
    )
    assert sep_gr.max() < 500.0, report
    assert sep_newton.max() < 2000.0, report
    assert 0.7 < drift[-1] / predicted < 1.4, report
    assert vs_horizons.max() < 100.0, report


@pytest.mark.slow
def test_mercury_gr_rate_unchanged_by_the_moon_ten_years(runs10):
    rate_split, _ = perihelion_rate_difference(runs10.split_gr, runs10.split_newton, "Mercury")
    rate_emb, _ = perihelion_rate_difference(runs10.emb_gr, runs10.emb_newton, "Mercury")
    shift, shift_err = perihelion_rate_difference(
        runs10.split_newton, runs10.emb_newton, "Mercury"
    )
    report = (
        f"10 yr: GR - Newton split {rate_split:.4f}, EMB {rate_emb:.4f}; Newtonian shift "
        f"{shift:+.5f} +- {shift_err:.5f} arcsec/cy"
    )
    assert rate_split == pytest.approx(rate_emb, abs=0.1), report
    assert rate_split == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.1), report
    assert shift == pytest.approx(-0.0025, abs=0.001), report
