"""Tests for gravsim.data.horizons: cached ephemerides, cache round trips, mocked fetches.

The shipped files are checked for internal consistency (snapshot == sample 0 of the series,
GM == constants.GM, 20-day grid over a Julian century) and for physical sanity (barycentre
at the origin, ecliptic frame: planetary inclinations of 7.0, 3.4, 0.0 and 1.85 degrees,
periods from Kepler's third law).  ``fetch_state``/``fetch_series`` are exercised with a
mocked ``_query`` so that the parsing of a Horizons table is tested without any network;
the one real-network test is marked ``@pytest.mark.network`` and deselected by default.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gravsim.analysis import state_to_elements
from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import (
    DEFAULT_BODIES,
    GM,
    GM_SUN,
    HORIZONS_IDS,
    J2000_JD,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
)
from gravsim.data import horizons as H

# Sidereal orbital periods [days] and ecliptic inclinations [deg] (J2000 mean elements,
# Standish & Williams, Explanatory Supplement table 8.10.2).
PERIOD_DAYS = {
    "Mercury": 87.969,
    "Venus": 224.701,
    "EMB": 365.256,
    "Mars": 686.980,
    "Jupiter": 4332.59,
    "Saturn": 10759.2,
    "Uranus": 30688.5,
    "Neptune": 60182.0,
}
INCLINATION_DEG = {"Mercury": 7.005, "Venus": 3.395, "EMB": 0.0, "Mars": 1.850}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def snapshot() -> SystemState:
    return H.load_j2000()


@pytest.fixture(scope="module")
def series() -> Trajectory:
    return H.load_reference_series()


class FakeTable(dict):
    """Minimal stand-in for the astropy Table returned by ``Horizons.vectors``."""


def fake_table(jd: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> FakeTable:
    jd, pos, vel = np.atleast_1d(jd), np.atleast_2d(pos), np.atleast_2d(vel)
    return FakeTable(
        datetime_jd=jd,
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        vx=vel[:, 0],
        vy=vel[:, 1],
        vz=vel[:, 2],
    )


# =========================================================================== shipped cache
class TestShippedSnapshot:
    def test_paths_exist_inside_the_package(self):
        assert H.J2000_PATH.is_file() and H.REF_PATH.is_file()
        assert H.J2000_PATH.parent == H.DATA_DIR == H.REF_PATH.parent

    def test_bodies_epoch_and_gm(self, snapshot):
        assert snapshot.names == DEFAULT_BODIES
        assert snapshot.t == 0.0  # J2000.0
        assert snapshot.n == 9
        for i, name in enumerate(snapshot.names):
            assert snapshot.gm[i] == GM[name], name

    def test_json_metadata(self):
        doc = json.loads(H.J2000_PATH.read_text())
        assert doc["epoch_jd_tdb"] == J2000_JD
        assert doc["units"] == {"length": "AU", "time": "day", "gm": "AU^3/day^2"}
        assert "ecliptic" in doc["frame"].lower() and "barycenter" in doc["frame"].lower()
        ids = [b["horizons_id"] for b in doc["bodies"]]
        assert ids == [HORIZONS_IDS[n] for n in DEFAULT_BODIES]
        # coordinates are stored as repr strings so the round trip is exact
        assert all(isinstance(x, str) for b in doc["bodies"] for x in b["pos"] + b["vel"])

    def test_barycentre_is_at_the_origin(self, snapshot):
        r0, v0 = snapshot.barycenter()
        assert np.linalg.norm(r0) < 2e-6  # AU; the missing Moon/asteroids/Pluto explain ~1e-6
        assert np.linalg.norm(v0) < 1e-9  # AU/day

    def test_sun_is_near_the_barycentre_and_slow(self, snapshot):
        r_sun = np.linalg.norm(snapshot.pos[0])
        assert 0.002 < r_sun < 0.012  # within about one solar radius ... a few
        assert np.linalg.norm(snapshot.vel[0]) < 2e-5

    @pytest.mark.parametrize("name", list(PERIOD_DAYS))
    def test_osculating_periods_match_kepler_third_law(self, snapshot, name):
        r, v = snapshot.relative_to("Sun")
        i = snapshot.index(name)
        el = state_to_elements(r[i], v[i], GM_SUN + snapshot.gm[i])
        # osculating periods differ from the mean ones by up to ~1 % (Jupiter on Saturn)
        assert float(el["period"]) == pytest.approx(PERIOD_DAYS[name], rel=1e-2), name
        assert 0.0 <= float(el["e"]) < 0.25

    @pytest.mark.parametrize("name", list(INCLINATION_DEG))
    def test_frame_is_the_ecliptic(self, snapshot, name):
        """In the ecliptic frame the EMB orbit is flat and Mercury is tilted by 7 degrees."""
        r, v = snapshot.relative_to("Sun")
        i = snapshot.index(name)
        el = state_to_elements(r[i], v[i], GM_SUN + snapshot.gm[i])
        assert math.degrees(float(el["inc"])) == pytest.approx(INCLINATION_DEG[name], abs=0.05)

    def test_mercury_elements(self, snapshot):
        r, v = snapshot.relative_to("Sun")
        i = snapshot.index("Mercury")
        el = state_to_elements(r[i], v[i], GM_SUN + snapshot.gm[i])
        assert float(el["a"]) == pytest.approx(0.38710, abs=2e-4)
        assert float(el["e"]) == pytest.approx(0.20563, abs=2e-4)


class TestShippedSeries:
    def test_grid_is_20_days_over_one_century(self, series):
        assert series.t[0] == 0.0
        assert np.all(np.diff(series.t) == 20.0)
        assert series.t[-1] <= JULIAN_CENTURY_DAYS < series.t[-1] + 20.0
        assert series.n_steps == 1827

    def test_names_gm_and_meta(self, series):
        assert series.names == DEFAULT_BODIES
        np.testing.assert_array_equal(series.gm, [GM[n] for n in DEFAULT_BODIES])
        assert series.meta["source"] == "JPL Horizons" and series.meta["step"] == "20d"
        assert series.pos.shape == series.vel.shape == (1827, 9, 3)

    def test_sample_zero_is_the_snapshot(self, series, snapshot):
        np.testing.assert_array_equal(series.pos[0], snapshot.pos)
        np.testing.assert_array_equal(series.vel[0], snapshot.vel)

    def test_velocities_are_consistent_with_positions(self, series):
        """Central differences of the positions reproduce the velocities.

        The truncation error of the central difference with step h = 20 d on an orbit of
        angular frequency w is ~(w h)^2 / 6 relative: 2e-5 for Jupiter, 2 % for the EMB,
        5 % for Venus and 45 % for Mercury (whose 88-day orbit is barely sampled).
        """
        h = 20.0
        v_fd = (series.pos[2:] - series.pos[:-2]) / (2.0 * h)
        scale = np.abs(series.vel[1:-1]).max(axis=(0, 2))
        for i, name in enumerate(series.names):
            if name == "Sun":
                continue
            err = np.abs(v_fd[:, i] - series.vel[1:-1, i]).max()
            w = 2.0 * math.pi / PERIOD_DAYS[name]
            tol = 2.0 * (w * h) ** 2 / 6.0 + 1e-4  # twice the leading truncation term
            assert err < tol * scale[i], name

    def test_orbits_stay_bound_for_a_century(self, series):
        r = np.linalg.norm(series.pos - series.pos[:, :1, :], axis=2)  # relative to the Sun
        for name, a in (("Mercury", 0.387), ("EMB", 1.0), ("Neptune", 30.1)):
            i = series.index(name)
            assert r[:, i].min() > 0.5 * a and r[:, i].max() < 1.5 * a, name


# =========================================================================== round trips
class TestCacheRoundTrips:
    def test_snapshot_round_trip_is_exact(self, tmp_path, snapshot):
        path = tmp_path / "snap.json"
        shifted = snapshot.with_(t=123.0, pos=snapshot.pos * 1.1, vel=snapshot.vel / 3.0)
        H.save_snapshot(shifted, path, source="test")
        back = H.load_snapshot(path)
        assert back.t == 123.0 and back.names == snapshot.names
        np.testing.assert_array_equal(back.pos, shifted.pos)
        np.testing.assert_array_equal(back.vel, shifted.vel)
        np.testing.assert_array_equal(back.gm, shifted.gm)
        doc = json.loads(path.read_text())
        assert doc["source"] == "test" and doc["epoch_jd_tdb"] == J2000_JD + 123.0

    def test_save_snapshot_does_not_modify_the_state(self, tmp_path, snapshot):
        pos0 = snapshot.pos.copy()
        H.save_snapshot(snapshot, tmp_path / "s.json")
        np.testing.assert_array_equal(snapshot.pos, pos0)

    def test_reference_series_round_trip_is_exact(self, tmp_path):
        t = np.array([0.0, 20.0, 40.0])
        rng = np.random.default_rng(1)
        pos = rng.normal(size=(3, 2, 3))
        vel = rng.normal(size=(3, 2, 3))
        traj = Trajectory(t, pos, vel, np.array([GM_SUN, GM["Mars"]]), ("Sun", "Mars"))
        path = tmp_path / "ref.npz"
        H.save_reference_series(traj, path)
        back = H.load_reference_series(path)
        np.testing.assert_array_equal(back.t, t)
        np.testing.assert_array_equal(back.pos, pos)
        np.testing.assert_array_equal(back.vel, vel)
        np.testing.assert_array_equal(back.gm, traj.gm)
        assert back.names == ("Sun", "Mars") and all(isinstance(n, str) for n in back.names)
        assert back.meta["source"] == "JPL Horizons"

    def test_load_snapshot_accepts_a_string_path(self, tmp_path, snapshot):
        path = tmp_path / "s.json"
        H.save_snapshot(snapshot, path)
        assert H.load_snapshot(str(path)).names == snapshot.names

    def test_load_j2000_is_the_shipped_file(self, snapshot):
        again = H.load_snapshot(H.J2000_PATH)
        np.testing.assert_array_equal(again.pos, snapshot.pos)
        assert again.t == snapshot.t


# =========================================================================== mocked network
class TestFetchWithMockedQuery:
    def test_fetch_state_parses_one_epoch(self, monkeypatch):
        calls: list[tuple[str, list[float]]] = []
        rng = np.random.default_rng(7)
        table = {n: (rng.normal(size=3), rng.normal(size=3)) for n in ("Sun", "Mars", "EMB")}

        def _query(name, epochs, refplane="ecliptic"):
            calls.append((name, epochs))
            assert refplane == "ecliptic"
            p, v = table[name]
            return fake_table(np.array(epochs), p, v)

        monkeypatch.setattr(H, "_query", _query)
        s = H.fetch_state(epoch_jd=J2000_JD + 10.0, names=("Sun", "Mars", "EMB"))
        assert isinstance(s, SystemState)
        assert s.t == 10.0 and s.names == ("Sun", "Mars", "EMB")
        assert [c[0] for c in calls] == ["Sun", "Mars", "EMB"]
        assert all(c[1] == [J2000_JD + 10.0] for c in calls)
        for i, n in enumerate(s.names):
            np.testing.assert_array_equal(s.pos[i], table[n][0])
            np.testing.assert_array_equal(s.vel[i], table[n][1])
            assert s.gm[i] == GM[n]

    def test_fetch_state_defaults(self, monkeypatch):
        seen = []

        def _query(name, epochs, refplane="ecliptic"):
            seen.append(name)
            return fake_table(np.array(epochs), np.zeros(3), np.zeros(3))

        monkeypatch.setattr(H, "_query", _query)
        s = H.fetch_state()
        assert tuple(seen) == DEFAULT_BODIES and s.t == 0.0

    def test_fetch_series_stacks_bodies_on_a_common_grid(self, monkeypatch):
        jd = J2000_JD + np.array([0.0, 20.0, 40.0, 60.0])
        rng = np.random.default_rng(3)
        data = {n: (rng.normal(size=(4, 3)), rng.normal(size=(4, 3))) for n in ("Sun", "Venus")}

        def _query(name, epochs, refplane="ecliptic"):
            assert epochs == {"start": "2000-01-01 12:00", "stop": "2000-03-01", "step": "20d"}
            p, v = data[name]
            return fake_table(jd, p, v)

        monkeypatch.setattr(H, "_query", _query)
        tr = H.fetch_series(stop="2000-03-01", names=("Sun", "Venus"))
        assert isinstance(tr, Trajectory)
        np.testing.assert_array_equal(tr.t, [0.0, 20.0, 40.0, 60.0])
        assert tr.pos.shape == (4, 2, 3) and tr.names == ("Sun", "Venus")
        np.testing.assert_array_equal(tr.pos[:, 1], data["Venus"][0])
        np.testing.assert_array_equal(tr.vel[:, 0], data["Sun"][1])
        np.testing.assert_array_equal(tr.gm, [GM_SUN, GM["Venus"]])
        assert tr.meta["step"] == "20d" and tr.meta["source"] == "JPL Horizons"

    def test_fetch_series_rejects_mismatched_grids(self, monkeypatch):
        def _query(name, epochs, refplane="ecliptic"):
            jd = J2000_JD + np.array([0.0, 20.0]) + (5.0 if name == "Venus" else 0.0)
            return fake_table(jd, np.zeros((2, 3)), np.zeros((2, 3)))

        monkeypatch.setattr(H, "_query", _query)
        with pytest.raises(RuntimeError, match="epoch grid mismatch"):
            H.fetch_series(names=("Sun", "Venus"))

    def test_query_uses_the_horizons_id_and_barycentre(self, monkeypatch):
        """``_query`` builds ``Horizons(id=HORIZONS_IDS[name], location="@0", ...)``."""
        import types

        captured = {}

        class FakeHorizons:
            def __init__(self, id, location, epochs):
                captured.update(id=id, location=location, epochs=epochs)

            def vectors(self, refplane):
                captured["refplane"] = refplane
                return "table"

        fake_pkg = types.ModuleType("astroquery")
        fake_mod = types.ModuleType("astroquery.jplhorizons")
        fake_mod.Horizons = FakeHorizons
        fake_pkg.jplhorizons = fake_mod
        monkeypatch.setitem(__import__("sys").modules, "astroquery", fake_pkg)
        monkeypatch.setitem(__import__("sys").modules, "astroquery.jplhorizons", fake_mod)
        assert H._query("Jupiter", [J2000_JD]) == "table"
        assert captured == {
            "id": HORIZONS_IDS["Jupiter"],
            "location": "@0",
            "epochs": [J2000_JD],
            "refplane": "ecliptic",
        }


# =========================================================================== real network
@pytest.mark.network
def test_fetch_state_matches_the_shipped_snapshot():
    """Live Horizons query (deselected by default): J2000 Sun + Mercury vs the cache."""
    pytest.importorskip("astroquery")
    live = H.fetch_state(names=("Sun", "Mercury"))
    cached = H.load_j2000().subset(("Sun", "Mercury"))
    assert np.abs(live.pos - cached.pos).max() < 1e-7  # AU (~15 km, ephemeris revisions)
    assert np.abs(live.vel - cached.vel).max() < 1e-9
    assert live.t == 0.0 and live.names == cached.names
    assert JULIAN_YEAR_DAYS == 365.25  # keep the import used when the test is deselected
