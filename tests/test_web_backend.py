"""Tests of the web back end: payload format, scenario builders, FastAPI server and export.

Every scenario is built once per module with its smallest parameters (a few seconds in all);
the server tests use ``fastapi.testclient`` (httpx) in process, no network and no browser.  The
shipped defaults are checked too: ``earth_moon`` (2 yr, ~3 s) in the fast suite and the 100-year
``solar_system`` build (~30 s) under ``@pytest.mark.slow`` with the ROADMAP tolerance on
Mercury.  The browser tests of the front end live in ``tests/test_web.py`` (marker ``web``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from gravsim import constants as C
from gravsim.web import payload
from gravsim.web import scenarios as S


def _load_export():
    """Import ``scripts/export_web.py`` from its path (``scripts/`` is not a package)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "export_web.py"
    spec = importlib.util.spec_from_file_location("gravsim_export_web", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SMALL_PARAMS: dict[str, dict] = {
    "solar_system": {"years": 1},
    "rosette": {"n_orbits": 5},
    "black_hole": {"n_orbits": 1},
    "light_rays": {"n_rays": 3},
    "earth_moon": {"years": 0.5},
}


@pytest.fixture(scope="module")
def built() -> dict[str, dict]:
    return {sid: S.build_scenario(sid, **params) for sid, params in SMALL_PARAMS.items()}


# --------------------------------------------------------------------------- payload
def test_payload_round_trip_mixed_dtypes_and_unicode():
    meta = {
        "title": "ϖ of Mercury ☉ — café",
        "nested": {"values": [1, 2.5, None, True], "name": "naïve"},
        "numpy_scalar": np.float64(1.5),
    }
    arrays = {
        "pos": np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3) / 7.0,
        "idx": np.array([0, 5, 17], dtype=np.int64),
        "flags": np.array([[1, -2], [3, 4]], dtype=np.int32),
        "scalar_like": np.array([math.pi], dtype=np.float32),
    }
    data = payload.encode({"meta": meta, "arrays": arrays})
    assert isinstance(data, bytes)
    back = payload.decode(data)
    assert back["meta"]["title"] == meta["title"]
    assert back["meta"]["nested"] == meta["nested"]
    assert back["meta"]["numpy_scalar"] == 1.5
    assert list(back["arrays"]) == list(arrays)  # order preserved
    assert back["arrays"]["pos"].dtype == np.float32 and back["arrays"]["pos"].shape == (2, 3, 3)
    np.testing.assert_allclose(back["arrays"]["pos"], arrays["pos"].astype(np.float32))
    assert back["arrays"]["idx"].dtype == np.int32
    np.testing.assert_array_equal(back["arrays"]["idx"], arrays["idx"])
    np.testing.assert_array_equal(back["arrays"]["flags"], arrays["flags"])
    assert back["arrays"]["scalar_like"][0] == np.float32(math.pi)
    for arr in back["arrays"].values():
        assert arr.flags.c_contiguous and arr.dtype.isnative


def test_payload_layout_header_and_alignment():
    data = payload.encode({"meta": {"k": "v"}, "arrays": {"a": np.zeros(3), "b": np.ones(2, int)}})
    header_len = int(np.frombuffer(data[:4], dtype="<u4")[0])
    assert (4 + header_len) % 4 == 0
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    assert header["version"] == payload.VERSION and header["meta"] == {"k": "v"}
    index = header["index"]
    assert [e["name"] for e in index] == ["a", "b"]
    assert index[0] == {"name": "a", "dtype": "float32", "shape": [3], "offset": 0, "nbytes": 12}
    assert index[1]["dtype"] == "int32" and index[1]["offset"] == 12 and index[1]["nbytes"] == 8
    assert len(data) == 4 + header_len + 20
    # offsets are relative to the array block: the raw bytes of "b" sit right there
    block = data[4 + header_len :]
    assert np.frombuffer(block[12:20], dtype="<i4").tolist() == [1, 1]


def test_payload_empty_and_errors():
    empty = payload.decode(payload.encode({"meta": {}, "arrays": {}}))
    assert empty == {"meta": {}, "arrays": {}}
    with pytest.raises(TypeError):
        payload.encode({"meta": {}, "arrays": {"bad": np.array([True, False])}})
    data = payload.encode({"meta": {"x": 1}, "arrays": {"a": np.arange(4.0)}})
    with pytest.raises(ValueError):
        payload.decode(data[:-3])  # truncated array block
    with pytest.raises(ValueError):
        payload.decode(b"\x00\x00")  # shorter than the length field
    with pytest.raises(ValueError):
        payload.decode(b"\xff\xff\xff\x7f" + b"{}")  # header length beyond the payload


# --------------------------------------------------------------------------- scenario registry
def test_list_scenarios_entries():
    entries = S.list_scenarios()
    assert [e["id"] for e in entries] == [
        "solar_system",
        "rosette",
        "black_hole",
        "light_rays",
        "earth_moon",
    ]
    for entry in entries:
        assert entry["title"] and entry["blurb"]
        for name, spec in entry["params"].items():
            assert spec["type"] in ("float", "int", "bool") and "default" in spec
            if spec["type"] != "bool":
                assert spec["min"] <= spec["default"] <= spec["max"], name
        json.dumps(entry)  # plain JSON
    assert S.default_params("solar_system") == {"years": 100.0, "step_days": 2.0}
    assert S.default_params("light_rays") == {"n_rays": 9, "compact": True}


@pytest.mark.parametrize(
    "scenario_id, params, message",
    [
        ("nope", {}, "unknown scenario"),
        ("solar_system", {"years": 0}, "below the minimum"),
        ("solar_system", {"years": 201}, "above the maximum"),
        ("solar_system", {"foo": 1}, "unknown parameter"),
        ("rosette", {"n_orbits": "abc"}, "must be a number"),
        ("rosette", {"n_orbits": 5.5}, "must be an integer"),
        ("rosette", {"n_orbits": float("nan")}, "finite"),
        ("light_rays", {"compact": "maybe"}, "boolean"),
        ("black_hole", {"peri": True}, "must be a number"),
    ],
)
def test_validate_params_rejects(scenario_id, params, message):
    with pytest.raises(ValueError, match=message):
        S.validate_params(scenario_id, params)
    with pytest.raises(ValueError):
        S.build_scenario(scenario_id, **params)


def test_validate_params_coerces_query_strings():
    assert S.validate_params("rosette", {"n_orbits": "5", "c_scale": "0.01"}) == {
        "c_scale": 0.01,
        "n_orbits": 5,
        "step_days": 0.25,
    }
    assert S.validate_params("rosette", {"n_orbits": 5.0})["n_orbits"] == 5
    assert S.validate_params("light_rays", {"compact": "false"}) == {"n_rays": 9, "compact": False}
    assert S.validate_params("light_rays", {"compact": 1})["compact"] is True


# --------------------------------------------------------------------------- builders
@pytest.mark.parametrize("scenario_id", list(SMALL_PARAMS))
def test_builders_document_every_array(built, scenario_id):
    result = built[scenario_id]
    meta, arrays = result["meta"], result["arrays"]
    assert meta["id"] == scenario_id and meta["title"] and meta["blurb"]
    assert meta["params"] == S.validate_params(scenario_id, SMALL_PARAMS[scenario_id])
    assert set(arrays) == set(meta["arrays"])
    for name, arr in arrays.items():
        doc = meta["arrays"][name]
        assert arr.dtype in (np.float32, np.int32)
        assert doc["dtype"] == str(arr.dtype)
        assert doc["shape"] == list(arr.shape)
        assert doc["description"]
        assert np.all(np.isfinite(arr))
    if "t" in arrays:
        assert meta["n_samples"] == arrays["t"].shape[0]
        assert np.all(np.diff(arrays["t"]) > 0)
    json.dumps(meta)  # every meta value must be JSON-serialisable
    data = payload.encode(result)
    back = payload.decode(data)
    assert back["meta"]["id"] == scenario_id
    for name, arr in arrays.items():
        np.testing.assert_array_equal(back["arrays"][name], arr)


def test_solar_system_small(built):
    res = built["solar_system"]
    arrays, meta = res["arrays"], res["meta"]
    n_t = int(round(C.JULIAN_YEAR_DAYS / 2.0)) + 1
    assert arrays["t"].shape == (n_t,) and arrays["t"][0] == 0.0
    assert arrays["pos_newton"].shape == arrays["pos_gr"].shape == (n_t, 9, 3)
    assert arrays["delta_gr"].shape == (n_t, 9, 3)
    assert meta["names"] == list(C.DEFAULT_BODIES)
    assert meta["units"] == {"length": "AU", "time": "day"}
    # Mercury completes ~4 orbits in a year: 4 perihelia, as sample indices in range
    for key in ("peri_idx_newton", "peri_idx_gr"):
        idx = arrays[key]
        assert idx.dtype == np.int32 and 3 <= idx.size <= 5
        assert np.all((idx > 0) & (idx < n_t)) and np.all(np.diff(idx) > 0)
        r = np.linalg.norm(arrays["pos_gr"][:, 1, :] - arrays["pos_gr"][:, 0, :], axis=1)
        assert np.all(r[idx] < 0.32)  # Mercury's perihelion distance is 0.307 AU
    merc = meta["perihelion_rates"]["Mercury"]
    assert merc["analytic_1pn_arcsec_cy"] == pytest.approx(42.98, abs=0.05)
    assert merc["gr_minus_newton_arcsec_cy"] == pytest.approx(42.98, abs=5.0)  # one year only
    assert set(meta["perihelion_rates"]) == {"Mercury", "Venus", "EMB", "Mars"}
    assert 0 < meta["mercury"]["final_separation_km"] < 1000
    assert 0.003 < meta["sun_wobble"]["amplitude_au"] < 0.012
    assert meta["sun_wobble"]["amplitude_r_sun"] == pytest.approx(
        meta["sun_wobble"]["amplitude_au"] / C.R_SUN_AU
    )
    assert meta["models"]["gr"]["law"] == "eih" and meta["models"]["newton"]["law"]
    # delta_gr is the float64 difference, not the difference of the float32 positions
    delta = arrays["pos_gr"].astype(np.float64) - arrays["pos_newton"].astype(np.float64)
    assert np.abs(arrays["delta_gr"] - delta).max() < 1e-6


def test_rosette_small(built):
    res = built["rosette"]
    arrays, meta = res["arrays"], res["meta"]
    assert meta["names"] == ["Sun", "Mercury"] and arrays["pos_gr"].shape[1:] == (2, 3)
    assert meta["c_scale"] == 0.01 and meta["c_au_day"] == pytest.approx(C.C_AU_DAY * 0.01)
    assert arrays["peri_idx_gr"].size == 5 and arrays["peri_idx_newton"].size == 5
    adv = meta["advance"]
    analytic = adv["analytic_1pn_rad_per_orbit"]
    orbit = meta["orbit"]
    assert analytic == pytest.approx(
        6 * math.pi * orbit["a_au"] ** 0 * (C.GM_SUN + C.GM["Mercury"])
        / ((C.C_AU_DAY * 0.01) ** 2 * orbit["a_au"] * (1 - orbit["e"] ** 2))
    )
    assert analytic == pytest.approx(5.02e-7 * 1e4, rel=0.02)
    assert adv["passages_rad_per_orbit"] == pytest.approx(analytic, rel=0.02)
    assert adv["fit_rad_per_orbit"] == pytest.approx(analytic, rel=0.1)  # 5 orbits: wobble bias
    assert adv["real_mercury_arcsec_cy"] == pytest.approx(42.98, abs=0.05)
    # barycentric: the Sun moves, the system's centre of mass stays at the origin
    com = np.einsum("n,tnk->tk", np.asarray(meta["gm"]), arrays["pos_gr"].astype(np.float64))
    assert np.abs(com).max() / sum(meta["gm"]) < 1e-6


def test_black_hole_small(built):
    res = built["black_hole"]
    arrays, meta = res["arrays"], res["meta"]
    g = arrays["geodesic"]
    assert g.shape[1] == 3 and np.all(g[:, 2] == 0.0)
    r = np.hypot(g[:, 0], g[:, 1])
    assert meta["periapsis"] == {
        "requested": 4.6,
        "used": 4.6,
        "separatrix": pytest.approx(80 / 18),
        "clamped": False,
    }
    assert meta["r_apo"] == pytest.approx(20.0) and meta["r_peri"] == pytest.approx(4.6, rel=1e-4)
    assert r.min() == pytest.approx(4.6, rel=1e-3) and r[0] == pytest.approx(20.0, rel=1e-6)
    assert r[-1] == pytest.approx(20.0, rel=1e-3)  # one radial period: back at apoapsis
    assert arrays["apo_idx"].tolist() == [g.shape[0] - 1] and arrays["peri_idx"].size == 1
    assert 0 < meta["E"] < 1 and meta["L"] == pytest.approx(S.zoom_whirl_angular_momentum(20, 4.6))
    assert (meta["r_horizon"], meta["r_photon_sphere"], meta["r_isco"]) == (2.0, 3.0, 6.0)
    assert meta["r_circular_unstable"] < 4.6 < meta["r_circular_stable"] < 20
    assert meta["advance_per_orbit_rad"] > 2 * math.pi  # a zoom-whirl: more than one extra turn
    turns = 1 + meta["advance_per_orbit_rad"] / (2 * math.pi)
    assert meta["turns_per_orbit"] == pytest.approx(turns)
    t = arrays["t"]
    assert t[0] == 0.0 and np.all(np.diff(t) > 0) and t[-1] > meta["tau_end"]  # time dilation
    assert meta["units"] == {"length": "GM/c^2", "time": "GM/c^3"}


def test_black_hole_clamps_unreachable_periapsis():
    res = S.build_scenario("black_hole", r0=20, peri=3.5, n_orbits=2)
    meta = res["meta"]
    assert meta["periapsis"]["clamped"] is True
    assert meta["periapsis"]["requested"] == 3.5
    assert meta["periapsis"]["used"] == pytest.approx(S.separatrix_periapsis(20) * 1.01)
    assert meta["r_peri"] == pytest.approx(meta["periapsis"]["used"], rel=1e-4)
    assert res["arrays"]["apo_idx"].size == 2
    assert meta["turns_per_orbit"] > 2
    with pytest.raises(ValueError):
        S.zoom_whirl_angular_momentum(20, 2.5)  # inside the photon sphere
    with pytest.raises(ValueError):
        S.separatrix_periapsis(2.0)


def test_light_rays_compact(built):
    res = built["light_rays"]
    arrays, meta = res["arrays"], res["meta"]
    assert set(arrays) == {"ray_0", "ray_1", "ray_2", "b_values"}
    assert meta["compact"] is True and meta["exaggeration"] == 1.0
    assert meta["b_crit"] == pytest.approx(3 * math.sqrt(3))
    b = arrays["b_values"]
    assert b.shape == (3,) and b[0] == pytest.approx(1.05 * meta["b_crit"], rel=1e-6)
    assert b[-1] == pytest.approx(12.0, rel=1e-6)
    assert np.all(np.diff(np.log(b)) > 0) and np.allclose(np.diff(np.log(b)), np.diff(np.log(b))[0])
    for k in range(3):
        ray = arrays[f"ray_{k}"]
        assert ray.shape == (S._RAYS_POINTS, 3) and np.all(ray[:, 2] == 0.0)
        # enters from -x along y = b (already bent by ~0.1 % at the cut radius r_max = 30)
        assert ray[0, 0] < 0 and ray[0, 1] == pytest.approx(b[k], rel=2e-2)
        r = np.hypot(ray[:, 0], ray[:, 1])
        assert r.min() == pytest.approx(meta["closest_approach"][k], rel=1e-3)
        assert r.min() > 3.0  # outside the photon sphere
    d = meta["deflection_rad"]
    assert d[0] > d[1] > d[2] > 0 and d[0] > 1.0  # strong bending close to b_crit
    assert meta["deflection_arcsec"][2] == pytest.approx(d[2] * C.ARCSEC_PER_RAD)
    assert meta["deflection_weak_rad"][2] == pytest.approx(4.0 / 12.0)


def test_light_rays_sun_exaggerated():
    res = S.build_scenario("light_rays", n_rays=3, compact=False)
    arrays, meta = res["arrays"], res["meta"]
    assert meta["compact"] is False and meta["units"] == {"length": "R_sun"}
    assert meta["exaggeration"] == 20000.0 and meta["body_radius"] == 1.0
    assert arrays["b_values"].tolist() == pytest.approx([1.0, 3.0, 5.0])
    assert meta["deflection_arcsec"][0] == pytest.approx(C.LIGHT_DEFLECTION_LIMB_ARCSEC, abs=1e-3)
    assert meta["limb_deflection_arcsec"] == pytest.approx(1.7512, abs=1e-3)
    ray = arrays["ray_0"]
    # the drawn ray bends by exaggeration * deflection: its outgoing arm drops below y = 1
    assert ray[0, 1] == pytest.approx(1.0, rel=1e-3)
    assert ray[-1, 1] < 1.0 - 0.5  # 20000 * 8.5e-6 rad * ~10 R_sun ~ 1.7 R_sun of drop
    assert np.hypot(ray[:, 0], ray[:, 1]).max() <= meta["r_max"] * 1.5


def test_earth_moon_small(built):
    res = built["earth_moon"]
    arrays, meta = res["arrays"], res["meta"]
    assert meta["names"] == list(C.DEFAULT_BODIES_MOON)
    assert arrays["pos_gr"].shape == (int(round(0.5 * C.JULIAN_YEAR_DAYS / 0.1)) + 1, 10, 3)
    moon = meta["moon"]
    assert moon["mean_distance_km"] == pytest.approx(384_400, abs=3000)
    assert 350_000 < moon["min_distance_km"] < moon["max_distance_km"] < 410_000
    assert moon["sidereal_period_days"] == pytest.approx(27.3217, abs=0.05)
    assert moon["anomalistic_period_days"] == pytest.approx(27.5545, abs=0.4)  # evection bias
    assert 5 <= moon["n_perigee_passages"] <= 8
    assert moon["perigee_rate_reference_deg_yr"] == C.MOON_PERIGEE_RATE_DEG_YR
    assert moon["node_rate_reference_deg_yr"] == C.MOON_NODE_RATE_DEG_YR
    assert math.isfinite(moon["perigee_rate_deg_yr"]) and math.isfinite(moon["node_rate_deg_yr"])
    assert moon["node_rate_deg_yr"] == pytest.approx(-19.34, abs=10)  # half a year is noisy
    assert 4.5 < moon["inc_mean_deg"] < 5.8


# --------------------------------------------------------------------------- shipped defaults
def test_earth_moon_defaults_match_lunar_theory():
    """The 2-year scenario the page ships: months, apsidal and nodal rates vs the references."""
    res = S.build_scenario("earth_moon")
    arrays, meta = res["arrays"], res["meta"]
    assert meta["params"] == S.default_params("earth_moon") == {"years": 2.0, "step_days": 0.1}
    n_t = int(round(2.0 * C.JULIAN_YEAR_DAYS / 0.1)) + 1
    assert arrays["pos_gr"].shape == arrays["pos_newton"].shape == (n_t, 10, 3)
    moon = meta["moon"]
    # mean motions over two years: the months to a few seconds / a few minutes
    assert moon["sidereal_period_days"] == pytest.approx(S.MOON_SIDEREAL_MONTH_DAYS, abs=2e-3)
    assert moon["anomalistic_period_days"] == pytest.approx(S.MOON_ANOMALISTIC_MONTH_DAYS, abs=1e-2)
    assert moon["anomalistic_period_from_passages_days"] == pytest.approx(
        S.MOON_ANOMALISTIC_MONTH_DAYS, abs=0.2
    )
    assert moon["n_perigee_passages"] == 26  # 730.5 d / 27.55 d
    # secular rates: +40.69 and -19.34 deg/yr; two years of evection and variation leave
    # ~0.5 deg/yr on the perigee (its fit error is ~0.3) and ~0.1 deg/yr on the node
    assert moon["perigee_rate_deg_yr"] == pytest.approx(C.MOON_PERIGEE_RATE_DEG_YR, abs=1.0)
    assert 0 < moon["perigee_rate_err_deg_yr"] < 0.5
    assert moon["node_rate_deg_yr"] == pytest.approx(C.MOON_NODE_RATE_DEG_YR, abs=0.2)
    assert 0 < moon["node_rate_err_deg_yr"] < 0.05
    assert moon["argp_rate_deg_yr"] == pytest.approx(
        moon["perigee_rate_deg_yr"] - moon["node_rate_deg_yr"], abs=0.5
    )
    assert moon["mean_distance_km"] == pytest.approx(S.EARTH_MOON_MEAN_DISTANCE_KM, rel=3e-3)
    assert 356_000 < moon["min_distance_km"] < 358_000
    assert 405_000 < moon["max_distance_km"] < 407_000
    assert moon["e_mean"] == pytest.approx(0.0549, abs=2e-3)
    assert moon["inc_mean_deg"] == pytest.approx(5.145, abs=0.05)
    # Mercury rides along: a two-year fit is biased by long-period terms but still at 1 %
    merc = meta["mercury"]
    assert merc["gr_minus_newton_arcsec_cy"] == pytest.approx(
        C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.5
    )
    assert 0 < merc["err_arcsec_cy"] < 0.3


@pytest.mark.slow
def test_solar_system_defaults_recover_mercury_precession():
    """The 100-year scenario the page ships: Mercury 42.98 +/- 0.1 "/cy (ROADMAP section 4).

    The GR rate is the run-vs-run difference of the fitted perihelion drifts, which cancels the
    ~-2.4 "/cy bias of the absolute Newtonian fit; Venus, EMB and Mars must agree with their 1PN
    values within a few fit sigmas.
    """
    res = S.build_scenario("solar_system")
    arrays, meta = res["arrays"], res["meta"]
    assert meta["params"] == S.default_params("solar_system") == {"years": 100.0, "step_days": 2.0}
    n_t = int(round(100.0 * C.JULIAN_YEAR_DAYS / 2.0)) + 1
    assert arrays["pos_gr"].shape == arrays["pos_newton"].shape == (n_t, 9, 3)
    # round(36525 / 2) + 1 samples: the grid stops one step short of the exact century
    assert meta["span_days"] == pytest.approx((n_t - 1) * 2.0)
    assert abs(meta["span_days"] - C.JULIAN_CENTURY_DAYS) <= 2.0
    rates = meta["perihelion_rates"]
    merc = rates["Mercury"]
    analytic = merc["analytic_1pn_arcsec_cy"]
    assert analytic == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.05)
    assert merc["gr_minus_newton_arcsec_cy"] == pytest.approx(analytic, abs=0.1)
    assert 0 < merc["err_arcsec_cy"] < 0.01
    assert meta["mercury"]["gr_minus_newton_arcsec_cy"] == merc["gr_minus_newton_arcsec_cy"]
    for body, expected in (("Venus", 8.62), ("EMB", 3.84), ("Mars", 1.35)):
        entry = rates[body]
        assert entry["analytic_1pn_arcsec_cy"] == pytest.approx(expected, abs=0.01), body
        diff = entry["gr_minus_newton_arcsec_cy"] - entry["analytic_1pn_arcsec_cy"]
        assert abs(diff) < 4 * entry["err_arcsec_cy"], body  # a few sigma of the fit
        assert abs(diff) < 0.03, body  # and below 1 % in any case
    # 100 yr / 87.97 d = 415.2 perihelia, all of them found in both runs
    assert meta["mercury"]["n_perihelia_gr"] == 415
    assert arrays["peri_idx_gr"].size == arrays["peri_idx_newton"].size == 415
    assert np.all((arrays["peri_idx_gr"] > 0) & (arrays["peri_idx_gr"] < n_t))
    r = np.linalg.norm(arrays["pos_gr"][:, 1, :] - arrays["pos_gr"][:, 0, :], axis=1)
    assert np.all(r[arrays["peri_idx_gr"]] < 0.32)  # perihelion distance 0.307 AU
    # the along-track GR shift of Mercury after a century is of order a * 43" ~ 10^4 km
    merc_sep = meta["mercury"]
    assert 0 < merc_sep["final_separation_km"] <= merc_sep["max_separation_km"]
    assert 5_000 < merc_sep["max_separation_km"] < 50_000
    # the Sun orbits the barycentre at about one Jupiter-plus-Saturn solar radius
    assert 1.5 < meta["sun_wobble"]["amplitude_r_sun"] < 2.5


def test_builders_are_deterministic():
    a = S.build_scenario("rosette", n_orbits=5)
    b = S.build_scenario("rosette", n_orbits=5)
    assert payload.encode(a) == payload.encode(b)


# --------------------------------------------------------------------------- server
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gravsim.web.server import create_app

    del fastapi
    web_dir = tmp_path_factory.mktemp("web")
    (web_dir / "index.html").write_text("<!doctype html><title>placeholder</title>gravsim test")
    (web_dir / "app.js").write_text("// placeholder\n")
    with TestClient(create_app(web_dir=web_dir)) as c:
        yield c


def test_server_health_and_scenarios(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and set(body["scenarios"]) == set(S.SCENARIOS)
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    entries = r.json()
    assert [e["id"] for e in entries] == [e["id"] for e in S.list_scenarios()]
    for entry in entries:
        assert entry["url"] == f"/api/scenario/{entry['id']}"
        assert entry["title"] and isinstance(entry["params"], dict)


def test_server_scenario_payload_and_cache(client):
    r = client.get("/api/scenario/rosette", params={"n_orbits": 5})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-gravsim-cache"] == "miss"
    res = payload.decode(r.content)
    assert res["meta"]["id"] == "rosette" and res["meta"]["params"]["n_orbits"] == 5
    assert res["arrays"]["pos_gr"].shape[1:] == (2, 3)
    r2 = client.get("/api/scenario/rosette?n_orbits=5")
    assert r2.status_code == 200 and r2.headers["x-gravsim-cache"] == "hit"
    assert r2.content == r.content
    r3 = client.get("/api/scenario/rosette?n_orbits=5.0&c_scale=0.01")  # same validated params
    assert r3.headers["x-gravsim-cache"] == "hit" and r3.content == r.content
    assert client.get("/api/health").json()["cache_entries"] == 1


def test_server_errors(client):
    assert client.get("/api/scenario/nope").status_code == 404
    for query in ("years=abc", "years=0", "foo=1", "step_days=nan"):
        r = client.get(f"/api/scenario/solar_system?{query}")
        assert r.status_code in (400, 422), query
        assert r.json()["detail"]
    assert client.get("/api/scenario/light_rays?compact=maybe").status_code in (400, 422)


def test_server_serves_static_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "gravsim test" in r.text
    assert "text/html" in r.headers["content-type"]
    assert client.get("/app.js").status_code == 200
    assert client.get("/missing.js").status_code == 404


def test_server_without_web_dir(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gravsim.web.server import create_app

    with TestClient(create_app(web_dir=tmp_path / "does-not-exist")) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/").status_code == 404


def test_scenario_cache_unit():
    from gravsim.web.server import ScenarioCache

    cache = ScenarioCache(maxsize=1)
    a, hit_a = cache.get_or_build("light_rays", S.validate_params("light_rays", {"n_rays": 2}))
    assert hit_a is False and len(cache) == 1
    b, hit_b = cache.get_or_build("light_rays", {"n_rays": 2, "compact": True})
    assert hit_b is True and a == b
    cache.get_or_build("light_rays", {"n_rays": 1, "compact": True})
    assert len(cache) == 1  # evicted the older entry
    cache.clear()
    assert len(cache) == 0


# --------------------------------------------------------------------------- export
def test_export_web_writes_manifest_and_files(tmp_path):
    export_web = _load_export()
    out = tmp_path / "data"
    lines: list[str] = []
    overrides = {
        "rosette": {"n_orbits": 5},
        "light_rays": {"n_rays": 2},
        "black_hole": {"n_orbits": 1},
    }
    manifest = export_web.export(
        out, only=["rosette", "light_rays", "black_hole"], overrides=overrides, log=lines.append
    )
    on_disk = json.loads((out / "manifest.json").read_text())
    # the front end requires the same shape as /api/scenarios: a bare, non-empty list of
    # entries with string ``id`` and ``url`` (``web/app.js`` ``loadManifest``)
    assert isinstance(on_disk, list) and on_disk == manifest and len(manifest) == 3
    # registry order is kept regardless of the order of ``only``
    assert [e["id"] for e in manifest] == ["rosette", "black_hole", "light_rays"]
    registry = {e["id"]: e for e in S.list_scenarios()}
    for entry in manifest:
        assert isinstance(entry["id"], str) and isinstance(entry["url"], str)
        assert entry["url"] == f"data/{entry['id']}.bin"
        path = out.parent / entry["url"]  # urls are relative to the web root
        assert path.is_file()
        res = payload.decode(path.read_bytes())
        assert res["meta"]["id"] == entry["id"]
        assert res["meta"]["params"] == entry["built_with"]
        assert entry["params"] == S.SCENARIOS[entry["id"]].to_dict()["params"]
        # same entry as the registry (and thus /api/scenarios) apart from url and built_with
        extra = {k: v for k, v in entry.items() if k not in ("url", "built_with")}
        assert extra == registry[entry["id"]]
    assert len(lines) == 4 and lines[-1].startswith("manifest:")
    assert sorted(p.name for p in out.iterdir()) == [
        "black_hole.bin",
        "light_rays.bin",
        "manifest.json",
        "rosette.bin",
    ]


def test_export_web_cli_parses_set(tmp_path):
    export_web = _load_export()
    assert export_web._parse_set(["rosette.n_orbits=5", "rosette.c_scale=0.1"]) == {
        "rosette": {"n_orbits": "5", "c_scale": "0.1"}
    }
    with pytest.raises(SystemExit):
        export_web._parse_set(["bad"])
    rc = export_web.main(
        ["--out", str(tmp_path / "d"), "--only", "light_rays", "--set", "light_rays.n_rays=1"]
    )
    assert rc == 0 and (tmp_path / "d" / "light_rays.bin").is_file()
    manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
    assert isinstance(manifest, list) and len(manifest) == 1
    assert manifest[0]["id"] == "light_rays" and manifest[0]["url"] == "d/light_rays.bin"
    assert manifest[0]["built_with"] == {"n_rays": 1, "compact": True}
