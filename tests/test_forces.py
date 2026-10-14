"""Tests for gravsim.forces (newton, pn1, combine). No network, all fast."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from gravsim.constants import C_AU_DAY, GM_SUN
from gravsim.data.horizons import load_j2000
from gravsim.forces import (
    HAVE_NUMBA,
    accel_newton,
    accel_newton_numba,
    accel_newton_numpy,
    combine,
    make_newton_pn1,
    make_pn1,
)
from gravsim.forces.newton import check_inputs

# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261013)


def random_system(rng: np.random.Generator, n: int, n_test: int = 0):
    """Random positions in a ~10 AU box, velocities ~0.01 AU/day, log-uniform GMs."""
    pos = rng.uniform(-10.0, 10.0, size=(n, 3))
    vel = rng.normal(0.0, 0.01, size=(n, 3))
    gm = 10.0 ** rng.uniform(-11.0, -4.0, size=n)
    if n_test:
        gm[-n_test:] = 0.0
    return pos, vel, gm


@pytest.fixture
def j2000():
    return load_j2000()


def pn1_scalar(r, v, gm_c, c, beta, gamma):
    """Hand-written scalar 1PN correction with plain Python floats."""
    rx, ry, rz = (float(x) for x in r)
    vx, vy, vz = (float(x) for x in v)
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    v2 = vx * vx + vy * vy + vz * vz
    rv = rx * vx + ry * vy + rz * vz
    pref = gm_c / (c * c * rmag**3)
    a = 2.0 * (beta + gamma) * gm_c / rmag - gamma * v2
    b = 2.0 * (1.0 + gamma) * rv
    return np.array([pref * (a * rx + b * vx), pref * (a * ry + b * vy), pref * (a * rz + b * vz)])


# --------------------------------------------------------------------------- newton


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not importable")
@pytest.mark.parametrize("n,n_test", [(2, 0), (5, 0), (9, 2), (30, 5)])
def test_numba_kernel_matches_numpy_fallback(rng, n, n_test):
    pos, vel, gm = random_system(rng, n, n_test)
    a_nb = accel_newton_numba(pos, vel, gm, 0.0)
    a_np = accel_newton_numpy(pos, vel, gm, 0.0)
    scale = np.abs(a_np).max()
    assert np.abs(a_nb - a_np).max() <= 1e-14 * scale
    assert a_nb.shape == (n, 3) and a_np.shape == (n, 3)


def test_accel_newton_dispatch_matches_both(rng):
    pos, vel, gm = random_system(rng, 7)
    a = accel_newton(pos, vel, gm, 0.0)
    a_np = accel_newton_numpy(pos, vel, gm, 0.0)
    assert np.allclose(a, a_np, rtol=1e-14, atol=1e-14 * np.abs(a_np).max())
    assert a.dtype == np.float64 and a.flags.c_contiguous


def test_two_body_analytic(rng):
    pos, vel, gm = random_system(rng, 2)
    d = pos[0] - pos[1]
    r3 = np.linalg.norm(d) ** 3
    expected = np.array([-gm[1] * d / r3, +gm[0] * d / r3])
    for fn in (accel_newton, accel_newton_numpy):
        np.testing.assert_allclose(fn(pos, vel, gm, 0.0), expected, rtol=1e-14)


def test_sun_mercury_scale():
    """Sanity check of units: a ~ GM_sun / r^2 at 1 AU is ~3e-4 AU/day^2."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    gm = np.array([GM_SUN, 0.0])
    a = accel_newton(pos, vel, gm, 0.0)
    np.testing.assert_allclose(a[1], [-GM_SUN, 0.0, 0.0], rtol=1e-15)
    assert np.all(a[0] == 0.0)


@pytest.mark.parametrize("fn", [accel_newton, accel_newton_numpy])
def test_momentum_conservation(rng, fn):
    pos, vel, gm = random_system(rng, 12, n_test=2)
    a = fn(pos, vel, gm, 0.0)
    total = (gm[:, None] * a).sum(axis=0)
    scale = (gm[:, None] * np.abs(a)).sum(axis=0).max()
    assert np.abs(total).max() <= 1e-13 * scale


@pytest.mark.parametrize("fn", [accel_newton, accel_newton_numpy])
def test_test_particle_feels_but_does_not_exert(rng, fn):
    pos, vel, gm = random_system(rng, 4, n_test=1)
    gm[0] = GM_SUN
    a = fn(pos, vel, gm, 0.0)
    # same system without the test particle: massive bodies are unaffected
    a_wo = fn(pos[:-1], vel[:-1], gm[:-1], 0.0)
    np.testing.assert_array_equal(a[:-1], a_wo)
    # the test particle feels everything
    d = pos[-1] - pos[:-1]
    expected = -(gm[:-1, None] * d / np.linalg.norm(d, axis=1)[:, None] ** 3).sum(axis=0)
    np.testing.assert_allclose(a[-1], expected, rtol=1e-13)


def test_test_particle_does_not_accelerate_sun():
    pos = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.05]])
    vel = np.array([[0.0, 0.0, 0.0], [0.01, 0.02, 0.0]])
    gm = np.array([GM_SUN, 0.0])
    for fn in (accel_newton, accel_newton_numpy):
        a = fn(pos, vel, gm, 0.0)
        assert np.all(a[0] == 0.0)
        assert np.linalg.norm(a[1]) > 0.0


@pytest.mark.parametrize("fn", [accel_newton, accel_newton_numpy])
def test_newton_is_pure_and_ignores_vel_t(rng, fn):
    pos, vel, gm = random_system(rng, 6)
    pos0, vel0, gm0 = pos.copy(), vel.copy(), gm.copy()
    a1 = fn(pos, vel, gm, 0.0)
    a2 = fn(pos, 100.0 * vel, gm, 12345.6)
    a3 = fn(pos, vel, gm)
    np.testing.assert_array_equal(pos, pos0)
    np.testing.assert_array_equal(vel, vel0)
    np.testing.assert_array_equal(gm, gm0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(a1, a3)
    assert not np.shares_memory(a1, pos)


def test_newton_accepts_non_float_non_contiguous_inputs():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.int64)
    vel = np.zeros((3, 3))
    gm = np.array([1e-4, 1e-6, 1e-8])
    big = np.zeros((3, 6))
    big[:, ::2] = pos
    a_ref = accel_newton_numpy(pos.astype(float), vel, gm, 0.0)
    for fn in (accel_newton, accel_newton_numpy):
        np.testing.assert_allclose(fn(pos, vel, gm, 0.0), a_ref, rtol=1e-14)
        np.testing.assert_allclose(fn(big[:, ::2], vel, gm, 0.0), a_ref, rtol=1e-14)


# --------------------------------------------------------------------------- pn1


def test_pn1_vs_hand_written_scalar(rng):
    c = 50.0  # exaggerated so the correction is far above roundoff
    beta, gamma = 0.8, 1.3
    pos, vel, gm = random_system(rng, 5)
    gm[0] = GM_SUN
    fn = make_pn1(c=c, central=0, beta=beta, gamma=gamma)
    a = fn(pos, vel, gm, 0.0)
    assert a.shape == (5, 3) and a.dtype == np.float64
    assert np.all(a[0] == 0.0)
    for i in range(1, 5):
        expected = pn1_scalar(pos[i] - pos[0], vel[i] - vel[0], gm[0], c, beta, gamma)
        np.testing.assert_allclose(a[i], expected, rtol=1e-13)


def test_pn1_gr_equals_explicit_schwarzschild_form(rng):
    pos, vel, gm = random_system(rng, 6)
    gm[0] = GM_SUN
    c = C_AU_DAY
    a = make_pn1(c=c)(pos, vel, gm, 0.0)  # defaults: central=0, beta=gamma=1
    r = pos - pos[0]
    v = vel - vel[0]
    rmag = np.linalg.norm(r, axis=1)
    v2 = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    expected = np.zeros_like(pos)
    for i in range(1, 6):
        expected[i] = (gm[0] / (c**2 * rmag[i] ** 3)) * (
            (4.0 * gm[0] / rmag[i] - v2[i]) * r[i] + 4.0 * rv[i] * v[i]
        )
    np.testing.assert_allclose(a, expected, rtol=1e-13, atol=0.0)


def test_pn1_central_index_other_than_zero(rng):
    pos, vel, gm = random_system(rng, 5)
    gm[0] = GM_SUN
    a0 = make_pn1(central=0)(pos, vel, gm, 0.0)
    perm = np.array([3, 1, 0, 4, 2])  # the Sun moves to row 2
    a2 = make_pn1(central=2)(pos[perm], vel[perm], gm[perm], 0.0)
    np.testing.assert_allclose(a2, a0[perm], rtol=1e-15, atol=0.0)
    assert np.all(a2[2] == 0.0)
    assert np.linalg.norm(a2[0]) > 0.0


def test_pn1_negative_central_and_bad_central():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.02, 0.0], [0.0, 0.0, 0.0]])
    gm = np.array([0.0, GM_SUN])
    a_last = make_pn1(central=1)(pos, vel, gm, 0.0)
    a_neg = make_pn1(central=-1)(pos, vel, gm, 0.0)
    np.testing.assert_array_equal(a_last, a_neg)
    with pytest.raises(TypeError):
        make_pn1(central="Sun")
    with pytest.raises(IndexError):
        make_pn1(central=5)(pos, vel, gm, 0.0)


def test_pn1_vanishes_for_huge_c(rng):
    pos, vel, gm = random_system(rng, 4)
    gm[0] = GM_SUN
    a_big = make_pn1(c=1e12)(pos, vel, gm, 0.0)
    a_ref = make_pn1(c=C_AU_DAY)(pos, vel, gm, 0.0)
    assert np.abs(a_big).max() < 1e-19 * np.abs(a_ref).max()  # (C_AU_DAY / 1e12)^2 ~ 3e-20
    assert np.abs(a_big).max() < 1e-30


def test_pn1_scales_as_inverse_c_squared(rng):
    pos, vel, gm = random_system(rng, 4)
    gm[0] = GM_SUN
    a1 = make_pn1(c=100.0)(pos, vel, gm, 0.0)
    a2 = make_pn1(c=10.0)(pos, vel, gm, 0.0)
    np.testing.assert_allclose(a2, 100.0 * a1, rtol=1e-13)


def test_pn1_circular_orbit_is_radial_outward():
    """On a circular orbit r.v = 0 and v^2 = gm/r, so a_1PN = 3 gm^2/(c^2 r^3) r_hat (GR)."""
    r0, c = 0.4, 20.0
    v0 = math.sqrt(GM_SUN / r0)
    pos = np.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, v0, 0.0]])
    gm = np.array([GM_SUN, 0.0])
    a = make_pn1(c=c)(pos, vel, gm, 0.0)
    expected = 3.0 * GM_SUN**2 / (c**2 * r0**3)
    np.testing.assert_allclose(a[1], [expected, 0.0, 0.0], rtol=1e-13, atol=1e-30)


def test_pn1_is_pure_and_ignores_t(rng):
    pos, vel, gm = random_system(rng, 4)
    pos0, vel0, gm0 = pos.copy(), vel.copy(), gm.copy()
    fn = make_pn1()
    a1 = fn(pos, vel, gm, 0.0)
    a2 = fn(pos, vel, gm, 9999.0)
    a3 = fn(pos, vel, gm)
    np.testing.assert_array_equal(pos, pos0)
    np.testing.assert_array_equal(vel, vel0)
    np.testing.assert_array_equal(gm, gm0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(a1, a3)
    assert fn.params["beta"] == 1.0 and fn.params["c"] == C_AU_DAY


def test_mercury_pn1_to_newton_ratio_at_j2000(j2000):
    s = j2000
    i_sun, i_merc = s.index("Sun"), s.index("Mercury")
    a_n = accel_newton(s.pos, s.vel, s.gm, s.t)
    a_pn = make_pn1(central=i_sun)(s.pos, s.vel, s.gm, s.t)
    ratio = np.linalg.norm(a_pn[i_merc]) / np.linalg.norm(a_n[i_merc])
    assert 1e-8 < ratio < 1e-6
    assert np.all(a_pn[i_sun] == 0.0)
    # every planet gets a correction and the inner ones get the largest
    mags = np.linalg.norm(a_pn, axis=1)
    assert np.all(mags[1:] > 0.0)
    assert mags[i_merc] == mags.max()


# --------------------------------------------------------------------------- combine


def test_combine_two_functions_equals_sum(rng):
    pos, vel, gm = random_system(rng, 5)
    gm[0] = GM_SUN
    pn1 = make_pn1(c=30.0)
    both = combine(accel_newton, pn1)
    a = both(pos, vel, gm, 0.0)
    expected = accel_newton(pos, vel, gm, 0.0) + pn1(pos, vel, gm, 0.0)
    np.testing.assert_array_equal(a, expected)
    assert a.shape == (5, 3)
    assert "newton" in both.params["law"] and "pn1" in both.params["law"]


def test_combine_single_and_three_and_empty(rng):
    pos, vel, gm = random_system(rng, 4)
    gm[0] = GM_SUN
    one = combine(accel_newton)
    np.testing.assert_array_equal(one(pos, vel, gm, 0.0), accel_newton(pos, vel, gm, 0.0))
    three = combine(accel_newton, accel_newton, accel_newton)
    np.testing.assert_allclose(
        three(pos, vel, gm, 0.0), 3.0 * accel_newton(pos, vel, gm, 0.0), rtol=1e-15
    )
    with pytest.raises(ValueError):
        combine()


def test_combine_does_not_alias_component_output(rng):
    pos, vel, gm = random_system(rng, 3)
    cached = np.ones((3, 3))

    def returns_cached(pos, vel, gm, t):
        return cached

    fn = combine(returns_cached, returns_cached)
    out = fn(pos, vel, gm, 0.0)
    np.testing.assert_array_equal(cached, np.ones((3, 3)))  # component output untouched
    np.testing.assert_array_equal(out, 2.0 * np.ones((3, 3)))


def test_make_newton_pn1_equals_combine(rng):
    pos, vel, gm = random_system(rng, 6)
    gm[0] = GM_SUN
    kw = dict(c=25.0, central=0, beta=1.1, gamma=0.9)
    a = make_newton_pn1(**kw)(pos, vel, gm, 0.0)
    expected = combine(accel_newton, make_pn1(**kw))(pos, vel, gm, 0.0)
    np.testing.assert_array_equal(a, expected)


def test_accelfn_signature_accepts_and_ignores_t(rng):
    pos, vel, gm = random_system(rng, 4)
    gm[0] = GM_SUN
    for fn in (
        accel_newton,
        accel_newton_numpy,
        make_pn1(),
        make_newton_pn1(),
        combine(accel_newton, make_pn1()),
    ):
        a0 = fn(pos, vel, gm, 0.0)
        a1 = fn(pos, vel, gm, 36525.0)
        a2 = fn(pos, vel, gm, t=-1.5)
        np.testing.assert_array_equal(a0, a1)
        np.testing.assert_array_equal(a0, a2)
        assert a0.shape == (4, 3) and a0.dtype == np.float64


def test_combine_accepts_partial_and_callable_instances(rng):
    """Any callable with the AccelFn signature works, not only objects with ``__name__``."""
    pos, vel, gm = random_system(rng, 5)
    gm[0] = GM_SUN
    a_ref = accel_newton(pos, vel, gm, 0.0)

    def scaled(pos, vel, gm, t, factor):
        return factor * accel_newton(pos, vel, gm, t)

    class Newton:
        def __call__(self, pos, vel, gm, t=0.0):
            return accel_newton(pos, vel, gm, t)

    partial_plain = functools.partial(accel_newton)
    partial_bound = functools.partial(scaled, factor=2.0)
    instance = Newton()
    lam = lambda pos, vel, gm, t: accel_newton(pos, vel, gm, t)  # noqa: E731

    for fn, factor in ((partial_plain, 1.0), (partial_bound, 2.0), (instance, 1.0), (lam, 1.0)):
        single = combine(fn)
        np.testing.assert_allclose(single(pos, vel, gm, 0.0), factor * a_ref, rtol=1e-15)
        assert isinstance(single.__name__, str) and isinstance(single.params["law"], str)
        assert "0x" not in single.params["law"]  # no memory addresses in meta

    both = combine(partial_bound, instance, make_pn1(c=30.0))
    expected = 2.0 * a_ref + a_ref + make_pn1(c=30.0)(pos, vel, gm, 0.0)
    np.testing.assert_allclose(both(pos, vel, gm, 0.0), expected, rtol=1e-15)
    assert both.params["law"] == "scaled+Newton+pn1"
    assert both.__name__ == "combine(scaled, Newton, accel_pn1)"
    assert both.params["terms"][2]["c"] == 30.0
    assert len(both.params["terms"]) == 3


def test_combine_params_do_not_alias_component_params(rng):
    pn1 = make_pn1(c=30.0)
    both = combine(accel_newton, pn1)
    both.params["terms"][1]["c"] = -1.0
    assert pn1.params["c"] == 30.0


# --------------------------------------------------------------------------- shape validation


def _bad_shape_cases(n: int = 6) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(1)
    pos, vel, gm = random_system(rng, n)
    gm[0] = GM_SUN
    return {
        "gm_short": (pos, vel, gm[:-1]),
        "gm_length_1": (pos, vel, gm[:1]),
        "gm_long": (pos, vel, np.append(gm, 1e-9)),
        "gm_2d": (pos, vel, gm[:, None]),
        "pos_2cols": (pos[:, :2], vel[:, :2], gm),
        "pos_1d": (pos[0], vel[0], gm[:1]),
        "pos_3d": (pos[:, :, None], vel[:, :, None], gm),
        "vel_mismatch": (pos, vel[:-1], gm),
        "vel_2cols": (pos, vel[:, :2], gm),
    }


def _all_laws():
    laws = [accel_newton, accel_newton_numpy, make_pn1(), make_newton_pn1()]
    laws.append(combine(accel_newton, make_pn1()))
    if HAVE_NUMBA:
        laws.append(accel_newton_numba)
    return laws


@pytest.mark.parametrize("case", sorted(_bad_shape_cases()))
@pytest.mark.parametrize("fn", _all_laws(), ids=lambda f: getattr(f, "__name__", repr(f)))
def test_every_law_rejects_bad_shapes(fn, case):
    pos, vel, gm = _bad_shape_cases()[case]
    with pytest.raises(ValueError):
        fn(pos, vel, gm, 0.0)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not importable")
@pytest.mark.parametrize("m", [1, 3, 50, 199])
def test_numba_kernel_never_reads_past_short_gm(m):
    """Regression: a short gm view into a longer buffer used to be read out of bounds."""
    rng = np.random.default_rng(2)
    n = 200
    pos, vel, gm_full = random_system(rng, n)
    with pytest.raises(ValueError, match="gm must have shape"):
        accel_newton_numba(pos, vel, gm_full[:m], 0.0)
    with pytest.raises(ValueError, match="gm must have shape"):
        accel_newton_numba(pos, vel, np.ascontiguousarray(gm_full[:m]), 0.0)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not importable")
@pytest.mark.parametrize("case", sorted(_bad_shape_cases()))
def test_numba_and_numpy_agree_on_error(case):
    """The two 'interchangeable' implementations raise the same error for the same input."""
    pos, vel, gm = _bad_shape_cases()[case]
    with pytest.raises(ValueError) as e_nb:
        accel_newton_numba(pos, vel, gm, 0.0)
    with pytest.raises(ValueError) as e_np:
        accel_newton_numpy(pos, vel, gm, 0.0)
    assert str(e_nb.value) == str(e_np.value)


def test_pn1_rejects_wrong_gm_even_when_only_central_is_read():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vel = np.zeros((3, 3))
    fn = make_pn1(central=0)
    with pytest.raises(ValueError, match="gm must have shape"):
        fn(pos, vel, np.array([GM_SUN, 0.0]), 0.0)
    with pytest.raises(ValueError, match="pos must have shape"):
        fn(pos[:, :2], vel[:, :2], np.array([GM_SUN, 0.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="vel shape"):
        fn(pos, vel[:2], np.array([GM_SUN, 0.0, 0.0]), 0.0)
    assert fn(pos, vel, np.array([GM_SUN, 0.0, 0.0]), 0.0).shape == (3, 3)


def test_check_inputs_coerces_without_copying_conforming_arrays(rng):
    pos, vel, gm = random_system(rng, 4)
    p, v, g = check_inputs(pos, vel, gm)
    assert p is pos and v is vel and g is gm  # already float64 C-contiguous: no copy
    # int / non-contiguous inputs are coerced to fresh float64 C-contiguous arrays
    pos_i = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    big = np.zeros((4, 6))
    big[:, ::2] = vel
    p, v, g = check_inputs(pos_i, big[:, ::2], gm)
    assert p.dtype == np.float64 and p.flags.c_contiguous and v.flags.c_contiguous
    np.testing.assert_array_equal(p, pos_i)
    np.testing.assert_array_equal(v, vel)
    assert not np.shares_memory(v, big)
    # scalar / empty systems
    p, v, g = check_inputs(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    assert p.shape == (0, 3) and g.shape == (0,)
    with pytest.raises(ValueError):
        check_inputs([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0])


@pytest.mark.parametrize("fn", [accel_newton, accel_newton_numpy, make_pn1()])
def test_single_body_and_empty_system(fn):
    assert fn(np.zeros((1, 3)), np.zeros((1, 3)), np.array([GM_SUN]), 0.0).shape == (1, 3)
    assert np.all(fn(np.zeros((1, 3)), np.zeros((1, 3)), np.array([GM_SUN]), 0.0) == 0.0)
