"""Tests for gravsim.bodies: SystemState and Trajectory containers.

Analytic fixtures only (hand-built states, a circular two-body orbit); no integrator, no
network.  The tests pin down the contract of CONTRIBUTING.md: float64 C-contiguous arrays, shape
validation, immutability of inputs, name lookup, frame changes and the Trajectory accessors.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import GM, GM_SUN

# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def three_body() -> SystemState:
    """Sun at rest at the origin, two planets on the x and y axes with circular velocities."""
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.7, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.027, 0.0], [-0.020, 0.0, 0.001]])
    gm = np.array([GM_SUN, GM["Mercury"], GM["Venus"]])
    return SystemState(t=12.5, pos=pos, vel=vel, gm=gm, names=("Sun", "Mercury", "Venus"))


def circular_two_body(n_samples: int = 50, periods: float = 1.0) -> Trajectory:
    """Sun + Mercury on a circular 0.387 AU orbit about their common barycenter (analytic)."""
    a = 0.387
    gm_s, gm_m = GM_SUN, GM["Mercury"]
    mu = gm_s + gm_m
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    t = np.linspace(0.0, periods * period, n_samples)
    w = 2.0 * math.pi / period
    r = a * np.column_stack([np.cos(w * t), np.sin(w * t), np.zeros_like(t)])
    v = a * w * np.column_stack([-np.sin(w * t), np.cos(w * t), np.zeros_like(t)])
    f_s, f_m = -gm_m / mu, gm_s / mu
    pos = np.stack([f_s * r, f_m * r], axis=1)
    vel = np.stack([f_s * v, f_m * v], axis=1)
    return Trajectory(t, pos, vel, np.array([gm_s, gm_m]), ("Sun", "Mercury"), {"period": period})


# =========================================================================== SystemState
class TestSystemStateConstruction:
    def test_arrays_are_float64_c_contiguous_copies(self):
        pos = [[0, 0, 0], [1, 0, 0]]  # ints, nested lists
        vel = np.asfortranarray(np.zeros((2, 3), dtype=np.float32))
        gm = (1.0, 0.0)
        s = SystemState(t=3, pos=pos, vel=vel, gm=gm)
        for arr in (s.pos, s.vel, s.gm):
            assert arr.dtype == np.float64 and arr.flags.c_contiguous
        assert s.pos.shape == (2, 3) and s.vel.shape == (2, 3) and s.gm.shape == (2,)
        assert isinstance(s.t, float) and s.t == 3.0
        assert s.n == 2

    def test_default_names_are_generated(self):
        s = SystemState(0.0, np.zeros((3, 3)), np.zeros((3, 3)), np.ones(3))
        assert s.names == ("body0", "body1", "body2")

    def test_names_are_stored_as_tuple(self):
        s = SystemState(0.0, np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2), names=["a", "b"])
        assert s.names == ("a", "b") and isinstance(s.names, tuple)

    def test_is_frozen(self, three_body):
        with pytest.raises(dataclasses.FrozenInstanceError):
            three_body.t = 0.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "pos, vel, gm, names",
        [
            (np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2), None),  # not (N, 3)
            (np.zeros(6), np.zeros(6), np.ones(2), None),  # 1-D
            (np.zeros((2, 3)), np.zeros((3, 3)), np.ones(2), None),  # vel mismatch
            (np.zeros((2, 3)), np.zeros((2, 3)), np.ones(3), None),  # gm length
            (np.zeros((2, 3)), np.zeros((2, 3)), np.ones((2, 1)), None),  # gm 2-D
            (np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2), ("only_one",)),  # names length
        ],
    )
    def test_shape_validation(self, pos, vel, gm, names):
        with pytest.raises(ValueError):
            SystemState(0.0, pos, vel, gm, names=names or ())


class TestSystemStateQueries:
    def test_index(self, three_body):
        assert three_body.index("Sun") == 0
        assert three_body.index("Venus") == 2
        with pytest.raises(KeyError, match="Pluto"):
            three_body.index("Pluto")

    def test_with_replaces_fields_and_revalidates(self, three_body):
        s2 = three_body.with_(t=100.0, pos=three_body.pos + 1.0)
        assert s2.t == 100.0
        np.testing.assert_array_equal(s2.pos, three_body.pos + 1.0)
        np.testing.assert_array_equal(s2.vel, three_body.vel)  # untouched fields shared
        assert s2.names == three_body.names
        assert three_body.t == 12.5  # the original is unchanged
        with pytest.raises(ValueError):
            three_body.with_(pos=np.zeros((5, 3)))

    def test_subset_selects_and_reorders(self, three_body):
        sub = three_body.subset(["Venus", "Sun"])
        assert sub.names == ("Venus", "Sun") and sub.n == 2
        np.testing.assert_array_equal(sub.pos[0], three_body.pos[2])
        np.testing.assert_array_equal(sub.pos[1], three_body.pos[0])
        np.testing.assert_array_equal(sub.gm, three_body.gm[[2, 0]])
        assert sub.t == three_body.t
        with pytest.raises(KeyError):
            three_body.subset(("Sun", "Mars"))

    def test_subset_is_a_copy(self, three_body):
        sub = three_body.subset(("Sun", "Mercury"))
        sub.pos[0, 0] = 99.0
        assert three_body.pos[0, 0] == 0.0


class TestSystemStateFrames:
    def test_barycenter_weights_by_gm(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        gm = np.array([3.0, 1.0])
        r0, v0 = SystemState(0.0, pos, vel, gm).barycenter()
        np.testing.assert_allclose(r0, [0.25, 0.0, 0.0])
        np.testing.assert_allclose(v0, [0.0, 0.5, 0.0])

    def test_test_particles_do_not_move_the_barycenter(self):
        pos = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        vel = np.zeros((2, 3))
        r0, _ = SystemState(0.0, pos, vel, np.array([1.0, 0.0])).barycenter()
        np.testing.assert_array_equal(r0, 0.0)

    def test_to_barycentric_removes_the_center_of_mass(self, three_body):
        shifted = three_body.with_(pos=three_body.pos + [1.0, -2.0, 0.5], vel=three_body.vel + 1e-3)
        b = shifted.to_barycentric()
        r0, v0 = b.barycenter()
        assert np.abs(r0).max() < 1e-15 and np.abs(v0).max() < 1e-18
        # relative geometry is preserved
        np.testing.assert_allclose(b.pos[1] - b.pos[0], shifted.pos[1] - shifted.pos[0])
        np.testing.assert_allclose(b.vel[2] - b.vel[0], shifted.vel[2] - shifted.vel[0])
        assert b.t == shifted.t and b.names == shifted.names
        # the input is untouched
        assert shifted.pos[0, 0] == 1.0

    def test_relative_to(self, three_body):
        r, v = three_body.relative_to("Mercury")
        assert r.shape == (3, 3) and v.shape == (3, 3)
        np.testing.assert_array_equal(r[1], 0.0)
        np.testing.assert_array_equal(v[1], 0.0)
        np.testing.assert_allclose(r[0], -three_body.pos[1])
        np.testing.assert_allclose(r[2], three_body.pos[2] - three_body.pos[1])
        with pytest.raises(KeyError):
            three_body.relative_to("Mars")
        np.testing.assert_array_equal(three_body.pos[1], [0.4, 0.0, 0.0])  # untouched


# =========================================================================== Trajectory
class TestTrajectoryConstruction:
    def test_arrays_and_shapes(self):
        tr = circular_two_body(7)
        assert tr.n_steps == 7 and tr.n == 2
        assert tr.t.shape == (7,) and tr.pos.shape == (7, 2, 3) and tr.vel.shape == (7, 2, 3)
        for arr in (tr.t, tr.pos, tr.vel, tr.gm):
            assert arr.dtype == np.float64 and arr.flags.c_contiguous
        assert tr.names == ("Sun", "Mercury") and tr.meta["period"] > 0.0

    def test_meta_defaults_to_empty_dict(self):
        zeros = np.zeros((1, 2, 3))
        tr = Trajectory(np.zeros(1), zeros, zeros, np.ones(2), ("a", "b"))
        assert tr.meta == {}

    @pytest.mark.parametrize(
        "t, pos, vel, gm",
        [
            (np.zeros(3), np.zeros((3, 2, 2)), np.zeros((3, 2, 2)), np.ones(2)),  # last axis
            (np.zeros(3), np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2)),  # 2-D pos
            (np.zeros(3), np.zeros((3, 2, 3)), np.zeros((3, 1, 3)), np.ones(2)),  # vel differs
            (np.zeros(4), np.zeros((3, 2, 3)), np.zeros((3, 2, 3)), np.ones(2)),  # t length
            (np.zeros((3, 1)), np.zeros((3, 2, 3)), np.zeros((3, 2, 3)), np.ones(2)),  # t 2-D
            (np.zeros(3), np.zeros((3, 2, 3)), np.zeros((3, 2, 3)), np.ones(3)),  # gm length
        ],
    )
    def test_shape_validation(self, t, pos, vel, gm):
        with pytest.raises(ValueError):
            Trajectory(t, pos, vel, gm, ("a", "b"))


class TestTrajectoryAccessors:
    def test_index(self):
        tr = circular_two_body()
        assert tr.index("Mercury") == 1
        with pytest.raises(KeyError):
            tr.index("Venus")

    def test_state_at_and_final_state(self):
        tr = circular_two_body(9)
        s3 = tr.state_at(3)
        assert isinstance(s3, SystemState)
        assert s3.t == tr.t[3] and s3.names == tr.names
        np.testing.assert_array_equal(s3.pos, tr.pos[3])
        np.testing.assert_array_equal(s3.vel, tr.vel[3])
        np.testing.assert_array_equal(s3.gm, tr.gm)
        last = tr.final_state
        assert last.t == tr.t[-1]
        np.testing.assert_array_equal(last.pos, tr.pos[-1])

    def test_body_returns_time_series_of_one_body(self):
        tr = circular_two_body(9)
        r, v = tr.body("Mercury")
        assert r.shape == (9, 3) and v.shape == (9, 3)
        np.testing.assert_array_equal(r, tr.pos[:, 1, :])
        np.testing.assert_array_equal(v, tr.vel[:, 1, :])

    def test_relative_is_the_two_body_separation(self):
        tr = circular_two_body(9)
        r, v = tr.relative("Mercury", "Sun")
        np.testing.assert_allclose(np.linalg.norm(r, axis=1), 0.387, rtol=1e-12)
        # circular orbit: r . v = 0 and |v| = sqrt(mu / a)
        np.testing.assert_allclose(np.sum(r * v, axis=1), 0.0, atol=1e-12)
        mu = tr.gm.sum()
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), math.sqrt(mu / 0.387), rtol=1e-12)
        r_sun, _ = tr.relative("Sun", "Mercury")
        np.testing.assert_allclose(r_sun, -r)
        with pytest.raises(KeyError):
            tr.relative("Mercury", "Venus")

    def test_from_states_round_trip(self):
        tr = circular_two_body(6)
        states = [tr.state_at(i) for i in range(tr.n_steps)]
        back = Trajectory.from_states(states, meta={"integrator": "none"})
        np.testing.assert_array_equal(back.t, tr.t)
        np.testing.assert_array_equal(back.pos, tr.pos)
        np.testing.assert_array_equal(back.vel, tr.vel)
        np.testing.assert_array_equal(back.gm, tr.gm)
        assert back.names == tr.names and back.meta == {"integrator": "none"}

    def test_from_states_copies_meta_and_rejects_empty(self):
        meta = {"k": 1}
        tr = Trajectory.from_states([circular_two_body(2).state_at(0)], meta=meta)
        meta["k"] = 2
        assert tr.meta == {"k": 1}
        assert tr.n_steps == 1
        with pytest.raises(ValueError):
            Trajectory.from_states([])


class TestTrajectoryImmutability:
    def test_accessors_do_not_modify_the_trajectory(self):
        tr = circular_two_body(5)
        pos0 = tr.pos.copy()
        tr.body("Sun")
        tr.relative("Mercury")
        tr.state_at(2)
        _ = tr.final_state
        np.testing.assert_array_equal(tr.pos, pos0)

    def test_state_at_is_a_view_not_a_copy(self):
        """state_at shares memory with the trajectory (documented: it is a read-only snapshot)."""
        tr = circular_two_body(5)
        s = tr.state_at(1)
        assert np.shares_memory(s.pos, tr.pos)
