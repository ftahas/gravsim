"""State containers shared by every force law, integrator and analysis routine.

Conventions
-----------
* `pos` and `vel` are float64 arrays of shape (N, 3) in AU and AU/day.
* `gm` is a float64 array of shape (N,) in AU^3/day^2 (G*M per body; a test particle has gm=0).
* `t` is time in days from the epoch (J2000.0 TDB for the default solar system).
* Body index 0 is the Sun / central body by convention for the default system; code must
  not rely on that silently -- use `SystemState.index("Sun")` or an explicit `central` argument.
* Containers are plain dataclasses; integrators return *new* arrays and never mutate inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

Vec3Array = np.ndarray  # (N, 3)


@dataclass(frozen=True)
class SystemState:
    """Positions, velocities and GMs of N bodies at one instant."""

    t: float
    pos: np.ndarray  # (N, 3) AU
    vel: np.ndarray  # (N, 3) AU/day
    gm: np.ndarray  # (N,) AU^3/day^2
    names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        gm = np.ascontiguousarray(self.gm, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
        if vel.shape != pos.shape:
            raise ValueError(f"vel shape {vel.shape} != pos shape {pos.shape}")
        if gm.shape != (pos.shape[0],):
            raise ValueError(f"gm must have shape (N,), got {gm.shape}")
        names = tuple(self.names) if self.names else tuple(f"body{i}" for i in range(len(gm)))
        if len(names) != len(gm):
            raise ValueError("names length must equal number of bodies")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "vel", vel)
        object.__setattr__(self, "gm", gm)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "t", float(self.t))

    # ---- basic queries -----------------------------------------------------------
    @property
    def n(self) -> int:
        return self.pos.shape[0]

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"no body named {name!r}; have {self.names}") from exc

    def with_(self, **changes) -> SystemState:
        """Return a copy with some fields replaced (e.g. `state.with_(t=..., pos=...)`)."""
        return replace(self, **changes)

    def subset(self, names: list[str] | tuple[str, ...]) -> SystemState:
        idx = [self.index(n) for n in names]
        return SystemState(self.t, self.pos[idx], self.vel[idx], self.gm[idx], tuple(names))

    # ---- frames ------------------------------------------------------------------
    def barycenter(self) -> tuple[np.ndarray, np.ndarray]:
        """Center of mass position and velocity (weights = gm)."""
        w = self.gm / self.gm.sum()
        return w @ self.pos, w @ self.vel

    def to_barycentric(self) -> SystemState:
        r0, v0 = self.barycenter()
        return self.with_(pos=self.pos - r0, vel=self.vel - v0)

    def relative_to(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities relative to body `name` (shape (N, 3))."""
        i = self.index(name)
        return self.pos - self.pos[i], self.vel - self.vel[i]


@dataclass(frozen=True)
class Trajectory:
    """Time series of a SystemState: arrays of shape (T,), (T, N, 3), (T, N, 3)."""

    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    gm: np.ndarray
    names: tuple[str, ...]
    meta: dict = field(default_factory=dict)  # integrator, dt, force law, c, ...

    def __post_init__(self) -> None:
        t = np.ascontiguousarray(self.t, dtype=np.float64)
        pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        gm = np.ascontiguousarray(self.gm, dtype=np.float64)
        if pos.ndim != 3 or pos.shape[2] != 3 or vel.shape != pos.shape:
            raise ValueError(f"pos/vel must have shape (T, N, 3); got {pos.shape}, {vel.shape}")
        if t.shape != (pos.shape[0],):
            raise ValueError(f"t must have shape (T,) = ({pos.shape[0]},), got {t.shape}")
        if gm.shape != (pos.shape[1],):
            raise ValueError(f"gm must have shape (N,), got {gm.shape}")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "vel", vel)
        object.__setattr__(self, "gm", gm)
        object.__setattr__(self, "names", tuple(self.names))

    @property
    def n_steps(self) -> int:
        return self.t.shape[0]

    @property
    def n(self) -> int:
        return self.pos.shape[1]

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"no body named {name!r}; have {self.names}") from exc

    def state_at(self, i: int) -> SystemState:
        return SystemState(self.t[i], self.pos[i], self.vel[i], self.gm, self.names)

    @property
    def final_state(self) -> SystemState:
        return self.state_at(-1)

    def body(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(pos, vel) of one body, each of shape (T, 3)."""
        i = self.index(name)
        return self.pos[:, i, :], self.vel[:, i, :]

    def relative(self, name: str, central: str = "Sun") -> tuple[np.ndarray, np.ndarray]:
        """(pos, vel) of `name` relative to `central`, each of shape (T, 3)."""
        i, j = self.index(name), self.index(central)
        return self.pos[:, i, :] - self.pos[:, j, :], self.vel[:, i, :] - self.vel[:, j, :]

    @classmethod
    def from_states(cls, states: list[SystemState], meta: dict | None = None) -> Trajectory:
        if not states:
            raise ValueError("need at least one state")
        s0 = states[0]
        return cls(
            t=np.array([s.t for s in states]),
            pos=np.stack([s.pos for s in states]),
            vel=np.stack([s.vel for s in states]),
            gm=s0.gm,
            names=s0.names,
            meta=dict(meta or {}),
        )
