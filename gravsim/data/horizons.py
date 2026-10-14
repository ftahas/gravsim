"""JPL Horizons access and cached ephemerides.

Frame: Solar System Barycenter, J2000 ecliptic and mean equinox (Horizons ``refplane='ecliptic'``).
Units: AU and AU/day, epochs in JD (TDB).

Cached files (shipped with the package, no network needed):
* ``j2000.json``        -- state vectors of Sun + 8 planets at J2000.0.
* ``horizons_ref.npz``  -- the same bodies sampled every 20 days from J2000.0 for 100 Julian years,
  used as the "observed" reference (DE441-based, includes GR and asteroid perturbations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import DEFAULT_BODIES, GM, HORIZONS_IDS, J2000_JD

DATA_DIR = Path(__file__).resolve().parent
J2000_PATH = DATA_DIR / "j2000.json"
REF_PATH = DATA_DIR / "horizons_ref.npz"


# --------------------------------------------------------------------------- network
def _query(name: str, epochs, refplane: str = "ecliptic"):
    from astroquery.jplhorizons import Horizons  # lazy import: network-only dependency

    obj = Horizons(id=HORIZONS_IDS[name], location="@0", epochs=epochs)
    return obj.vectors(refplane=refplane)


def fetch_state(epoch_jd: float = J2000_JD, names: tuple[str, ...] = DEFAULT_BODIES) -> SystemState:
    """Fetch barycentric state vectors of ``names`` at one epoch (network)."""
    pos, vel = [], []
    for name in names:
        tab = _query(name, [epoch_jd])
        pos.append([float(tab["x"][0]), float(tab["y"][0]), float(tab["z"][0])])
        vel.append([float(tab["vx"][0]), float(tab["vy"][0]), float(tab["vz"][0])])
    gm = np.array([GM[n] for n in names])
    return SystemState(
        t=epoch_jd - J2000_JD, pos=np.array(pos), vel=np.array(vel), gm=gm, names=names
    )


def fetch_series(
    start: str = "2000-01-01 12:00",
    stop: str = "2100-01-01 12:00",
    step: str = "20d",
    names: tuple[str, ...] = DEFAULT_BODIES,
) -> Trajectory:
    """Fetch a time series of barycentric state vectors (network). Times are TDB."""
    epochs = {"start": start, "stop": stop, "step": step}
    t = None
    pos, vel = [], []
    for name in names:
        tab = _query(name, epochs)
        jd = np.asarray(tab["datetime_jd"], dtype=float)
        if t is None:
            t = jd - J2000_JD
        elif not np.allclose(jd - J2000_JD, t):
            raise RuntimeError(f"epoch grid mismatch for {name}")
        pos.append(np.column_stack([tab["x"], tab["y"], tab["z"]]).astype(float))
        vel.append(np.column_stack([tab["vx"], tab["vy"], tab["vz"]]).astype(float))
    gm = np.array([GM[n] for n in names])
    return Trajectory(
        t=t,
        pos=np.stack(pos, axis=1),
        vel=np.stack(vel, axis=1),
        gm=gm,
        names=names,
        meta={"source": "JPL Horizons", "frame": "SSB, J2000 ecliptic", "step": step},
    )


# --------------------------------------------------------------------------- cache
def save_snapshot(
    state: SystemState, path: Path = J2000_PATH, source: str = "JPL Horizons"
) -> None:
    doc = {
        "epoch_jd_tdb": J2000_JD + state.t,
        "t_days_since_j2000": state.t,
        "frame": "Solar System Barycenter, J2000 ecliptic and mean equinox",
        "units": {"length": "AU", "time": "day", "gm": "AU^3/day^2"},
        "source": source,
        "bodies": [
            {
                "name": n,
                "horizons_id": HORIZONS_IDS[n],
                "gm": float(state.gm[i]),
                "pos": [repr(float(x)) for x in state.pos[i]],
                "vel": [repr(float(x)) for x in state.vel[i]],
            }
            for i, n in enumerate(state.names)
        ],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")


def load_snapshot(path: Path = J2000_PATH) -> SystemState:
    doc = json.loads(Path(path).read_text())
    names = tuple(b["name"] for b in doc["bodies"])
    pos = np.array([[float(x) for x in b["pos"]] for b in doc["bodies"]])
    vel = np.array([[float(x) for x in b["vel"]] for b in doc["bodies"]])
    gm = np.array([b["gm"] for b in doc["bodies"]])
    return SystemState(t=float(doc["t_days_since_j2000"]), pos=pos, vel=vel, gm=gm, names=names)


def load_j2000() -> SystemState:
    """Sun + 8 planets at J2000.0 from the shipped cache (no network)."""
    return load_snapshot(J2000_PATH)


def save_reference_series(traj: Trajectory, path: Path = REF_PATH) -> None:
    np.savez_compressed(
        path, t=traj.t, pos=traj.pos, vel=traj.vel, gm=traj.gm, names=np.array(traj.names)
    )


def load_reference_series(path: Path = REF_PATH) -> Trajectory:
    """100-year Horizons series of Sun + 8 planets, every 20 days from J2000.0 (no network)."""
    with np.load(path) as z:
        return Trajectory(
            t=z["t"],
            pos=z["pos"],
            vel=z["vel"],
            gm=z["gm"],
            names=tuple(str(n) for n in z["names"]),
            meta={"source": "JPL Horizons", "frame": "SSB, J2000 ecliptic", "step": "20d"},
        )
