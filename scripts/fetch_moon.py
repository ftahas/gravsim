"""Fetch the Earth+Moon variant of the J2000 snapshot and a 10-year daily reference (network)."""

from gravsim.constants import DEFAULT_BODIES_MOON
from gravsim.data import horizons as H

state = H.fetch_state(names=DEFAULT_BODIES_MOON)
H.save_snapshot(state, H.J2000_MOON_PATH)
print("snapshot:", H.J2000_MOON_PATH, state.names)
traj = H.fetch_series(stop="2010-01-01 12:00", step="1d", names=DEFAULT_BODIES_MOON)
H.save_reference_series(traj, H.MOON_REF_PATH)
print("series:", H.MOON_REF_PATH, traj.pos.shape, traj.t[0], traj.t[-1])
