"""Fetch the J2000 snapshot and the 100-year reference series from JPL Horizons (network)."""

from gravsim.data import horizons as H

state = H.fetch_state()
H.save_snapshot(state)
print("snapshot:", H.J2000_PATH, state.names)
traj = H.fetch_series()
H.save_reference_series(traj)
print("series:", H.REF_PATH, traj.pos.shape, "t[0..-1] =", traj.t[0], traj.t[-1])
