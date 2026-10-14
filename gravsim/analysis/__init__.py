"""Measurement toolkit: Kepler solution, osculating elements, perihelion precession, invariants.

Submodules
----------
kepler      analytic two-body solution (Kepler's equation, elements <-> state, propagation)
elements    angular momentum, node and eccentricity (LRL) vectors, longitude of perihelion
precession  varpi(t) series, perihelion passages, secular-rate fits, GR/PPN analytic rate
conserved   energy and angular momentum (times G) along a trajectory
compare     run-versus-run separations and differential precession rates

Every routine is vectorised over a leading time axis and never mutates its inputs.
Units: AU, day, GM in AU^3/day^2, angles in radians (see CONTRIBUTING.md).
"""

from gravsim.analysis.compare import perihelion_rate_difference, separation
from gravsim.analysis.conserved import angular_momentum, energy, relative_energy_error
from gravsim.analysis.elements import (
    lrl_vector,
    node_vector,
    perihelion_longitude,
    specific_h,
    wrap_angle,
)
from gravsim.analysis.kepler import (
    elements_to_state,
    mean_motion,
    propagate,
    solve_kepler,
    state_to_elements,
)
from gravsim.analysis.precession import (
    SecularFit,
    fit_secular_rate,
    gr_precession_arcsec_per_century,
    perihelion_longitude_series,
    perihelion_passages,
    rate_arcsec_per_century,
)

__all__ = [
    "solve_kepler",
    "elements_to_state",
    "state_to_elements",
    "propagate",
    "mean_motion",
    "wrap_angle",
    "specific_h",
    "node_vector",
    "lrl_vector",
    "perihelion_longitude",
    "SecularFit",
    "perihelion_longitude_series",
    "perihelion_passages",
    "fit_secular_rate",
    "rate_arcsec_per_century",
    "gr_precession_arcsec_per_century",
    "energy",
    "angular_momentum",
    "relative_energy_error",
    "separation",
    "perihelion_rate_difference",
]
