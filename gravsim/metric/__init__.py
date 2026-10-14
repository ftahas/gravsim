"""Exact geodesics of the Schwarzschild metric.

* :mod:`gravsim.metric.schwarzschild` -- timelike geodesics (effective potential, ISCO, Binet
  orbit, proper-time integration, perihelion advance).
* :mod:`gravsim.metric.null` -- photon geodesics, light deflection and Shapiro delay.

See CONTRIBUTING.md for the package contract.
"""

from gravsim.metric import null, schwarzschild

__all__ = ["null", "schwarzschild"]
