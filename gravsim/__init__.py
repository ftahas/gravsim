"""gravsim: solar-system dynamics with Newtonian gravity and General Relativity.

Pedagogical package. Units everywhere: AU, day, GM in AU^3/day^2 (see `gravsim.constants`).
Frame: barycentric, J2000 ecliptic (x toward the J2000 equinox, z toward the ecliptic pole).
"""

from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import C_AU_DAY, GM_SUN

__all__ = ["SystemState", "Trajectory", "C_AU_DAY", "GM_SUN"]
__version__ = "0.1.0"
