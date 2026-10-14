"""Interactive plotly figures for the app (plotly is optional: ``pip install plotly``).

Same physics and conventions as :mod:`gravsim.viz.plots`:

* Orbits: barycentric J2000-ecliptic positions in AU drawn relative to the central body,
  ``r_i(t) = pos_i(t) - pos_c(t)`` (``relative=False`` keeps the barycentric paths); the
  central body is marked at its last sample.  The three scene axes share one range (a cube
  enclosing all data) with ``aspectmode="cube"`` so distances are not distorted.
* Precession: ``Delta varpi(t) = varpi(t) - varpi(t_0)`` after unwrapping, converted to
  arcseconds (``constants.ARCSEC_PER_RAD``) against Julian years (365.25 d); a secular rate in
  rad/day is drawn as the best-fitting line of that slope and quoted per Julian century.

plotly is imported inside the functions, so importing this module never requires it; calling
a function without plotly raises ``ImportError`` that says how to install it.  Colours and the
numeric conversions are shared with the matplotlib module so both front ends agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gravsim.bodies import Trajectory
from gravsim.viz.plots import (
    INK,
    PALETTE,
    body_colors,
    body_paths,
    cube_limits,
    precession_series,
    resolve_bodies,
)

__all__ = ["orbits_3d_figure", "precession_figure"]

_INSTALL_MSG = (
    "gravsim.viz.interactive needs plotly, which is an optional dependency: "
    "run 'pip install plotly' (or 'uv pip install plotly') in the project environment."
)


def _graph_objects() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc
    return go


def orbits_3d_figure(
    traj: Trajectory,
    bodies: Sequence[str | int] | str | int | None = None,
    central: str | int = "Sun",
    relative: bool = True,
) -> Any:
    """Interactive 3-D orbit plot: one line trace per body plus a marker for the central body.

    Parameters
    ----------
    traj : Trajectory
    bodies : sequence of str or int, optional
        Bodies to draw; default every body except ``central``.
    central : str or int, default "Sun"
        Reference body, marked at its last sample (the origin when ``relative``).
    relative : bool, default True
        Subtract the central body's position.  When False the central body's own barycentric
        path is drawn as a thin line as well.

    Returns
    -------
    plotly.graph_objects.Figure
        Traces: one ``Scatter3d`` line per body (in ``bodies`` order), the central path when
        not ``relative``, and finally the central-body marker.  Scene axes share one range.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    go = _graph_objects()
    c_name, names = resolve_bodies(traj, bodies, central)
    paths = body_paths(traj, names, c_name, relative)
    hover = (
        "%{fullData.name}<br>x = %{x:.5f} AU<br>y = %{y:.5f} AU<br>z = %{z:.5f} AU"
        "<br>t = %{customdata:.1f} d<extra></extra>"
    )

    fig = go.Figure()
    for name, path, color in zip(names, paths, body_colors(names), strict=True):
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                customdata=traj.t,
                hovertemplate=hover,
            )
        )
    if relative:
        c_path = np.zeros((1, 3))
        frame = f"relative to {c_name}"
    else:
        c_path = traj.body(c_name)[0]
        frame = "barycentric"
        fig.add_trace(
            go.Scatter3d(
                x=c_path[:, 0],
                y=c_path[:, 1],
                z=c_path[:, 2],
                mode="lines",
                name=f"{c_name} (path)",
                line={"color": INK, "width": 1},
                customdata=traj.t,
                hovertemplate=hover,
            )
        )
    mark = c_path[-1]
    fig.add_trace(
        go.Scatter3d(
            x=[mark[0]],
            y=[mark[1]],
            z=[mark[2]],
            mode="markers",
            name=c_name,
            marker={"size": 6, "color": INK, "symbol": "diamond"},
            hovertemplate="%{fullData.name}<extra></extra>",
        )
    )

    xl, yl, zl = cube_limits(np.concatenate([*paths, c_path], axis=0))
    fig.update_layout(
        template="plotly_white",
        title=f"Orbits ({frame})",
        scene={
            "xaxis": {"title": "x [AU]", "range": list(xl)},
            "yaxis": {"title": "y [AU]", "range": list(yl)},
            "zaxis": {"title": "z [AU]", "range": list(zl)},
            "aspectmode": "cube",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0.0},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig


def precession_figure(
    t: np.ndarray,
    varpi_rad: np.ndarray,
    rate: float | None = None,
    units: str = "arcsec",
    t_units: str = "years",
) -> Any:
    """Interactive perihelion-longitude drift ``varpi(t) - varpi(t_0)`` with an optional fit.

    Parameters
    ----------
    t : array_like, shape (T,)
        Time in days.
    varpi_rad : array_like, shape (T,)
        Longitude of perihelion in radians.
    rate : float, optional
        Secular rate in rad/day; drawn as a dashed line labelled with the rate per century.
    units : {"arcsec", "deg", "rad"}, default "arcsec"
    t_units : {"days", "years", "centuries"}, default "years"

    Returns
    -------
    plotly.graph_objects.Figure
        One ``Scatter`` trace for the series and, when ``rate`` is given, one for the fit.

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    go = _graph_objects()
    s = precession_series(t, varpi_rad, rate, units=units, t_units=t_units)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.t,
            y=s.dvarpi,
            mode="lines",
            name="Δϖ",
            line={"color": PALETTE[0], "width": 2},
            hovertemplate="%{x:.3f}<br>Δϖ = %{y:.4g}<extra></extra>",
        )
    )
    if s.fit is not None:
        fig.add_trace(
            go.Scatter(
                x=s.t,
                y=s.fit,
                mode="lines",
                name=s.rate_label,
                line={"color": INK, "width": 1, "dash": "dash"},
                hovertemplate="%{x:.3f}<br>fit = %{y:.4g}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=s.t_label,
        yaxis_title=s.y_label,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
    )
    return fig
