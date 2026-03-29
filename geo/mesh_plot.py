"""Plotting helpers for ``ObjectMesh``."""

from __future__ import annotations

from typing import Sequence

from .geometric import ObjectMesh
from .mesh_export import mesh_to_matplotlib_data, mesh_to_plotly_data


def plot_mesh_matplotlib(
    mesh: ObjectMesh,
    axes: Sequence[int] | None = None,
    ax=None,
    *,
    color: str = "#1f1f1f",
    linewidth: float = 1.5,
    marker_size: float = 20.0,
):
    """Return a Matplotlib figure and axes with the mesh drawn."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_mesh_matplotlib requires matplotlib to be installed"
        ) from exc

    data = mesh_to_matplotlib_data(mesh, axes=axes)
    if ax is None:
        if data["dim"] >= 3:
            figure = plt.figure()
            ax = figure.add_subplot(111, projection="3d")
        else:
            figure, ax = plt.subplots()
    else:
        figure = getattr(ax, "figure", None)

    for start, end in data["line_segments"]:
        if data["dim"] >= 3:
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=linewidth,
            )
        else:
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=linewidth,
            )

    points = data["points"]
    if data["dim"] >= 3:
        ax.scatter(
            points["x"],
            points["y"],
            points["z"],
            color=color,
            s=marker_size,
        )
    else:
        ax.scatter(
            points["x"],
            points["y"],
            color=color,
            s=marker_size,
        )

    return figure, ax


def plot_mesh_plotly(
    mesh: ObjectMesh,
    axes: Sequence[int] | None = None,
    figure=None,
    *,
    name: str = "mesh",
):
    """Return a Plotly figure with mesh traces added."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plot_mesh_plotly requires plotly to be installed"
        ) from exc

    traces = mesh_to_plotly_data(mesh, axes=axes, name=name)
    if figure is None:
        figure = go.Figure()

    constructors = {
        "scatter": go.Scatter,
        "scatter3d": go.Scatter3d,
        "mesh3d": go.Mesh3d,
    }
    for trace in traces:
        trace_type = trace["type"]
        payload = {key: value for key, value in trace.items() if key != "type"}
        figure.add_trace(constructors[trace_type](**payload))
    return figure


__all__ = ["plot_mesh_matplotlib", "plot_mesh_plotly"]
