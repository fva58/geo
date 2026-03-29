"""Adapters from ``ObjectMesh`` to plotting and export data formats."""

from __future__ import annotations

from typing import Sequence

from .euclidean import FloatPoint
from .geometric import ObjectMesh


def _project_if_requested(
    mesh: ObjectMesh,
    axes: Sequence[int] | None,
) -> ObjectMesh:
    """Return the mesh projected to selected axes when requested."""
    if axes is None:
        return mesh
    return mesh.projected(tuple(axes))


def _point_lists(mesh: ObjectMesh) -> tuple[list[float], ...]:
    """Return coordinate-wise vertex lists."""
    return tuple(
        [float(vertex[axis]) for vertex in mesh.vertices]
        for axis in range(mesh.dim)
    )


def _segments(mesh: ObjectMesh) -> tuple[tuple[FloatPoint, FloatPoint], ...]:
    """Return geometric line segments derived from mesh edges."""
    return tuple(
        (mesh.vertices[start], mesh.vertices[end])
        for start, end in mesh.edge_indices()
    )


def _triangulated_cells(mesh: ObjectMesh) -> tuple[tuple[int, int, int], ...]:
    """Return triangle indices by fan-triangulating cells."""
    triangles = []
    for cell in mesh.cells:
        if len(cell) < 3:
            continue
        if len(cell) == 3:
            triangles.append((cell[0], cell[1], cell[2]))
            continue
        for index in range(1, len(cell) - 1):
            triangles.append((cell[0], cell[index], cell[index + 1]))
    return tuple(triangles)


def mesh_to_wireframe_data(
    mesh: ObjectMesh,
    axes: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return generic wireframe data from an object mesh."""
    mesh = _project_if_requested(mesh, axes)
    return {
        "dim": mesh.dim,
        "vertices": [tuple(float(value) for value in vertex) for vertex in mesh.vertices],
        "edges": list(mesh.edge_indices()),
        "cells": [tuple(int(index) for index in cell) for cell in mesh.cells],
    }


def mesh_to_matplotlib_data(
    mesh: ObjectMesh,
    axes: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return plain data structures shaped for Matplotlib-style plotting."""
    mesh = _project_if_requested(mesh, axes)
    coordinates = _point_lists(mesh)
    segments = _segments(mesh)
    triangles = _triangulated_cells(mesh)

    data = {
        "dim": mesh.dim,
        "points": {
            "x": coordinates[0],
            "y": coordinates[1] if mesh.dim >= 2 else [0.0] * len(mesh.vertices),
        },
        "line_segments": [
            tuple(tuple(float(value) for value in endpoint) for endpoint in segment)
            for segment in segments
        ],
        "triangles": [
            tuple(
                tuple(float(value) for value in mesh.vertices[index])
                for index in triangle
            )
            for triangle in triangles
        ],
    }
    if mesh.dim >= 3:
        data["points"]["z"] = coordinates[2]
    return data


def mesh_to_plotly_data(
    mesh: ObjectMesh,
    axes: Sequence[int] | None = None,
    name: str = "mesh",
) -> list[dict[str, object]]:
    """Return Plotly-compatible trace dictionaries."""
    mesh = _project_if_requested(mesh, axes)
    coordinates = _point_lists(mesh)
    segments = mesh.edge_indices()
    triangles = _triangulated_cells(mesh)
    traces: list[dict[str, object]] = []

    if mesh.dim == 2:
        if segments:
            x_values = []
            y_values = []
            for start, end in segments:
                x_values.extend([coordinates[0][start], coordinates[0][end], None])
                y_values.extend([coordinates[1][start], coordinates[1][end], None])
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"{name}-edges",
                    "x": x_values,
                    "y": y_values,
                }
            )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": f"{name}-points",
                "x": coordinates[0],
                "y": coordinates[1],
            }
        )
        return traces

    if mesh.dim == 3 and triangles:
        traces.append(
            {
                "type": "mesh3d",
                "name": f"{name}-surface",
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
                "i": [triangle[0] for triangle in triangles],
                "j": [triangle[1] for triangle in triangles],
                "k": [triangle[2] for triangle in triangles],
            }
        )

    if mesh.dim == 3 and segments:
        x_values = []
        y_values = []
        z_values = []
        for start, end in segments:
            x_values.extend([coordinates[0][start], coordinates[0][end], None])
            y_values.extend([coordinates[1][start], coordinates[1][end], None])
            z_values.extend([coordinates[2][start], coordinates[2][end], None])
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"{name}-edges",
                "x": x_values,
                "y": y_values,
                "z": z_values,
            }
        )

    if mesh.dim == 3:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{name}-points",
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
            }
        )

    return traces


def mesh_to_threejs_data(mesh: ObjectMesh) -> dict[str, object]:
    """Return Three.js-style indexed-geometry data."""
    triangles = _triangulated_cells(mesh)
    return {
        "position": [
            float(value)
            for vertex in mesh.vertices
            for value in vertex
        ],
        "line_indices": [
            index
            for edge in mesh.edge_indices()
            for index in edge
        ],
        "triangle_indices": [
            index
            for triangle in triangles
            for index in triangle
        ],
        "vertex_count": len(mesh.vertices),
        "dim": mesh.dim,
    }


__all__ = [
    "mesh_to_wireframe_data",
    "mesh_to_matplotlib_data",
    "mesh_to_plotly_data",
    "mesh_to_threejs_data",
]
