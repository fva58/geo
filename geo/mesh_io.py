"""Text and file exporters for ``ObjectMesh``."""

from __future__ import annotations

import json
from pathlib import Path

from .geometric import ObjectMesh
from .mesh_export import mesh_to_threejs_data


def _vertex_xyz(vertex) -> tuple[float, float, float]:
    """Return a 3D vertex tuple, padding lower dimensions with zeros."""
    values = tuple(float(value) for value in vertex)
    if len(values) == 1:
        return (values[0], 0.0, 0.0)
    if len(values) == 2:
        return (values[0], values[1], 0.0)
    return (values[0], values[1], values[2])


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


def mesh_to_obj_text(mesh: ObjectMesh) -> str:
    """Return a Wavefront OBJ representation of the mesh."""
    lines = ["# geo ObjectMesh"]
    for vertex in mesh.vertices:
        x_value, y_value, z_value = _vertex_xyz(vertex)
        lines.append(f"v {x_value} {y_value} {z_value}")
    for start, end in mesh.edge_indices():
        lines.append(f"l {start + 1} {end + 1}")
    for first, second, third in _triangulated_cells(mesh):
        lines.append(f"f {first + 1} {second + 1} {third + 1}")
    return "\n".join(lines) + "\n"


def mesh_to_ply_text(mesh: ObjectMesh) -> str:
    """Return an ASCII PLY representation of the mesh."""
    triangles = _triangulated_cells(mesh)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(mesh.vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(triangles)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    vertex_lines = [
        f"{x_value} {y_value} {z_value}"
        for x_value, y_value, z_value in (_vertex_xyz(vertex) for vertex in mesh.vertices)
    ]
    face_lines = [
        f"3 {first} {second} {third}"
        for first, second, third in triangles
    ]
    return "\n".join(header + vertex_lines + face_lines) + "\n"


def mesh_to_gltf_json_data(mesh: ObjectMesh) -> dict[str, object]:
    """Return a glTF-friendly JSON geometry structure."""
    threejs = mesh_to_threejs_data(mesh)
    return {
        "asset": {
            "version": "2.0",
            "generator": "geo",
        },
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 4,
                        "attributes": {
                            "POSITION": threejs["position"],
                        },
                        "indices": threejs["triangle_indices"],
                    },
                    {
                        "mode": 1,
                        "attributes": {
                            "POSITION": threejs["position"],
                        },
                        "indices": threejs["line_indices"],
                    },
                ],
            }
        ],
        "extras": {
            "vertex_count": threejs["vertex_count"],
            "dim": threejs["dim"],
        },
    }


def write_obj(mesh: ObjectMesh, path: str | Path) -> Path:
    """Write an OBJ file and return the target path."""
    target = Path(path)
    target.write_text(mesh_to_obj_text(mesh), encoding="utf-8")
    return target


def write_ply(mesh: ObjectMesh, path: str | Path) -> Path:
    """Write an ASCII PLY file and return the target path."""
    target = Path(path)
    target.write_text(mesh_to_ply_text(mesh), encoding="utf-8")
    return target


def write_gltf_json(mesh: ObjectMesh, path: str | Path) -> Path:
    """Write a glTF-friendly JSON file and return the target path."""
    target = Path(path)
    target.write_text(
        json.dumps(mesh_to_gltf_json_data(mesh), indent=2),
        encoding="utf-8",
    )
    return target


__all__ = [
    "mesh_to_obj_text",
    "mesh_to_ply_text",
    "mesh_to_gltf_json_data",
    "write_obj",
    "write_ply",
    "write_gltf_json",
]
