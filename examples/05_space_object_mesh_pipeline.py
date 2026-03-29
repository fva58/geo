"""End-to-end pipeline from spaces to native objects and meshes."""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo import SphereSpace, TorusSpace


def main(output_dir: str | Path | None = None) -> dict[str, object]:
    """Run the space -> object -> mesh pipeline and return a summary."""
    output_dir = Path(output_dir) if output_dir is not None else None

    sphere = SphereSpace()
    north = sphere.point_from_angles(0.0, math.pi / 2.0)
    cap = sphere.cap(north, math.pi / 3.0)
    cap_mesh = cap.mesh(resolution=16)

    torus = TorusSpace()
    patch = torus.patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))
    patch_mesh = patch.mesh(resolution=16)

    summary = {
        "sphere_cap_samples": len(cap.sample_points(resolution=16)),
        "sphere_cap_vertices": len(cap_mesh.vertices),
        "sphere_cap_cells": len(cap_mesh.cells),
        "torus_patch_samples": len(patch.sample_points(resolution=16)),
        "torus_patch_vertices": len(patch_mesh.vertices),
        "torus_patch_cells": len(patch_mesh.cells),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "space_object_mesh_summary.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n",
            encoding="utf-8",
        )

    return summary


if __name__ == "__main__":
    default_output_dir = Path(__file__).resolve().parent / "_generated"
    result = main(default_output_dir)
    for key, value in result.items():
        print(f"{key}: {value}")
