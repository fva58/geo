"""End-to-end pipeline through the broader metric object zoo."""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo import Ball, EuclideanPlaneSpace, FloatPoint, RealLineSpace, UnitCircleSpace


def main(output_dir: str | Path | None = None) -> dict[str, object]:
    """Run the metric-object zoo pipeline and return a summary."""
    output_dir = Path(output_dir) if output_dir is not None else None

    line_object = RealLineSpace().subset((0.0, 2.0), 5.0)
    arc_object = UnitCircleSpace().arc(0.0, math.pi / 2.0)
    disk_object = EuclideanPlaneSpace().half_plane((0.0, 1.0), offset=0.0)
    ball_object = EuclideanPlaneSpace()
    wrapped_disk = ball_object.point(FloatPoint(0.0, 0.0))

    line_mesh = line_object.mesh(resolution=8)
    arc_mesh = arc_object.mesh(resolution=12)
    point_mesh = wrapped_disk.mesh(resolution=8)
    ball_mesh = Ball(FloatPoint(0.0, 0.0), 1.0).mesh(resolution=12)

    summary = {
        "line_samples": len(line_object.sample_points(resolution=8)),
        "line_cells": len(line_mesh.cells),
        "arc_samples": len(arc_object.sample_points(resolution=12)),
        "arc_cells": len(arc_mesh.cells),
        "point_samples": len(wrapped_disk.sample_points(resolution=8)),
        "point_cells": len(point_mesh.cells),
        "ball_vertices": len(ball_mesh.vertices),
        "ball_cells": len(ball_mesh.cells),
        "half_plane_contains_origin": FloatPoint(0.0, 0.0) in disk_object,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metric_object_zoo_summary.txt").write_text(
            "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n",
            encoding="utf-8",
        )

    return summary


if __name__ == "__main__":
    default_output_dir = Path(__file__).resolve().parent / "_generated"
    result = main(default_output_dir)
    for key, value in result.items():
        print(f"{key}: {value}")
