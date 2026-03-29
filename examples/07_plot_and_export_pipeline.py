"""End-to-end pipeline from meshes to plots and files."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from geo import (
    Ball,
    FloatPoint,
    plot_mesh_matplotlib,
    plot_mesh_plotly,
    write_gltf_json,
    write_obj,
    write_ply,
)


def main(output_dir: str | Path | None = None) -> dict[str, object]:
    """Run the mesh -> plot/export pipeline and return a summary."""
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "_generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = Ball(FloatPoint(0.0, 0.0), 1.0).mesh(resolution=12)

    figure, axis = plot_mesh_matplotlib(mesh)
    png_path = output_dir / "disk_mesh.png"
    figure.savefig(png_path, dpi=120)

    plotly_trace_count = 0
    plotly_error = ""
    try:
        plotly_figure = plot_mesh_plotly(mesh, name="disk")
        plotly_trace_count = len(plotly_figure.data)
    except ImportError as exc:
        plotly_error = str(exc)

    obj_path = write_obj(mesh, output_dir / "disk_mesh.obj")
    ply_path = write_ply(mesh, output_dir / "disk_mesh.ply")
    gltf_path = write_gltf_json(mesh, output_dir / "disk_mesh.gltf.json")

    summary = {
        "png_exists": png_path.exists(),
        "obj_exists": obj_path.exists(),
        "ply_exists": ply_path.exists(),
        "gltf_exists": gltf_path.exists(),
        "plotly_trace_count": plotly_trace_count,
        "plotly_error": plotly_error,
    }
    (output_dir / "plot_and_export_summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = main()
    for key, value in result.items():
        print(f"{key}: {value}")
