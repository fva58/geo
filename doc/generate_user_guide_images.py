"""Generate static images used by the Sphinx user guide."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from geo import (
    Ball,
    EllipsoidSurface,
    EuclideanNeighborhood,
    EuclideanPlaneSpace,
    FloatCircleSet,
    FloatPoint,
    FloatVector,
    HalfPlane,
    Hyperplane,
    ManifoldChart,
    MetricGeometricObject,
    PlanarAngle,
    RealLineSpace,
    SphereSpace,
    TorusPoint,
    TorusSpace,
)


OUTPUT_DIR = ROOT / "_static" / "user_guide"


def membership_grid(
    obj,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a rasterized membership grid for a planar object."""
    xs = np.linspace(xlim[0], xlim[1], size)
    ys = np.linspace(ylim[0], ylim[1], size)
    grid = np.zeros((size, size), dtype=float)
    for y_index, y_value in enumerate(ys):
        for x_index, x_value in enumerate(xs):
            grid[y_index, x_index] = (
                1.0 if FloatPoint(x_value, y_value) in obj else 0.0
            )
    return xs, ys, grid


def draw_mesh(
    ax,
    obj,
    *,
    axes: tuple[int, int] = (0, 1),
    bounds: tuple[tuple[float, float], ...] | None = None,
    color: str = "#1f1f1f",
    linewidth: float = 2.0,
    resolution: int = 192,
) -> bool:
    """Draw a planar mesh representation when the object exposes one."""
    mesh_source = getattr(obj, "_charted_source_object", obj)
    try:
        mesh = mesh_source.mesh(resolution=resolution, bounds=bounds)
    except (AttributeError, NotImplementedError, ValueError):
        return False

    if not mesh.vertices:
        return False

    if mesh.dim > 2:
        mesh = mesh.projected(axes)

    if mesh.dim != 2 or not mesh.vertices:
        return False

    vertices = np.asarray([tuple(vertex) for vertex in mesh.vertices], dtype=float)
    edges = mesh.edge_indices()
    if not edges:
        ax.scatter(vertices[:, 0], vertices[:, 1], s=20.0, c=color)
        return True

    for start, end in sorted(edges):
        ax.plot(
            vertices[[start, end], 0],
            vertices[[start, end], 1],
            color=color,
            linewidth=linewidth,
        )
    return True


def draw_projected_mesh(
    ax,
    mesh,
    *,
    axes: tuple[int, int] = (0, 1),
    color: str = "#1f1f1f",
    linewidth: float = 1.8,
    point_size: float = 14.0,
) -> bool:
    """Draw a projected mesh directly."""
    if not mesh.vertices:
        return False
    if mesh.dim > 2:
        mesh = mesh.projected(axes)
    if mesh.dim != 2 or not mesh.vertices:
        return False

    vertices = np.asarray([tuple(vertex) for vertex in mesh.vertices], dtype=float)
    edges = mesh.edge_indices()
    if edges:
        for start, end in sorted(edges):
            ax.plot(
                vertices[[start, end], 0],
                vertices[[start, end], 1],
                color=color,
                linewidth=linewidth,
            )
    ax.scatter(vertices[:, 0], vertices[:, 1], s=point_size, c=color, zorder=3)
    return True


def _ordered_polyline(mesh) -> list[int] | None:
    """Return an ordered polyline for a mesh built from edges."""
    if not mesh.cells or not all(len(cell) == 2 for cell in mesh.cells):
        return None

    adjacency = {index: set() for index in range(len(mesh.vertices))}
    for start, end in mesh.cells:
        adjacency[start].add(end)
        adjacency[end].add(start)

    degrees = {len(neighbors) for neighbors in adjacency.values()}
    if degrees == {2}:
        start = 0
        order = [start]
        previous = None
        current = start
        while True:
            excluded = {previous} if previous is not None else set()
            choices = sorted(adjacency[current] - excluded)
            if not choices:
                break
            nxt = choices[0]
            if nxt == start:
                break
            order.append(nxt)
            previous, current = current, nxt
            if len(order) > len(mesh.vertices):
                return None
        return order

    endpoints = [
        index for index, neighbors in adjacency.items()
        if len(neighbors) == 1
    ]
    if len(endpoints) != 2:
        return None

    start = min(endpoints)
    order = [start]
    previous = None
    current = start
    while True:
        excluded = {previous} if previous is not None else set()
        choices = sorted(adjacency[current] - excluded)
        if not choices:
            break
        nxt = choices[0]
        order.append(nxt)
        previous, current = current, nxt
        if len(order) > len(mesh.vertices):
            return None
    return order


def draw_object(
    ax,
    obj,
    *,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    mesh_only: bool = False,
) -> None:
    """Draw a planar object with raster fill and optional mesh overlay."""
    xs, ys, grid = membership_grid(obj, xlim=xlim, ylim=ylim)
    if np.any(grid) and not mesh_only:
        ax.imshow(
            grid,
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            origin="lower",
            cmap="Blues",
            alpha=0.8,
            interpolation="nearest",
        )
    mesh_source = getattr(obj, "_charted_source_object", obj)
    try:
        mesh = mesh_source.mesh(bounds=(xlim, ylim))
    except (AttributeError, NotImplementedError, ValueError):
        mesh = None

    if mesh is not None and mesh.vertices and mesh.dim == 2:
        polyline = _ordered_polyline(mesh)
        if polyline is not None and not np.any(grid) and len(polyline) >= 3:
            vertices = np.asarray(
                [tuple(vertex) for vertex in mesh.vertices],
                dtype=float,
            )
            polygon = vertices[polyline]
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="#ff9896",
                alpha=0.18,
                zorder=1,
            )
    draw_mesh(ax, obj, bounds=(xlim, ylim))
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def save_line_and_circle_sets() -> None:
    """Save a figure for set-based line and circle workflows."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    line_ax, circle_ax = axes

    line_ax.axhline(0.0, color="#666666", linewidth=1.0)
    line_ax.plot([0.0, 2.0], [0.0, 0.0], color="#1f77b4", linewidth=8.0)
    line_ax.scatter(
        [0.0, 2.0],
        [0.0, 0.0],
        s=90.0,
        color="#1f77b4",
        zorder=3,
    )
    line_ax.scatter([5.0], [0.0], s=90.0, color="#d62728", zorder=3)
    line_ax.set_xlim(-1.0, 6.0)
    line_ax.set_ylim(-1.0, 1.0)
    line_ax.set_yticks([])
    line_ax.set_title("FloatSet on the real line")
    line_ax.grid(True, axis="x", alpha=0.2)

    theta = np.linspace(0.0, 2.0 * math.pi, 512)
    circle_ax.plot(np.cos(theta), np.sin(theta), color="#bbbbbb", linewidth=1.5)
    arc = FloatCircleSet.from_single_interval(0.0, math.pi / 2.0)
    assert math.pi / 4.0 in arc
    arc_angles = np.linspace(0.0, math.pi / 2.0, 128)
    circle_ax.plot(
        np.cos(arc_angles),
        np.sin(arc_angles),
        color="#ff7f0e",
        linewidth=5.0,
    )
    circle_ax.scatter(
        [1.0, 0.0],
        [0.0, 1.0],
        s=80.0,
        color="#ff7f0e",
        zorder=3,
    )
    circle_ax.set_title("FloatCircleSet arc")
    circle_ax.set_aspect("equal")
    circle_ax.set_xlim(-1.2, 1.2)
    circle_ax.set_ylim(-1.2, 1.2)
    circle_ax.grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "line_and_circle_sets.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_planar_object_zoo() -> None:
    """Save a figure for ready-made planar Euclidean objects."""
    objects = [
        (Ball(FloatPoint(0.0, 0.0), 1.5), "Ball", False),
        (HalfPlane((0.0, 1.0), offset=0.0), "HalfPlane", False),
        (
            PlanarAngle(FloatPoint(0.0, 0.0), 0.0, math.pi / 3.0),
            "PlanarAngle",
            False,
        ),
        (
            EllipsoidSurface(
                FloatPoint(0.0, 0.0),
                ((1.8, 0.0), (0.0, 1.0)),
            ),
            "EllipsoidSurface",
            True,
        ),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (obj, title, mesh_only) in zip(axes.flat, objects):
        draw_object(
            ax,
            obj,
            title=title,
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            mesh_only=mesh_only,
        )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "planar_object_zoo.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_metric_workflows() -> None:
    """Save a figure for set operations, projections, and smooth images."""
    plane = EuclideanPlaneSpace()

    upper = plane.half_plane((0.0, 1.0), offset=0.0)
    right = plane.half_plane((1.0, 0.0), offset=0.0)
    quadrant = upper & right

    source_line = MetricGeometricObject.from_charted(
        plane,
        Hyperplane((0.0, 1.0), offset=1.0),
    )
    source_half_line = source_line & plane.half_plane((1.0, 0.0), offset=0.0)
    target_line = Hyperplane((0.0, 1.0), offset=0.0)
    projected = source_half_line.project_along_direction_onto(
        Hyperplane((0.0, 1.0), offset=1.0),
        target_line,
        (0.0, -1.0),
    )

    source_space = RealLineSpace()
    interval = source_space.subset((0.0, 2.0))

    def target_chart(point):
        center = FloatPoint(point)
        return ManifoldChart(
            lambda candidate: FloatPoint(candidate) - center,
            lambda coordinates: center + FloatVector(coordinates),
            dim=2,
            domain_contains=plane.contains,
            image=EuclideanNeighborhood.whole(2),
        )

    parabola = interval.image_under_smooth_map(
        lambda point: FloatPoint(point, point * point),
        lambda point: float(FloatPoint(point)[0]),
        plane,
        target_chart,
        contains_image_point=lambda point: (
            0.0 <= FloatPoint(point)[0] <= 2.0 and
            math.isclose(
                FloatPoint(point)[1],
                FloatPoint(point)[0] * FloatPoint(point)[0],
            )
        ),
    )

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    draw_object(
        axes[0],
        quadrant,
        title="Intersection in a metric space",
        xlim=(-0.5, 3.0),
        ylim=(-0.5, 3.0),
    )

    axes[1].plot([0.0, 2.5], [1.0, 1.0], color="#d62728", linewidth=4.0)
    axes[1].plot([0.0, 2.5], [0.0, 0.0], color="#1f77b4", linewidth=4.0)
    for x_value in np.linspace(0.3, 2.2, 5):
        axes[1].annotate(
            "",
            xy=(x_value, 0.05),
            xytext=(x_value, 0.95),
            arrowprops={"arrowstyle": "->", "color": "#555555"},
        )
    axes[1].set_title("Projection onto a hyperplane")
    axes[1].set_xlim(-0.3, 3.0)
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.2)

    xs = np.linspace(0.0, 2.0, 200)
    ys = xs * xs
    axes[2].plot(xs, ys, color="#2ca02c", linewidth=3.0)
    sample_points = [FloatPoint(0.0, 0.0), FloatPoint(1.0, 1.0), FloatPoint(2.0, 4.0)]
    axes[2].scatter(
        [point[0] for point in sample_points],
        [point[1] for point in sample_points],
        color="#2ca02c",
        s=50.0,
        zorder=3,
    )
    axes[2].set_title("Smooth image of an interval")
    axes[2].set_xlim(-0.1, 2.1)
    axes[2].set_ylim(-0.2, 4.3)
    axes[2].set_aspect(0.5)
    axes[2].grid(True, alpha=0.2)

    assert FloatPoint(1.0, 0.0) in projected
    assert FloatPoint(1.0, 1.0) in parabola

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "metric_workflows.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_space_visualizations() -> None:
    """Save a figure for the visualization-aware space layer."""
    sphere = SphereSpace()
    north = sphere.point_from_angles(0.0, math.pi / 2.0)
    cap = sphere.cap(north, math.pi / 3.0)
    sphere_samples = sphere.sample_points(resolution=28)
    cap_samples = cap.sample_points(resolution=28)

    torus = TorusSpace()
    patch = torus.patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))
    torus_samples = torus.sample_points(resolution=22)
    patch_samples = patch.sample_points(resolution=22)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    sphere_ax, torus_ax = axes
    sphere_xy = np.asarray(
        [sphere.to_2d(point, method="equirectangular") for point in sphere_samples],
        dtype=float,
    )
    cap_xy = np.asarray(
        [sphere.to_2d(point, method="equirectangular") for point in cap_samples],
        dtype=float,
    )
    sphere_ax.scatter(
        sphere_xy[:, 0],
        sphere_xy[:, 1],
        s=14.0,
        color="#cfcfcf",
        alpha=0.7,
    )
    sphere_ax.scatter(
        cap_xy[:, 0],
        cap_xy[:, 1],
        s=18.0,
        color="#d62728",
        alpha=0.9,
    )
    sphere_ax.set_title("SphereSpace in equirectangular coordinates")
    sphere_ax.set_xlabel("longitude")
    sphere_ax.set_ylabel("latitude")
    sphere_ax.grid(True, alpha=0.2)

    torus_xy = np.asarray([torus.to_2d(point) for point in torus_samples], dtype=float)
    patch_xy = np.asarray([torus.to_2d(point) for point in patch_samples], dtype=float)
    torus_ax.scatter(
        torus_xy[:, 0],
        torus_xy[:, 1],
        s=14.0,
        color="#cfcfcf",
        alpha=0.65,
    )
    torus_ax.scatter(
        patch_xy[:, 0],
        patch_xy[:, 1],
        s=18.0,
        color="#1f77b4",
        alpha=0.9,
    )
    torus_ax.set_title("TorusSpace in flat angular coordinates")
    torus_ax.set_xlabel("major angle")
    torus_ax.set_ylabel("minor angle")
    torus_ax.set_xlim(-0.2, 2.0 * math.pi + 0.2)
    torus_ax.set_ylim(-0.2, 2.0 * math.pi + 0.2)
    torus_ax.grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "space_visualizations.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_space_meshes() -> None:
    """Save a figure for native-object meshing workflows."""
    sphere = SphereSpace()
    north = sphere.point_from_angles(0.0, math.pi / 2.0)
    cap = sphere.cap(north, math.pi / 3.0)
    cap_mesh = cap.mesh(resolution=14)

    torus = TorusSpace()
    patch = torus.patch((0.0, math.pi / 2.0), (0.0, math.pi / 2.0))
    patch_mesh = patch.mesh(resolution=14)

    line = RealLineSpace().subset((0.0, 2.0), 5.0)
    line_mesh = line.mesh(resolution=12)

    circle = EuclideanPlaneSpace()
    disk = MetricGeometricObject.from_charted(
        circle,
        Ball(FloatPoint(0.0, 0.0), 1.0),
        name="disk",
    )
    disk_mesh = disk.mesh(resolution=14)

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))

    draw_projected_mesh(
        axes[0, 0],
        cap_mesh,
        axes=(0, 2),
        color="#d62728",
    )
    axes[0, 0].set_title("Spherical cap mesh (projected)")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].grid(True, alpha=0.2)

    draw_projected_mesh(
        axes[0, 1],
        patch_mesh,
        axes=(0, 1),
        color="#1f77b4",
    )
    axes[0, 1].set_title("Torus patch mesh (projected)")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].grid(True, alpha=0.2)

    draw_projected_mesh(
        axes[1, 0],
        line_mesh,
        color="#2ca02c",
    )
    axes[1, 0].set_title("Real-line subset mesh")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(True, alpha=0.2)

    draw_projected_mesh(
        axes[1, 1],
        disk_mesh,
        color="#9467bd",
    )
    axes[1, 1].set_title("Wrapped Euclidean object mesh")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "space_meshes.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_mesh_exports() -> None:
    """Save a figure for mesh export and plotting workflows."""
    mesh = Ball(FloatPoint(0.0, 0.0), 1.0).mesh(resolution=18)
    projected = mesh.projected((0, 1))
    wireframe = projected.wireframe_data()
    plotly_data = projected.plotly_data(name="disk")
    threejs_data = projected.threejs_data()

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    draw_projected_mesh(
        axes[0],
        projected,
        color="#1f77b4",
    )
    axes[0].set_title("Matplotlib-style mesh view")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2)

    axes[1].axis("off")
    axes[1].set_title("Wireframe / Plotly summary")
    axes[1].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"vertices: {len(wireframe['vertices'])}",
                f"edges: {len(wireframe['edges'])}",
                f"cells: {len(wireframe['cells'])}",
                f"plotly traces: {len(plotly_data)}",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    axes[2].axis("off")
    axes[2].set_title("Three.js / file export summary")
    axes[2].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"position values: {len(threejs_data['position'])}",
                f"line indices: {len(threejs_data['line_indices'])}",
                f"triangle indices: {len(threejs_data['triangle_indices'])}",
                "file targets: OBJ / PLY / glTF JSON",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mesh_exports.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_visibility_workflows() -> None:
    """Save a figure for visibility from a direction and from a point."""
    plane = EuclideanPlaneSpace()
    disk = MetricGeometricObject.from_charted(
        plane,
        Ball(FloatPoint(0.0, 0.0), 1.0),
    )
    top_half = disk.visible_from_direction((0.0, 1.0))

    ellipse = MetricGeometricObject.from_charted(
        plane,
        EllipsoidSurface(
            FloatPoint(0.0, 0.0),
            ((2.0, 0.0), (0.0, 1.0)),
        ),
    )
    observer = FloatPoint(0.0, 3.0)
    visible_arc = ellipse.visible_from_point(observer)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    draw_object(
        axes[0],
        disk,
        title="Visible part of a ball from a direction",
        xlim=(-1.4, 1.4),
        ylim=(-1.4, 1.4),
    )
    top_angles = np.linspace(0.0, math.pi, 200)
    axes[0].plot(
        np.cos(top_angles),
        np.sin(top_angles),
        color="#d62728",
        linewidth=3.0,
    )
    for x_value in np.linspace(-0.8, 0.8, 5):
        axes[0].annotate(
            "",
            xy=(x_value, 1.25),
            xytext=(x_value, 1.6),
            arrowprops={"arrowstyle": "->", "color": "#555555"},
        )

    draw_object(
        axes[1],
        ellipse,
        title="Observer-facing part of an ellipsoid surface",
        xlim=(-2.4, 2.4),
        ylim=(-1.4, 3.4),
        mesh_only=True,
    )
    visible_xs = np.linspace(-1.2, 1.2, 200)
    visible_ys = np.sqrt(np.maximum(0.0, 1.0 - (visible_xs / 2.0) ** 2))
    axes[1].plot(
        visible_xs,
        visible_ys,
        color="#d62728",
        linewidth=3.0,
    )
    axes[1].scatter([observer[0]], [observer[1]], color="#2ca02c", s=55.0, zorder=3)
    for target in [
        FloatPoint(-1.2, 0.8),
        FloatPoint(0.0, 1.0),
        FloatPoint(1.2, 0.8),
    ]:
        axes[1].plot(
            [observer[0], target[0]],
            [observer[1], target[1]],
            color="#2ca02c",
            linewidth=1.5,
            alpha=0.8,
        )

    assert FloatPoint(0.0, 1.0) in top_half
    assert FloatPoint(0.0, -1.0) not in top_half
    assert FloatPoint(1.2, 0.8) in visible_arc
    assert FloatPoint(0.0, -1.0) not in visible_arc

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "visibility_workflows.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    result_figure, result_axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    draw_object(
        result_axes[0],
        top_half,
        title="Result of visible_from_direction",
        xlim=(-1.4, 1.4),
        ylim=(-1.4, 1.4),
        mesh_only=True,
    )

    draw_object(
        result_axes[1],
        visible_arc,
        title="Result of visible_from_point",
        xlim=(-2.4, 2.4),
        ylim=(-1.4, 3.4),
        mesh_only=True,
    )
    result_axes[1].scatter(
        [observer[0]],
        [observer[1]],
        color="#2ca02c",
        s=55.0,
        zorder=3,
    )

    result_figure.tight_layout()
    result_figure.savefig(
        OUTPUT_DIR / "visibility_results.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(result_figure)


def main() -> None:
    """Generate every image used by the user guide."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_line_and_circle_sets()
    save_planar_object_zoo()
    save_space_visualizations()
    save_space_meshes()
    save_mesh_exports()
    save_visibility_workflows()
    save_metric_workflows()


if __name__ == "__main__":
    main()
