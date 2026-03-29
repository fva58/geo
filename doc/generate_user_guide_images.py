"""Generate static images used by the Sphinx user guide."""

from __future__ import annotations

import math
import sys
from pathlib import Path

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
    PlanarAngle,
    RealLineSpace,
    RiemannianGeometricObject,
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


def save_riemannian_workflows() -> None:
    """Save a figure for set operations, projections, and smooth images."""
    plane = EuclideanPlaneSpace()

    upper = plane.half_plane((0.0, 1.0), offset=0.0)
    right = plane.half_plane((1.0, 0.0), offset=0.0)
    quadrant = upper & right

    source_line = RiemannianGeometricObject.from_charted(
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
        title="Intersection in EuclideanPlaneSpace",
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
        OUTPUT_DIR / "riemannian_workflows.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_visibility_workflows() -> None:
    """Save a figure for visibility from a direction and from a point."""
    plane = EuclideanPlaneSpace()
    disk = RiemannianGeometricObject.from_charted(
        plane,
        Ball(FloatPoint(0.0, 0.0), 1.0),
    )
    top_half = disk.visible_from_direction((0.0, 1.0))

    ellipse = RiemannianGeometricObject.from_charted(
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
    save_visibility_workflows()
    save_riemannian_workflows()


if __name__ == "__main__":
    main()
