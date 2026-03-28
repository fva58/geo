Object Zoo
==========

The package now includes a small zoo of ready-made geometric objects on top
of the abstract local-model layer.

Zero-dimensional objects
------------------------

Singleton objects on the line, circle, and Euclidean spaces::

    >>> import math
    >>> from geo import CirclePointObject, EuclideanPointObject, RealPointObject
    >>> RealPointObject(2.0).contains(2.0)
    True
    >>> CirclePointObject(math.pi / 4.0).contains(math.pi / 4.0)
    True
    >>> EuclideanPointObject((1.0, 2.0)).contains((1.0, 2.0))
    True

Set-based one-dimensional objects
---------------------------------

Objects on the real line and on the unit circle built from the corresponding
set classes::

    >>> import math
    >>> from geo import CircleSetObject, FloatCircleSet, RealSetObject
    >>> line_object = RealSetObject((0.0, 2.0), 5.0)
    >>> 1.0 in line_object
    True
    >>> 3.0 in line_object
    False
    >>> circle_object = CircleSetObject(
    ...     FloatCircleSet.from_single_interval(0.0, math.pi / 2.0)
    ... )
    >>> math.pi / 4.0 in circle_object
    True
    >>> math.pi in circle_object
    False

Euclidean families
------------------

Whole spaces, affine objects, quadrics, and polyhedral examples are available
in arbitrary Euclidean dimension::

    >>> from geo import Ball, Cube, HalfSpace, Hyperplane, Sphere
    >>> Ball((0.0, 0.0, 0.0), 1.0).contains((0.0, 0.0, 0.0))
    True
    >>> Sphere((0.0, 0.0, 0.0), 1.0).contains((1.0, 0.0, 0.0))
    True
    >>> Hyperplane((0.0, 0.0, 1.0)).contains((1.0, 2.0, 0.0))
    True
    >>> HalfSpace((0.0, 0.0, 1.0)).contains((1.0, 2.0, -1.0))
    False
    >>> Cube((0.0, 0.0, 0.0), 1.0).contains((1.0, 1.0, 1.0))
    True

There are also affine-image variants for smooth and polyhedral families:

- ``Ellipsoid`` and ``EllipsoidSurface``;
- ``Parallelepiped`` and ``ParallelepipedSurface``;
- ``Cube`` and ``CubeSurface``.

Projections
-----------

Euclidean objects also support projections onto hyperplanes, either from a
point or along a fixed direction::

    >>> from geo import EuclideanPlaneSpace, FloatPoint, Hyperplane, RiemannianGeometricObject
    >>> plane = EuclideanPlaneSpace()
    >>> source_line = RiemannianGeometricObject.from_charted(
    ...     plane,
    ...     Hyperplane((0.0, 1.0), offset=1.0),
    ... )
    >>> source_half_line = source_line & plane.half_plane((1.0, 0.0), offset=0.0)
    >>> target_line = Hyperplane((0.0, 1.0), offset=0.0)
    >>> projected = source_half_line.project_along_direction_onto(
    ...     Hyperplane((0.0, 1.0), offset=1.0),
    ...     target_line,
    ...     (0.0, -1.0),
    ... )
    >>> FloatPoint(1.0, 0.0) in projected
    True
    >>> FloatPoint(-1.0, 0.0) in projected
    False

Visible parts
-------------

Convex Euclidean families also support visible parts, either from an observer
point or from a fixed direction::

    >>> from geo import Ball, EllipsoidSurface, FloatPoint, RiemannianGeometricObject
    >>> plane = EuclideanPlaneSpace()
    >>> disk = RiemannianGeometricObject.from_charted(
    ...     plane,
    ...     Ball((0.0, 0.0), 1.0),
    ... )
    >>> top_half = disk.visible_from_direction((0.0, 1.0))
    >>> FloatPoint(0.0, 1.0) in top_half
    True
    >>> FloatPoint(0.0, -1.0) in top_half
    False
    >>> ellipse = RiemannianGeometricObject.from_charted(
    ...     plane,
    ...     EllipsoidSurface((0.0, 0.0), ((2.0, 0.0), (0.0, 1.0))),
    ... )
    >>> visible = ellipse.visible_from_point((0.0, 3.0))
    >>> FloatPoint(0.0, 1.0) in visible
    True
    >>> FloatPoint(0.0, -1.0) in visible
    False

Smooth images
-------------

A geometric object can also be defined as the image of another object under a
smooth map with a local inverse on the image::

    >>> import math
    >>> from geo import EuclideanNeighborhood, EuclideanPlaneSpace, FloatPoint
    >>> from geo import FloatVector, ManifoldChart, RealLineSpace
    >>> source_space = RealLineSpace()
    >>> target_space = EuclideanPlaneSpace()
    >>> source = source_space.subset((0.0, 2.0))
    >>> def target_chart(point):
    ...     center = FloatPoint(point)
    ...     return ManifoldChart(
    ...         lambda candidate: FloatPoint(candidate) - center,
    ...         lambda coordinates: center + FloatVector(coordinates),
    ...         dim=2,
    ...         domain_contains=target_space.contains,
    ...         image=EuclideanNeighborhood.whole(2),
    ...     )
    >>> parabola = source.image_under_smooth_map(
    ...     lambda point: FloatPoint(point, point * point),
    ...     lambda point: float(FloatPoint(point)[0]),
    ...     target_space,
    ...     target_chart,
    ...     contains_image_point=lambda point: (
    ...         0.0 <= FloatPoint(point)[0] <= 2.0 and
    ...         math.isclose(
    ...             FloatPoint(point)[1],
    ...             FloatPoint(point)[0] * FloatPoint(point)[0],
    ...         )
    ...     ),
    ... )
    >>> FloatPoint(1.0, 1.0) in parabola
    True
    >>> FloatPoint(1.0, 0.0) in parabola
    False

Riemannian spaces with ready-made objects
-----------------------------------------

The Riemannian layer exposes standard spaces with object factories::

    >>> import math
    >>> from geo import EuclideanPlaneSpace, FloatPoint, UnitCircleSpace
    >>> plane = EuclideanPlaneSpace()
    >>> angle = plane.angle(FloatPoint(0.0, 0.0), 0.0, math.pi / 2.0)
    >>> FloatPoint(2.0, 3.0) in angle
    True
    >>> FloatPoint(-2.0, 3.0) in angle
    False
    >>> circle = UnitCircleSpace()
    >>> arc = circle.arc(0.0, math.pi / 2.0)
    >>> arc.local_model_at(0.0).cone.contains((1.0,))
    True
