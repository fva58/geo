User Guide
==========

This guide collects a few typical workflows that the package already supports
well.

1. Build and combine sets on the real line and on the circle.
2. Use ready-made Euclidean objects from the object zoo.
3. Compose objects inside a Riemannian space.
4. Transport objects by projection and smooth maps.

Working With Sets on the Line and Circle
----------------------------------------

The lowest layer of the package is useful when the main task is set-theoretic
modeling rather than differential geometry.

.. code-block:: python

   import math

   from geo import FloatCircleSet, FloatSet

   line_set = FloatSet((0.0, 2.0), 5.0)
   circle_set = FloatCircleSet.from_single_interval(0.0, math.pi / 2.0)

   print(1.0 in line_set)
   print(3.0 in line_set)
   print(math.pi / 4.0 in circle_set)
   print(math.pi in circle_set)

This is a good fit for:

- finite unions of intervals on the real line;
- arcs on the unit circle;
- exact set operations on representable ``float`` values.

.. image:: _static/user_guide/line_and_circle_sets.png
   :alt: FloatSet on the line and FloatCircleSet on the circle
   :align: center
   :width: 90%

Using the Euclidean Object Zoo
------------------------------

The next layer provides ready-made geometric objects in Euclidean spaces. This
is the fastest way to build examples with interior, boundary, and local cone
models.

.. code-block:: python

   import math

   from geo import Ball, EllipsoidSurface, FloatPoint, HalfPlane, PlanarAngle

   ball = Ball(FloatPoint(0.0, 0.0), 1.5)
   upper_half_plane = HalfPlane((0.0, 1.0), offset=0.0)
   angle = PlanarAngle(FloatPoint(0.0, 0.0), 0.0, math.pi / 3.0)
   boundary = EllipsoidSurface(
       FloatPoint(0.0, 0.0),
       ((1.8, 0.0), (0.0, 1.0)),
   )

   print(FloatPoint(0.0, 0.0) in ball)
   print(FloatPoint(0.0, -1.0) in upper_half_plane)
   print(FloatPoint(1.0, 0.5) in angle)
   print(boundary.mesh().cells[:3])

This is the right layer for:

- balls, spheres, ellipsoids, cubes, and parallelepipeds;
- affine half-spaces and hyperplanes;
- planar examples with visible boundary structure.

.. image:: _static/user_guide/planar_object_zoo.png
   :alt: Ball, half-plane, planar angle, and ellipsoid surface
   :align: center
   :width: 85%

Working in a Riemannian Space
-----------------------------

The Riemannian layer wraps charted objects into a named ambient space and adds
set-theoretic operations between objects that live in the same space.

.. code-block:: python

   import math

   from geo import EuclideanPlaneSpace, FloatPoint

   plane = EuclideanPlaneSpace()
   upper = plane.half_plane((0.0, 1.0), offset=0.0)
   right = plane.half_plane((1.0, 0.0), offset=0.0)
   quadrant = upper & right
   angle = plane.angle(FloatPoint(0.0, 0.0), 0.0, math.pi / 2.0)

   print(FloatPoint(1.0, 1.0) in quadrant)
   print(FloatPoint(-1.0, 1.0) in quadrant)
   print(FloatPoint(1.0, 1.0) in angle)

Typical use cases are:

- intersections and unions inside one ambient space;
- examples where the ambient space matters explicitly;
- building reusable objects through the standard factories of
  ``RealLineSpace``, ``UnitCircleSpace``, and ``EuclideanPlaneSpace``.

Projecting Objects and Building Smooth Images
---------------------------------------------

Higher-level workflows become useful when an object should be transported to a
new place instead of being rebuilt from scratch.

.. code-block:: python

   import math

   from geo import EuclideanNeighborhood, EuclideanPlaneSpace, FloatPoint
   from geo import FloatVector, Hyperplane, ManifoldChart, RealLineSpace
   from geo import RiemannianGeometricObject

   plane = EuclideanPlaneSpace()
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

   print(FloatPoint(1.0, 0.0) in projected)
   print(FloatPoint(1.0, 1.0) in parabola)

Use this layer when you need:

- parallel or central projection onto a target hyperplane;
- visible caps and observer-facing subsets of convex Euclidean objects;
- geometric objects defined as images of source objects;
- workflows that mix explicit geometry with local differential structure.

.. image:: _static/user_guide/riemannian_workflows.png
   :alt: Intersection, projection, and smooth image workflows
   :align: center
   :width: 100%

Rebuilding the Images
---------------------

The static figures in this page are generated by::

   python doc/generate_user_guide_images.py

This keeps the documentation reproducible and avoids storing notebook output as
the only source of the illustrations.
