geo
===

``geo`` is an early-stage geometry package built around immutable interval
sets.

Current model
-------------

At the current stage, real numbers are modeled by Python ``float``.

This has two direct consequences:

- interval and set operations work on the discrete lattice of representable
  floating-point values;
- when a set operation removes a boundary point but still returns closed
  intervals, the implementation uses ``math.nextafter`` to move to the nearest
  representable float.

Main types
----------

Real line:

- ``FloatInterval``: one closed interval of ``float`` values;
- ``FloatSet``: normalized union of disjoint ``FloatInterval`` objects.

Circle:

- ``FloatAngle``: angle normalized to ``[0, 2π)``;
- ``FloatCirclePoint``: point on the unit circle represented by an angle;
- ``FloatCircleInterval``: connected arc on the circle;
- ``FloatCircleSet``: set of circle points built on top of ``FloatSet``.

For arcs crossing zero, the internal representation is split into two linear
intervals: one before zero and one after zero.

Examples
--------

Real-line intervals and sets::

    >>> from geo import FloatInterval, FloatSet
    >>> interval = FloatInterval(0.0, 5.0)
    >>> diff = interval.difference(FloatInterval(2.0, 3.0))
    >>> len(diff)
    2
    >>> diff[0].contains(2.0)
    False
    >>> diff[1].contains(3.0)
    False
    >>> fset = FloatSet.from_single_interval(0.0, 1.0)
    >>> (fset | FloatSet.from_single_interval(2.0, 3.0)).to_tuple()
    ((0.0, 1.0), (2.0, 3.0))

Circle intervals::

    >>> import math
    >>> from geo import FloatCircleInterval
    >>> arc = FloatCircleInterval(3 * math.pi / 2, math.pi / 2)
    >>> arc.to_tuple()
    (4.71238898038469, 1.5707963267948966)
    >>> len(arc)
    2
    >>> list(arc)
    [(0.0, 1.5707963267948966), (4.71238898038469, 6.283185307179585)]
    >>> 0.0 in arc
    True
    >>> math.pi in arc
    False

Top-level imports::

    >>> from geo import FloatAngle, FloatCircleSet, FloatSet, real
    >>> real is float
    True
    >>> FloatAngle(2 * math.pi).value
    0.0

Status
------

The package is still under active development. The current focus is:

1. stabilize the ``float``-based core;
2. keep the public API coherent;
3. extend the geometric model only after the interval foundation is stable.
