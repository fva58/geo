Circle Geometry
===============

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

For arcs crossing zero, the internal representation is split into two linear
intervals: one before zero and one after zero.

Top-level imports::

    >>> from geo import FloatAngle, FloatCircleSet, FloatSet, real
    >>> real is float
    True
    >>> FloatAngle(2 * math.pi).value
    0.0
