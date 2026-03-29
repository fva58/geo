Intervals And Sets
==================

Real-line intervals and sets::

    >>> import math
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
    >>> adjacent = FloatInterval(0.0, 1.0) | FloatInterval(
    ...     math.nextafter(1.0, math.inf),
    ...     2.0,
    ... )
    >>> adjacent
    (FloatInterval(0.0, 2.0),)
    >>> FloatSet((0.0, 2.0), 5.0).to_tuple()
    ((0.0, 2.0), (5.0, 5.0))

The implementation keeps closed intervals even for set difference. Since the
package currently models machine ``float`` values rather than abstract real
numbers, removed boundary points are excluded by stepping to adjacent
representable values with ``math.nextafter``.

The same float-lattice model also controls normalization: if there is no
representable ``float`` between two intervals, they are merged into one closed
interval.

``FloatSet`` accepts explicit constructor inputs only:

- scalars are interpreted as points;
- ``FloatInterval`` values are kept as intervals;
- a single numeric pair ``(left, right)`` means one interval;
- sequences are flattened only when they contain supported values.

Unsupported inputs raise ``TypeError`` instead of being recursively guessed.
