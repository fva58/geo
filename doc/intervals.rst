Intervals And Sets
==================

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

The implementation keeps closed intervals even for set difference. Since the
package currently models machine ``float`` values rather than abstract real
numbers, removed boundary points are excluded by stepping to adjacent
representable values with ``math.nextafter``.
