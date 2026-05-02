"""Compatibility layer for older metric-oriented names."""

from __future__ import annotations

from .gobject import (
    GeometricObject as MetricGeometricObject,
    LazyExpressionObject as LazyMetricExpressionObject,
    LazyMappedObject as LazyMetricMappedObject,
    LazyObject as LazyMetricObject,
)
from .space.base import ChartedSpace as ChartedMetricSpace
from .space.base import Space as MetricSpace


MetricTensor = tuple[tuple[float, ...], ...]
RiemannianSpace = MetricSpace
ChartedRiemannianSpace = ChartedMetricSpace
RiemannianGeometricObject = MetricGeometricObject


from .space.euclidean import Space as EuclideanMetricSpace


EuclideanRiemannianSpace = EuclideanMetricSpace
EuclideanPlaneSpace = EuclideanMetricSpace


__all__ = [
    "MetricTensor",
    "MetricSpace",
    "ChartedMetricSpace",
    "MetricGeometricObject",
    "RiemannianSpace",
    "ChartedRiemannianSpace",
    "RiemannianGeometricObject",
    "LazyMetricObject",
    "LazyMetricExpressionObject",
    "LazyMetricMappedObject",
    "EuclideanMetricSpace",
    "EuclideanRiemannianSpace",
    "EuclideanPlaneSpace",
]
