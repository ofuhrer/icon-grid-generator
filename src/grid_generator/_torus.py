"""Compatibility wrappers for planar doubly periodic triangular grids."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import GeometryData, MetricsData, RefinementData, TopologyData


class PlanarTorusGeometry:
    """Build a torus through the shared skew-periodic planar pipeline."""

    def build(self, spec: Any, options: Any) -> GeometryData:
        from ._planar import PlanarTriangularGeometry

        return PlanarTriangularGeometry().build(spec, options)


class PeriodicTopologyBuilder:
    """Build torus topology through the shared periodic planar pipeline."""

    def build(self, spec: Any, options: Any, geometry: GeometryData) -> TopologyData:
        from ._planar import PlanarTriangularTopologyBuilder

        del options
        return PlanarTriangularTopologyBuilder().build(spec, geometry)


class PlanarTorusMetricsBuilder:
    """Build torus metrics through the shared periodic planar pipeline."""

    def build(
        self,
        spec: Any,
        geometry: GeometryData,
        topology: TopologyData,
    ) -> MetricsData:
        from ._planar import PlanarTriangularMetricsBuilder

        return PlanarTriangularMetricsBuilder().build(spec, geometry, topology)


class TorusRefinementBuilder:
    """Build standalone torus refinement fields through the shared pipeline."""

    def build(
        self,
        geometry: GeometryData,
        topology: TopologyData,
    ) -> RefinementData:
        from ._planar import PlanarRefinementBuilder

        return PlanarRefinementBuilder().build(geometry, topology)


def _scale_to_degrees(
    values: np.ndarray,
    period: float,
    start: float,
    end: float,
) -> np.ndarray:
    return start + (np.asarray(values, dtype=np.float64) % period) / period * (
        end - start
    )
