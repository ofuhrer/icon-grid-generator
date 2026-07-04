"""Metric and orientation field computation."""

from __future__ import annotations

from typing import Any

from ._types import GeometryData, MetricsData, TopologyData


class SphericalMetricsBuilder:
    """Compute ICON metric fields for a spherical triangular grid."""

    def build(
        self,
        options: Any,
        geometry: GeometryData,
        topology: TopologyData,
    ) -> MetricsData:
        from . import grid_generator as gg

        return MetricsData(
            fields=gg._geometry_fields(
                geometry.vertices,
                geometry.cells,
                geometry.cell_center_xyz,
                topology.edges,
                topology.edge_cells,
                topology.edge_center_xyz,
                topology.icon_connectivity,
                options.sphere_radius,
            )
        )
