"""Topology construction for closed triangular spherical grids."""

from __future__ import annotations

from typing import Any

from ._types import GeometryData, TopologyData


class GlobalTopologyBuilder:
    """Build global edge and adjacency tables from triangular cells."""

    def build(self, spec: Any, options: Any, geometry: GeometryData) -> TopologyData:
        from . import grid_generator as gg

        edges, cell_edges, edge_cells = gg._build_edges(geometry.cells)
        if edges.shape[0] != spec.expected_edges:
            raise RuntimeError(
                f"generated {edges.shape[0]} edges, expected {spec.expected_edges}"
            )

        edge_center_xyz = gg._edge_centers(geometry.vertices, edges, options.radius)
        edge_lon, edge_lat = gg._lon_lat(edge_center_xyz)
        icon_connectivity = gg._icon_connectivity(
            geometry.vertices,
            geometry.cells,
            geometry.cell_center_xyz,
            edges,
            cell_edges,
            edge_cells,
        )
        return TopologyData(
            edges=edges,
            cell_edges=cell_edges,
            edge_cells=edge_cells,
            edge_center_xyz=edge_center_xyz,
            edge_lon=edge_lon,
            edge_lat=edge_lat,
            icon_connectivity=icon_connectivity,
            connectivity=gg._public_connectivity(
                geometry.cells,
                edges,
                edge_cells,
                icon_connectivity,
            ),
            neighbor_tables=gg._neighbor_tables(
                geometry.cells,
                edges,
                edge_cells,
                icon_connectivity,
            ),
        )
