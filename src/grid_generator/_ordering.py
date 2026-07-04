"""Ordering helpers matching ICON grid-generator conventions where possible."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import GeometryData


CHILD_ORDER = {
    200: 0,
    201: 1,
    202: 2,
    203: 3,
}


class IconOrderingBuilder:
    """Apply deterministic ICON child ordering."""

    def order_spherical_bisection(self, spec: Any, options: Any, geometry: GeometryData) -> GeometryData:
        if getattr(spec, "bisections", 0) == 0:
            return geometry

        from . import grid_generator as gg

        parent = gg.generate_grid(
            f"R{spec.root:02d}B{spec.bisections - 1:02d}",
            options=options,
        )
        parent_vertex_index = gg._parent_vertex_indices(geometry.vertices, parent)
        parent_cell_index, parent_cell_type = gg._parent_cell_fields(
            geometry.cells,
            parent_vertex_index,
            parent,
        )
        child_order = np.asarray(
            [CHILD_ORDER[int(child_type)] for child_type in parent_cell_type],
            dtype=np.int32,
        )
        permutation = np.lexsort((child_order, parent_cell_index))
        return _permute_cells(geometry, permutation)


def _permute_cells(geometry: GeometryData, permutation: np.ndarray) -> GeometryData:
    return GeometryData(
        vertices=geometry.vertices,
        cells=geometry.cells[permutation],
        lon=geometry.lon[permutation],
        lat=geometry.lat[permutation],
        vertex_lon=geometry.vertex_lon,
        vertex_lat=geometry.vertex_lat,
        cell_center_xyz=geometry.cell_center_xyz[permutation],
        cell_vertex_lon=geometry.cell_vertex_lon[permutation],
        cell_vertex_lat=geometry.cell_vertex_lat[permutation],
        source_cell_index=(
            None
            if geometry.source_cell_index is None
            else geometry.source_cell_index[permutation]
        ),
        source_vertex_index=geometry.source_vertex_index,
    )
