"""Ordering helpers matching ICON grid-generator conventions where possible."""

from __future__ import annotations

from dataclasses import replace
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

    def __init__(self, context: Any | None = None) -> None:
        self.context = context

    def order_spherical_bisection(self, spec: Any, options: Any, geometry: GeometryData) -> GeometryData:
        if getattr(spec, "bisections", 0) == 0:
            return geometry
        if geometry.bisection_provenance is not None:
            return geometry

        from . import grid_generator as gg

        context = self.context if self.context is not None else gg._GlobalGenerationContext()
        parent_vertex_index, parent = gg._parent_vertex_indices_cached(
            spec,
            options,
            geometry.vertices,
            context,
        )
        parent_cell_index, parent_cell_type = gg._parent_cell_fields(
            geometry.cells,
            parent_vertex_index,
            parent,
            options.accelerator,
        )
        child_order = np.asarray(
            [CHILD_ORDER[int(child_type)] for child_type in parent_cell_type],
            dtype=np.int32,
        )
        permutation = np.lexsort((child_order, parent_cell_index))
        return _permute_cells(geometry, permutation)


def _permute_cells(geometry: GeometryData, permutation: np.ndarray) -> GeometryData:
    provenance = geometry.bisection_provenance
    if provenance is not None:
        provenance = replace(
            provenance,
            parent_cell_index=provenance.parent_cell_index[permutation],
            parent_cell_type=provenance.parent_cell_type[permutation],
        )
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
        bisection_provenance=provenance,
    )
