"""Spherical icosahedral geometry construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._types import BisectionProvenance, GeometryData


class SphericalIcosahedralGeometry:
    """Build global triangular ICON R<n>B<k> geometry on a sphere."""

    def build(self, spec: Any, options: Any) -> GeometryData:
        from . import grid_generator as gg

        base_vertices, faces = gg._icosahedron()
        vertices = base_vertices
        cells = np.asarray(
            [gg._orient_cell(tuple(face), vertices) for face in faces],
            dtype=np.int32,
        )
        if spec.root > 1:
            vertices, cells = gg._refine_triangles(vertices, cells, spec.root)
        bisection_provenance: BisectionProvenance | None = None
        for _ in range(spec.bisections):
            vertices, cells, bisection_provenance = (
                gg._refine_triangles_bisection_with_provenance(vertices, cells)
            )

        vertices = gg._rotate_points(
            vertices,
            options.rotation_axis,
            options.rotation_angle_degrees,
        )
        vertices = vertices * options.radius
        gg._check_expected_counts(spec, vertices, cells)

        vertex_lon, vertex_lat = gg._lon_lat(vertices)
        cell_center_xyz = gg._cell_centers(vertices, cells, options.radius)
        lon, lat = gg._lon_lat(cell_center_xyz)
        return GeometryData(
            vertices=vertices,
            cells=cells,
            lon=lon,
            lat=lat,
            vertex_lon=vertex_lon,
            vertex_lat=vertex_lat,
            cell_center_xyz=cell_center_xyz,
            cell_vertex_lon=vertex_lon[cells],
            cell_vertex_lat=vertex_lat[cells],
            bisection_provenance=bisection_provenance,
        )
