"""xarray conversion for complete in-memory ICON grid data."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_xarray_dataset(grid: Any) -> Any:
    """Return an xarray Dataset containing topology, metrics, and provenance."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ModuleNotFoundError(
            "xarray conversion requires the xarray package; "
            "install icon-grid-generator[xarray]"
        ) from exc

    data_vars: dict[str, Any] = {
        "vertices": (("vertex", "xyz"), grid.vertices),
        "cells": (
            ("cell", "cell_vertex"),
            grid.cells,
            {"start_index": 0},
        ),
        "lon": (("cell",), grid.lon, {"units": "degrees_east"}),
        "lat": (("cell",), grid.lat, {"units": "degrees_north"}),
        "vertex_lon": (("vertex",), grid.vertex_lon, {"units": "degrees_east"}),
        "vertex_lat": (("vertex",), grid.vertex_lat, {"units": "degrees_north"}),
        "cell_center_xyz": (("cell", "xyz"), grid.cell_center_xyz),
        "cell_vertex_lon": (
            ("cell", "cell_vertex"),
            grid.cell_vertex_lon,
            {"units": "degrees_east"},
        ),
        "cell_vertex_lat": (
            ("cell", "cell_vertex"),
            grid.cell_vertex_lat,
            {"units": "degrees_north"},
        ),
        "edges": (("edge", "edge_vertex"), grid.edges, {"start_index": 0}),
        "cell_edges": (
            ("cell", "cell_vertex"),
            grid.cell_edges,
            {"start_index": 0},
        ),
        "edge_cells": (
            ("edge", "edge_cell"),
            grid.edge_cells,
            {"start_index": 0, "missing_value": -1},
        ),
        "edge_center_xyz": (("edge", "xyz"), grid.edge_center_xyz),
        "edge_lon": (("edge",), grid.edge_lon, {"units": "degrees_east"}),
        "edge_lat": (("edge",), grid.edge_lat, {"units": "degrees_north"}),
    }

    _add_named_arrays(data_vars, grid.geometry, _GEOMETRY_DIMS, _GEOMETRY_ATTRS)
    _add_named_arrays(
        data_vars,
        grid.refinement,
        _REFINEMENT_DIMS,
        _REFINEMENT_ATTRS,
    )
    _add_named_arrays(
        data_vars,
        grid.connectivity,
        _CONNECTIVITY_DIMS,
        _CONNECTIVITY_ATTRS,
    )

    coords = {
        "xyz": np.asarray(["x", "y", "z"]),
        "cell_vertex": np.arange(3, dtype=np.int32),
        "edge_vertex": np.arange(2, dtype=np.int32),
        "edge_cell": np.arange(2, dtype=np.int32),
        "vertex_neighbor": np.arange(6, dtype=np.int32),
        "max_chdom": np.arange(1, dtype=np.int32),
        "cell_grf": np.arange(14, dtype=np.int32),
        "edge_grf": np.arange(24, dtype=np.int32),
        "vert_grf": np.arange(13, dtype=np.int32),
    }
    attrs = {
        "name": grid.name,
        "root": getattr(grid.spec, "root", 0),
        "bisections": getattr(grid.spec, "bisections", 0),
        "frequency": getattr(grid.spec, "frequency", 0),
        "radius": grid.options.radius,
        "sphere_radius": grid.options.sphere_radius,
        **grid.metadata,
    }
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _add_named_arrays(
    data_vars: dict[str, Any],
    arrays: dict[str, np.ndarray],
    dimensions: dict[str, tuple[str, ...]],
    attributes: dict[str, dict[str, Any]] | None = None,
) -> None:
    for name, values in arrays.items():
        dims = dimensions.get(name)
        if dims is None:
            continue
        attrs = {} if attributes is None else attributes.get(name, {})
        data_vars[name] = (dims, values, attrs)


_GEOMETRY_DIMS = {
    "cell_area": ("cell",),
    "dual_area": ("vertex",),
    "edge_length": ("edge",),
    "dual_edge_length": ("edge",),
    "edge_cell_distance": ("edge", "edge_cell"),
    "edge_vert_distance": ("edge", "edge_vertex"),
    "orientation_of_normal": ("cell", "cell_vertex"),
    "edge_system_orientation": ("edge",),
    "edge_orientation": ("vertex", "vertex_neighbor"),
    "edgequad_area": ("edge",),
    "edge_primal_normal_cartesian": ("edge", "xyz"),
    "edge_dual_normal_cartesian": ("edge", "xyz"),
    "zonal_normal_primal_edge": ("edge",),
    "meridional_normal_primal_edge": ("edge",),
    "zonal_normal_dual_edge": ("edge",),
    "meridional_normal_dual_edge": ("edge",),
}

_GEOMETRY_ATTRS = {
    "cell_area": {"units": "m2"},
    "dual_area": {"units": "m2"},
    "edge_length": {"units": "m"},
    "dual_edge_length": {"units": "m"},
    "edge_cell_distance": {"units": "m"},
    "edge_vert_distance": {"units": "m"},
    "edgequad_area": {"units": "m2"},
}

_REFINEMENT_DIMS = {
    "refin_c_ctrl": ("cell",),
    "refin_e_ctrl": ("edge",),
    "refin_v_ctrl": ("vertex",),
    "start_idx_c": ("max_chdom", "cell_grf"),
    "end_idx_c": ("max_chdom", "cell_grf"),
    "start_idx_e": ("max_chdom", "edge_grf"),
    "end_idx_e": ("max_chdom", "edge_grf"),
    "start_idx_v": ("max_chdom", "vert_grf"),
    "end_idx_v": ("max_chdom", "vert_grf"),
    "parent_cell_index": ("cell",),
    "parent_cell_type": ("cell",),
    "edge_parent_type": ("edge",),
    "parent_edge_index": ("edge",),
    "parent_vertex_index": ("vertex",),
    "smooth_c_ctrl": ("cell",),
}

_REFINEMENT_ATTRS = {
    "parent_cell_index": {"start_index": 1, "missing_value": 0},
    "parent_edge_index": {"start_index": 1, "missing_value": 0},
    "parent_vertex_index": {"start_index": 1, "missing_value": 0},
}

_CONNECTIVITY_DIMS = {
    "edge_of_cell": ("cell", "cell_vertex"),
    "vertex_of_cell": ("cell", "cell_vertex"),
    "neighbor_cell_index": ("cell", "cell_vertex"),
    "adjacent_cell_of_edge": ("edge", "edge_cell"),
    "edge_vertices": ("edge", "edge_vertex"),
    "cells_of_vertex": ("vertex", "vertex_neighbor"),
    "edges_of_vertex": ("vertex", "vertex_neighbor"),
    "vertices_of_vertex": ("vertex", "vertex_neighbor"),
}

_CONNECTIVITY_ATTRS = {
    name: {"start_index": 0, "missing_value": -1}
    for name in _CONNECTIVITY_DIMS
}
