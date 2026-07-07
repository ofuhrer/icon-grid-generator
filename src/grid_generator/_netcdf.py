"""ICON NetCDF field assembly and writing."""

from __future__ import annotations

from datetime import datetime
import getpass
from pathlib import Path
import platform
from typing import Any

import numpy as np

IconNetcdfField = tuple[str, tuple[str, ...], Any, dict[str, Any]]
CELL_COORD_ATTRS = {
    "coordinates": "clon clat",
    "grid_type": "unstructured",
    "number_of_grid_in_reference": 1,
}
EDGE_COORD_ATTRS = {"coordinates": "elon elat"}
VERTEX_COORD_ATTRS = {"coordinates": "vlon vlat"}
ICON_VARIABLE_ATTRS: dict[str, dict[str, Any]] = {
    "clon": {
        "bounds": "clon_vertices",
        "long_name": "center longitude",
        "standard_name": "grid_longitude",
    },
    "clat": {
        "bounds": "clat_vertices",
        "long_name": "center latitude",
        "standard_name": "grid_latitude",
    },
    "vlon": {"long_name": "vertex longitude", "standard_name": "grid_longitude"},
    "vlat": {"long_name": "vertex latitude", "standard_name": "grid_latitude"},
    "elon": {
        "bounds": "elon_vertices",
        "long_name": "edge midpoint longitude",
        "standard_name": "grid_longitude",
    },
    "elat": {
        "bounds": "elat_vertices",
        "long_name": "edge midpoint latitude",
        "standard_name": "grid_latitude",
    },
    "lon_cell_centre": {**CELL_COORD_ATTRS, "long_name": "longitude of cell centre"},
    "lat_cell_centre": {**CELL_COORD_ATTRS, "long_name": "latitude of cell centre"},
    "longitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "longitude of vertices"},
    "latitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "latitude of vertices"},
    "lon_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "longitudes of edge midpoints"},
    "lat_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "latitudes of edge midpoints"},
    "edge_of_cell": {"long_name": "edges of each cell"},
    "vertex_of_cell": {"long_name": "vertices of each cell"},
    "neighbor_cell_index": {"long_name": "cell neighbor index"},
    "adjacent_cell_of_edge": {"long_name": "cells adjacent to each edge"},
    "edge_vertices": {"long_name": "vertices at the end of each edge"},
    "cells_of_vertex": {"long_name": "cells around each vertex"},
    "edges_of_vertex": {"long_name": "edges around each vertex"},
    "vertices_of_vertex": {"long_name": "vertices around each vertex"},
    "cell_area": {
        **CELL_COORD_ATTRS,
        "long_name": "area of grid cell",
        "standard_name": "area",
    },
    "dual_area": {
        **VERTEX_COORD_ATTRS,
        "long_name": "areas of dual hexagonal/pentagonal cells",
        "standard_name": "area",
    },
    "cell_area_p": {**CELL_COORD_ATTRS, "long_name": "area of grid cell"},
    "dual_area_p": {"long_name": "areas of dual hexagonal/pentagonal cells"},
    "edge_length": {**EDGE_COORD_ATTRS, "long_name": "lengths of edges of triangular cells"},
    "dual_edge_length": {
        **EDGE_COORD_ATTRS,
        "long_name": "lengths of dual edges (distances between triangular cell circumcenters)",
    },
    "edge_cell_distance": {
        "long_name": "distances between edge midpoint and adjacent triangle midpoints",
    },
    "edge_vert_distance": {
        "long_name": "distances between edge midpoint and vertices of that edge",
    },
    "edgequad_area": {
        **EDGE_COORD_ATTRS,
        "long_name": "area around the edge formed by the two adjacent triangles",
    },
    "orientation_of_normal": {"long_name": "orientations of normals to triangular cell edges"},
    "edge_system_orientation": {**EDGE_COORD_ATTRS, "long_name": "edge system orientation"},
    "edge_orientation": {"long_name": "edge orientation"},
    "refin_c_ctrl": {"long_name": "refinement control flag for cells"},
    "refin_e_ctrl": {"long_name": "refinement control flag for edges"},
    "refin_v_ctrl": {"long_name": "refinement control flag for vertices"},
    "start_idx_c": {"long_name": "list of start indices for each refinement control level for cells"},
    "end_idx_c": {"long_name": "list of end indices for each refinement control level for cells"},
    "start_idx_e": {"long_name": "list of start indices for each refinement control level for edges"},
    "end_idx_e": {"long_name": "list of end indices for each refinement control level for edges"},
    "start_idx_v": {"long_name": "list of start indices for each refinement control level for vertices"},
    "end_idx_v": {"long_name": "list of end indices for each refinement control level for vertices"},
    "cell_elevation": {**CELL_COORD_ATTRS, "long_name": "elevation at the cell centers"},
    "edge_elevation": {**EDGE_COORD_ATTRS, "long_name": "elevation at the edge centers"},
    "cell_sea_land_mask": {
        **CELL_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "edge_sea_land_mask": {
        **EDGE_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "cartesian_x_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate x on unit sp",
    },
    "cartesian_y_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate y on unit sp",
    },
    "cartesian_z_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate z on unit sp",
    },
    "cell_circumcenter_cartesian_x": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate x",
    },
    "cell_circumcenter_cartesian_y": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate y",
    },
    "cell_circumcenter_cartesian_z": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate z",
    },
    "edge_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate x on unit sphere",
    },
    "edge_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate y on unit sphere",
    },
    "edge_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate z on unit sphere",
    },
    "phys_cell_id": {**CELL_COORD_ATTRS, "long_name": "physical domain ID of cell"},
    "phys_edge_id": {**EDGE_COORD_ATTRS, "long_name": "physical domain ID of edge"},
    "cell_index": {"long_name": "cell index"},
    "edge_index": {"long_name": "edge index"},
    "vertex_index": {"long_name": "vertices index"},
    "edge_dual_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate x on unit sphere",
    },
    "edge_dual_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate y on unit sphere",
    },
    "edge_dual_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate z on unit sphere",
    },
    "edge_primal_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate x",
    },
    "edge_primal_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate y",
    },
    "edge_primal_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate z",
    },
    "edge_dual_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate x",
    },
    "edge_dual_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate y",
    },
    "edge_dual_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate z",
    },
    "zonal_normal_primal_edge": {"long_name": "zonal component of normal to primal edge"},
    "meridional_normal_primal_edge": {
        "long_name": "meridional component of normal to primal edge",
    },
    "zonal_normal_dual_edge": {"long_name": "zonal component of normal to dual edge"},
    "meridional_normal_dual_edge": {
        "long_name": "meridional component of normal to dual edge",
    },
    "parent_cell_index": {**CELL_COORD_ATTRS, "long_name": "parent cell index"},
    "parent_cell_type": {"long_name": "parent cell type"},
    "edge_parent_type": {"long_name": "edge parent type"},
    "parent_edge_index": {"long_name": "parent edge index"},
    "parent_vertex_index": {"long_name": "parent vertex index"},
    "child_cell_index": {"long_name": "child cell index"},
    "child_cell_id": {"long_name": "domain ID of child cell"},
    "child_edge_index": {"long_name": "child edge index"},
    "child_edge_id": {"long_name": "domain ID of child edge"},
}



def write_icon_grid(
    grid: Any,
    path: str | Path,
    *,
    sphere_radius: float | None = None,
) -> Path:
    """Write a compact ICON-style NetCDF grid file."""
    _require_complete_icon_grid(grid)
    if sphere_radius is None:
        sphere_radius = grid.options.sphere_radius
    if not np.isclose(sphere_radius, grid.options.sphere_radius):
        raise ValueError(
            "sphere_radius must match the value used by generate_grid(); "
            "pass options={'sphere_radius': ...} when generating the grid"
        )

    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ModuleNotFoundError("NetCDF export requires the netCDF4 package") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        _write_icon_dimensions(dataset, grid)
        _write_icon_attributes(dataset, grid, path)
        for name, dims, data, attrs in _icon_fields(grid):
            variable = dataset.createVariable(name, np.asarray(data).dtype, dims)
            variable[:] = data
            for attr_name, attr_value in attrs.items():
                variable.setncattr(attr_name, attr_value)

    return path


def _require_complete_icon_grid(grid: Any) -> None:
    for name, fields in {
        "icon_connectivity": grid.icon_connectivity,
        "geometry": grid.geometry,
        "refinement": grid.refinement,
    }.items():
        if not fields:
            raise ValueError(f"ICON NetCDF export requires populated {name}")


def _write_icon_dimensions(dataset: Any, grid: Any) -> None:
    dataset.createDimension("cell", grid.dims["cell"])
    dataset.createDimension("vertex", grid.dims["vertex"])
    dataset.createDimension("edge", grid.dims["edge"])
    for name, size in _gg().FIXED_DIMS.items():
        dataset.createDimension(name, size)


def _write_icon_attributes(dataset: Any, grid: Any, path: Path) -> None:
    external_attrs = {
        "revision": "pure-python",
        "history": f"grid.to_netcdf {path}",
        "date": datetime.now().strftime("%Y%m%d at %H%M%S"),
        "user_name": getpass.getuser(),
        "os_name": platform.platform(),
        "grid_ID": 1,
        "parent_grid_ID": 0,
        "no_of_subgrids": 1,
        "start_subgrid_id": 0,
        "max_childdom": 1,
        "boundary_depth_index": 0,
        "rotation_vector": np.zeros(3, dtype=np.float64),
        "domain_length": grid.metadata.get(
            "domain_length",
            2.0 * np.pi * grid.options.sphere_radius,
        ),
        "domain_height": grid.metadata.get(
            "domain_height",
            2.0 * np.pi * grid.options.sphere_radius,
        ),
        "domain_cartesian_center": np.zeros(3, dtype=np.float64),
    }
    attrs = {
        "title": f"Pure Python ICON grid {grid.name}",
        "institution": "grid_generator",
        "source": "grid_generator Python ICON grid generator",
        "ICON_grid_file_uri": str(path),
        **external_attrs,
        **grid.metadata,
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _icon_fields(grid: Any) -> list[IconNetcdfField]:
    fields = (
        _coordinate_fields(grid)
        + _connectivity_fields(grid)
        + _metric_fields(grid)
        + _refinement_fields_for_netcdf(grid)
        + _static_surface_fields(grid)
        + _cartesian_fields(grid)
        + _normal_vector_fields(grid)
        + _hierarchy_fields(grid)
    )
    return [
        (name, dims, data, _with_icon_variable_attrs(name, attrs))
        for name, dims, data, attrs in fields
    ]


def _coordinate_fields(grid: Any) -> list[IconNetcdfField]:
    edge_bounds_lon, edge_bounds_lat = _edge_lon_lat_bounds(grid)
    return [
        ("clon", ("cell",), np.radians(grid.lon), {"units": "radian"}),
        ("clat", ("cell",), np.radians(grid.lat), {"units": "radian"}),
        ("clon_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lon), {"units": "radian"}),
        ("clat_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lat), {"units": "radian"}),
        ("vlon", ("vertex",), np.radians(grid.vertex_lon), {"units": "radian"}),
        ("vlat", ("vertex",), np.radians(grid.vertex_lat), {"units": "radian"}),
        ("elon", ("edge",), np.radians(grid.edge_lon), {"units": "radian"}),
        ("elat", ("edge",), np.radians(grid.edge_lat), {"units": "radian"}),
        ("elon_vertices", ("edge", "no"), edge_bounds_lon, {"units": "radian"}),
        ("elat_vertices", ("edge", "no"), edge_bounds_lat, {"units": "radian"}),
        ("lon_cell_centre", ("cell",), np.radians(grid.lon), {"units": "radian"}),
        ("lat_cell_centre", ("cell",), np.radians(grid.lat), {"units": "radian"}),
        ("longitude_vertices", ("vertex",), np.radians(grid.vertex_lon), {"units": "radian"}),
        ("latitude_vertices", ("vertex",), np.radians(grid.vertex_lat), {"units": "radian"}),
        ("lon_edge_centre", ("edge",), np.radians(grid.edge_lon), {"units": "radian"}),
        ("lat_edge_centre", ("edge",), np.radians(grid.edge_lat), {"units": "radian"}),
    ]


def _connectivity_fields(grid: Any) -> list[IconNetcdfField]:
    connectivity = grid.icon_connectivity
    return [
        ("edge_of_cell", ("nv", "cell"), connectivity["c2e"].T + 1, {}),
        ("vertex_of_cell", ("nv", "cell"), grid.cells.T + 1, {}),
        ("neighbor_cell_index", ("nv", "cell"), connectivity["c2c"].T + 1, {}),
        ("adjacent_cell_of_edge", ("nc", "edge"), grid.edge_cells.T + 1, {}),
        ("edge_vertices", ("nc", "edge"), grid.edges.T + 1, {}),
        ("cells_of_vertex", ("ne", "vertex"), connectivity["v2c"].T, {}),
        ("edges_of_vertex", ("ne", "vertex"), connectivity["v2e"].T, {}),
        ("vertices_of_vertex", ("ne", "vertex"), connectivity["v2v"].T, {}),
    ]


def _metric_fields(grid: Any) -> list[IconNetcdfField]:
    geometry = grid.geometry
    edgequad_normalizer = (
        1.0 if grid.metadata.get("grid_geometry") == 2 else grid.options.sphere_radius**2
    )
    return [
        ("cell_area", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("cell_area_p", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area_p", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("edge_length", ("edge",), geometry["edge_length"], {"units": "m"}),
        ("dual_edge_length", ("edge",), geometry["dual_edge_length"], {"units": "m"}),
        ("edge_cell_distance", ("nc", "edge"), geometry["edge_cell_distance"].T, {"units": "m"}),
        ("edge_vert_distance", ("nc", "edge"), geometry["edge_vert_distance"].T, {"units": "m"}),
        (
            "edgequad_area",
            ("edge",),
            geometry["edgequad_area"] / edgequad_normalizer,
            {"units": "m2"},
        ),
        ("orientation_of_normal", ("nv", "cell"), geometry["orientation_of_normal"].T, {}),
        ("edge_system_orientation", ("edge",), geometry["edge_system_orientation"], {}),
        ("edge_orientation", ("ne", "vertex"), geometry["edge_orientation"].T, {}),
    ]


def _refinement_fields_for_netcdf(grid: Any) -> list[IconNetcdfField]:
    refinement = grid.refinement
    return [
        ("refin_c_ctrl", ("cell",), refinement["refin_c_ctrl"], {}),
        ("refin_e_ctrl", ("edge",), refinement["refin_e_ctrl"], {}),
        ("refin_v_ctrl", ("vertex",), refinement["refin_v_ctrl"], {}),
        ("start_idx_c", ("max_chdom", "cell_grf"), refinement["start_idx_c"], {}),
        ("end_idx_c", ("max_chdom", "cell_grf"), refinement["end_idx_c"], {}),
        ("start_idx_e", ("max_chdom", "edge_grf"), refinement["start_idx_e"], {}),
        ("end_idx_e", ("max_chdom", "edge_grf"), refinement["end_idx_e"], {}),
        ("start_idx_v", ("max_chdom", "vert_grf"), refinement["start_idx_v"], {}),
        ("end_idx_v", ("max_chdom", "vert_grf"), refinement["end_idx_v"], {}),
    ]


def _static_surface_fields(grid: Any) -> list[IconNetcdfField]:
    zeros_cell = np.zeros(grid.dims["cell"], dtype=np.float64)
    zeros_edge = np.zeros(grid.dims["edge"], dtype=np.float64)
    return [
        ("cell_elevation", ("cell",), zeros_cell, {"units": "m"}),
        ("edge_elevation", ("edge",), zeros_edge, {"units": "m"}),
        ("cell_sea_land_mask", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("edge_sea_land_mask", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
    ]


def _cartesian_fields(grid: Any) -> list[IconNetcdfField]:
    if grid.metadata.get("grid_geometry") == 2:
        unit_vertices = grid.vertices
        unit_centers = grid.cell_center_xyz
        unit_edge_centers = grid.edge_center_xyz
    else:
        unit_vertices = _gg()._normalize_rows(grid.vertices)
        unit_centers = _gg()._normalize_rows(grid.cell_center_xyz)
        unit_edge_centers = _gg()._normalize_rows(grid.edge_center_xyz)
    return [
        ("cartesian_x_vertices", ("vertex",), unit_vertices[:, 0], {"units": "meters"}),
        ("cartesian_y_vertices", ("vertex",), unit_vertices[:, 1], {"units": "meters"}),
        ("cartesian_z_vertices", ("vertex",), unit_vertices[:, 2], {"units": "meters"}),
        ("cell_circumcenter_cartesian_x", ("cell",), unit_centers[:, 0], {"units": "meters"}),
        ("cell_circumcenter_cartesian_y", ("cell",), unit_centers[:, 1], {"units": "meters"}),
        ("cell_circumcenter_cartesian_z", ("cell",), unit_centers[:, 2], {"units": "meters"}),
        ("edge_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], {"units": "meters"}),
        ("edge_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], {"units": "meters"}),
        ("edge_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], {"units": "meters"}),
        ("phys_cell_id", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("phys_edge_id", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("cell_index", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("edge_index", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("vertex_index", ("vertex",), np.arange(1, grid.dims["vertex"] + 1, dtype=np.int32), {}),
        ("edge_dual_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], {"units": "meters"}),
        ("edge_dual_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], {"units": "meters"}),
        ("edge_dual_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], {"units": "meters"}),
    ]


def _normal_vector_fields(grid: Any) -> list[IconNetcdfField]:
    geometry = grid.geometry
    return [
        (
            "edge_primal_normal_cartesian_x",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 0],
            {"units": "meters"},
        ),
        (
            "edge_primal_normal_cartesian_y",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 1],
            {"units": "meters"},
        ),
        (
            "edge_primal_normal_cartesian_z",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 2],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_x",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 0],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_y",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 1],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_z",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 2],
            {"units": "meters"},
        ),
        ("zonal_normal_primal_edge", ("edge",), geometry["zonal_normal_primal_edge"], {"units": "radian"}),
        (
            "meridional_normal_primal_edge",
            ("edge",),
            geometry["meridional_normal_primal_edge"],
            {"units": "radian"},
        ),
        ("zonal_normal_dual_edge", ("edge",), geometry["zonal_normal_dual_edge"], {"units": "radian"}),
        (
            "meridional_normal_dual_edge",
            ("edge",),
            geometry["meridional_normal_dual_edge"],
            {"units": "radian"},
        ),
    ]


def _hierarchy_fields(grid: Any) -> list[IconNetcdfField]:
    refinement = grid.refinement
    return [
        ("parent_cell_index", ("cell",), refinement["parent_cell_index"], {}),
        ("parent_cell_type", ("cell",), refinement["parent_cell_type"], {}),
        ("edge_parent_type", ("edge",), refinement["edge_parent_type"], {}),
        ("parent_edge_index", ("edge",), refinement["parent_edge_index"], {}),
        ("parent_vertex_index", ("vertex",), refinement["parent_vertex_index"], {}),
        ("child_cell_index", ("no", "cell"), np.zeros((4, grid.dims["cell"]), dtype=np.int32), {}),
        ("child_cell_id", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("child_edge_index", ("no", "edge"), np.zeros((4, grid.dims["edge"]), dtype=np.int32), {}),
        ("child_edge_id", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
    ]


def _with_icon_variable_attrs(name: str, attrs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(ICON_VARIABLE_ATTRS.get(name, {}))
    merged.update(attrs)
    return merged


def _edge_lon_lat_bounds(grid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ICON-style four-point edge bounds in radians.

    The upstream grid generator stores bounds for each edge as a quadrilateral:
    first edge vertex, second adjacent cell center, second edge vertex, first
    adjacent cell center.
    """
    edge_vertices = np.asarray(grid.edges, dtype=np.int32)
    edge_cells = np.asarray(grid.edge_cells, dtype=np.int32)
    lon = np.empty((grid.dims["edge"], 4), dtype=np.float64)
    lat = np.empty((grid.dims["edge"], 4), dtype=np.float64)

    lon[:, 0] = grid.vertex_lon[edge_vertices[:, 0]]
    lat[:, 0] = grid.vertex_lat[edge_vertices[:, 0]]
    second_cell = edge_cells[:, 1]
    second_cell_lon = np.where(second_cell >= 0, grid.lon[np.maximum(second_cell, 0)], grid.edge_lon)
    second_cell_lat = np.where(second_cell >= 0, grid.lat[np.maximum(second_cell, 0)], grid.edge_lat)
    lon[:, 1] = second_cell_lon
    lat[:, 1] = second_cell_lat
    lon[:, 2] = grid.vertex_lon[edge_vertices[:, 1]]
    lat[:, 2] = grid.vertex_lat[edge_vertices[:, 1]]
    first_cell = edge_cells[:, 0]
    first_cell_lon = np.where(first_cell >= 0, grid.lon[np.maximum(first_cell, 0)], grid.edge_lon)
    first_cell_lat = np.where(first_cell >= 0, grid.lat[np.maximum(first_cell, 0)], grid.edge_lat)
    lon[:, 3] = first_cell_lon
    lat[:, 3] = first_cell_lat

    pole_mask = np.isclose(np.abs(lat), 90.0)
    lon[pole_mask] = np.repeat(grid.edge_lon[:, np.newaxis], 4, axis=1)[pole_mask]
    return np.radians(lon), np.radians(lat)


def _gg() -> Any:
    from . import grid_generator as gg

    return gg
