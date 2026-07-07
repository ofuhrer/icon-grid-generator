from __future__ import annotations

from typing import Any
import math

import numpy as np

from grid_generator.diagnostics import check_grid


def assert_valid_grid_math(
    grid: Any,
    *,
    geometry: str,
    boundary: str,
) -> None:
    assert geometry in {"global", "periodic_planar", "open_planar", "regional"}
    assert boundary in {"closed", "open"}

    _assert_core_shapes(grid)
    _assert_finite_arrays(grid)
    _assert_topology_bounds(grid)
    _assert_edge_cell_reciprocity(grid)
    _assert_cell_edge_vertices_are_consistent(grid)
    _assert_metrics_are_physical(grid)
    _assert_orientation_fields_are_valid(grid)

    check = check_grid(grid)
    assert check.ok, check.errors

    boundary_edges = int(np.count_nonzero(grid.edge_cells[:, 1] < 0))
    if boundary == "closed":
        assert boundary_edges == 0
    else:
        assert boundary_edges > 0

    if geometry == "global":
        _assert_global_spherical_invariants(grid)
    elif geometry == "periodic_planar":
        assert grid.metadata["grid_geometry"] == 2
        assert grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] == 0
    elif geometry == "open_planar":
        assert grid.metadata["grid_geometry"] == 2
        assert grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] in {0, 1}
    else:
        assert grid.metadata["grid_geometry"] == 3
        assert np.all(grid.refinement["parent_cell_index"] > 0)
        assert np.all(grid.refinement["parent_edge_index"] > 0)
        assert np.all(grid.refinement["parent_vertex_index"] > 0)


def assert_same_topology(left: Any, right: Any) -> None:
    assert left.dims == right.dims
    for name in ("cells", "edges", "cell_edges", "edge_cells"):
        assert np.array_equal(getattr(left, name), getattr(right, name))
    assert np.array_equal(
        left.icon_connectivity["orientation_of_normal"],
        right.icon_connectivity["orientation_of_normal"],
    )


def assert_netcdf_grid_contract(dataset: Any, grid: Any) -> None:
    assert dataset.dimensions["cell"].size == grid.dims["cell"]
    assert dataset.dimensions["edge"].size == grid.dims["edge"]
    assert dataset.dimensions["vertex"].size == grid.dims["vertex"]
    for name in (
        "vertex_of_cell",
        "edge_of_cell",
        "neighbor_cell_index",
        "orientation_of_normal",
        "edge_vertices",
        "adjacent_cell_of_edge",
        "edge_system_orientation",
        "cell_area",
        "edge_length",
        "dual_area",
    ):
        assert name in dataset.variables

    for name in ("cell_area", "edge_length", "dual_area"):
        values = dataset.variables[name][:]
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)

    for name in ("vertex_of_cell", "edge_of_cell", "edge_vertices"):
        values = dataset.variables[name][:]
        assert np.min(values) >= 1


def _assert_core_shapes(grid: Any) -> None:
    assert grid.vertices.shape == (grid.dims["vertex"], 3)
    assert grid.cells.shape == (grid.dims["cell"], 3)
    assert grid.edges.shape == (grid.dims["edge"], 2)
    assert grid.cell_edges.shape == (grid.dims["cell"], 3)
    assert grid.edge_cells.shape == (grid.dims["edge"], 2)
    assert grid.lon.shape == (grid.dims["cell"],)
    assert grid.lat.shape == (grid.dims["cell"],)
    assert grid.vertex_lon.shape == (grid.dims["vertex"],)
    assert grid.vertex_lat.shape == (grid.dims["vertex"],)
    assert grid.edge_lon.shape == (grid.dims["edge"],)
    assert grid.edge_lat.shape == (grid.dims["edge"],)


def _assert_finite_arrays(grid: Any) -> None:
    for value in (
        grid.vertices,
        grid.cells,
        grid.edges,
        grid.cell_edges,
        grid.edge_cells,
        grid.lon,
        grid.lat,
        grid.vertex_lon,
        grid.vertex_lat,
        grid.edge_lon,
        grid.edge_lat,
        grid.cell_center_xyz,
        grid.edge_center_xyz,
    ):
        assert np.all(np.isfinite(value))
    for value in grid.geometry.values():
        assert np.all(np.isfinite(value))


def _assert_topology_bounds(grid: Any) -> None:
    assert np.all((0 <= grid.cells) & (grid.cells < grid.dims["vertex"]))
    assert np.all((0 <= grid.edges) & (grid.edges < grid.dims["vertex"]))
    assert np.all((0 <= grid.cell_edges) & (grid.cell_edges < grid.dims["edge"]))
    assert np.all((-1 <= grid.edge_cells) & (grid.edge_cells < grid.dims["cell"]))
    assert np.all(grid.edge_cells[:, 0] >= 0)
    active_edge_cells = grid.edge_cells[grid.edge_cells >= 0]
    assert active_edge_cells.size > 0


def _assert_edge_cell_reciprocity(grid: Any) -> None:
    for cell_index, edge_indices in enumerate(grid.cell_edges):
        for edge_index in edge_indices:
            assert cell_index in set(map(int, grid.edge_cells[int(edge_index)]))

    for edge_index, edge_cells in enumerate(grid.edge_cells):
        for cell_index in edge_cells:
            if cell_index >= 0:
                assert edge_index in set(map(int, grid.cell_edges[int(cell_index)]))


def _assert_cell_edge_vertices_are_consistent(grid: Any) -> None:
    for cell_index, vertex_indices in enumerate(grid.cells):
        cell_vertices = set(map(int, vertex_indices))
        for edge_index in grid.cell_edges[cell_index]:
            edge_vertices = set(map(int, grid.edges[int(edge_index)]))
            assert edge_vertices <= cell_vertices


def _assert_metrics_are_physical(grid: Any) -> None:
    assert np.all(grid.geometry["cell_area"] > 0.0)
    assert np.all(grid.geometry["edge_length"] > 0.0)
    assert np.all(grid.geometry["dual_area"] > 0.0)
    assert np.all(grid.geometry["dual_edge_length"] > 0.0)
    assert np.all(grid.geometry["edgequad_area"] >= 0.0)
    assert np.allclose(
        np.linalg.norm(grid.geometry["edge_primal_normal_cartesian"], axis=1),
        1.0,
    )
    assert np.allclose(
        np.linalg.norm(grid.geometry["edge_dual_normal_cartesian"], axis=1),
        1.0,
    )


def _assert_orientation_fields_are_valid(grid: Any) -> None:
    orientation = grid.icon_connectivity["orientation_of_normal"]
    assert orientation.shape == (grid.dims["cell"], 3)
    assert set(np.unique(orientation)) <= {-1, 1}
    edge_system_orientation = grid.geometry["edge_system_orientation"]
    assert edge_system_orientation.shape == (grid.dims["edge"],)
    assert set(np.unique(edge_system_orientation)) <= {-1, 1}


def _assert_global_spherical_invariants(grid: Any) -> None:
    assert grid.metadata["grid_geometry"] == 1
    assert grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] == 2
    assert np.allclose(np.linalg.norm(grid.vertices, axis=1), grid.options.radius)

    total_area = float(np.sum(grid.geometry["cell_area"]))
    sphere_radius = float(grid.metadata["sphere_radius"])
    assert math.isclose(total_area, 4.0 * math.pi * sphere_radius**2, rel_tol=2e-12)

    vertex_degree = np.bincount(grid.edges.ravel(), minlength=grid.dims["vertex"])
    assert set(np.unique(vertex_degree)) <= {5, 6}
    assert int(np.count_nonzero(vertex_degree == 5)) == 12
