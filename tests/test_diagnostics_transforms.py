from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from grid_generator import ChannelGridSpec, TorusGridSpec, generate_grid
from grid_generator.diagnostics import (
    cell_divergence,
    cell_vorticity_fnorm,
    check_grid,
    grid_statistics,
    triangle_properties,
)
from grid_generator.transforms import (
    DiffusionOptions,
    OptimizationOptions,
    diffuse_grid,
    optimize_grid,
)


def test_geometry_optimization_and_diffusion_preserve_topology_and_boundaries():
    grid = generate_grid(ChannelGridSpec(nx=4, ny=3, edge_length=1.0))
    boundary_vertices = np.unique(grid.edges[grid.edge_cells[:, 1] < 0])
    optimized = optimize_grid(grid, OptimizationOptions(iterations=2, relaxation=0.2))
    diffused = diffuse_grid(grid, DiffusionOptions(iterations=2, diffusion_constant=0.05))

    for transformed in [optimized, diffused]:
        assert transformed is not grid
        assert np.array_equal(transformed.cells, grid.cells)
        assert np.array_equal(transformed.edges, grid.edges)
        assert np.array_equal(transformed.edge_cells, grid.edge_cells)
        assert np.allclose(transformed.vertices[boundary_vertices], grid.vertices[boundary_vertices])
        assert np.all(np.isfinite(transformed.geometry["cell_area"]))
        assert np.all(transformed.geometry["cell_area"] > 0.0)


def test_geometry_postprocessing_rejects_invalid_option_objects():
    grid = generate_grid(ChannelGridSpec(nx=3, ny=2, edge_length=1.0))

    with pytest.raises(TypeError, match="OptimizationOptions"):
        optimize_grid(grid, options=0)
    with pytest.raises(TypeError, match="DiffusionOptions"):
        diffuse_grid(grid, options=0)


def test_diagnostics_and_postprocessing_core_operators():
    grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=2.0))
    check = check_grid(grid)
    stats = grid_statistics(grid)
    props = triangle_properties(grid)

    assert check.ok
    assert not check.errors
    assert stats.cells == grid.dims["cell"]
    assert stats.boundary_edges == 0
    assert props.area.shape == (grid.dims["cell"],)
    assert props.edge_lengths.shape == (grid.dims["cell"], 3)
    assert np.all(props.min_angle_degrees > 0.0)
    assert np.allclose(cell_divergence(grid, np.zeros(grid.dims["edge"])), 0.0)

    vertex_vorticity = np.arange(grid.dims["vertex"], dtype=np.float64)
    fnorm = cell_vorticity_fnorm(grid, vertex_vorticity, coriolis=2.0)
    assert np.allclose(fnorm, vertex_vorticity[grid.cells].mean(axis=1) / 2.0)

    coriolis = np.linspace(1.0, 2.0, grid.dims["cell"])
    fnorm_by_cell = cell_vorticity_fnorm(grid, vertex_vorticity, coriolis=coriolis)
    assert np.allclose(fnorm_by_cell, vertex_vorticity[grid.cells].mean(axis=1) / coriolis)


def test_diagnostics_reject_invalid_field_shapes_and_values():
    grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=2.0))

    with pytest.raises(ValueError, match="edge_flux"):
        cell_divergence(grid, np.zeros(grid.dims["edge"] + 1))
    with pytest.raises(ValueError, match="vertex_vorticity"):
        cell_vorticity_fnorm(grid, np.zeros(grid.dims["vertex"] + 1), coriolis=1.0)
    with pytest.raises(ValueError, match="non-zero"):
        cell_vorticity_fnorm(grid, np.zeros(grid.dims["vertex"]), coriolis=0.0)
    with pytest.raises(ValueError, match="shape"):
        cell_vorticity_fnorm(
            grid,
            np.zeros(grid.dims["vertex"]),
            coriolis=np.ones(grid.dims["cell"] + 1),
        )
    with pytest.raises(ValueError, match="non-zero"):
        cell_vorticity_fnorm(
            grid,
            np.zeros(grid.dims["vertex"]),
            coriolis=np.zeros(grid.dims["cell"]),
        )


def test_check_grid_reports_multiple_structural_errors():
    grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=2.0))
    vertices = grid.vertices.copy()
    vertices[0, 0] = np.nan
    cells = grid.cells.copy()
    cells[0, 0] = grid.dims["vertex"]
    edges = grid.edges.copy()
    edges[0, 0] = grid.dims["vertex"]
    cell_edges = grid.cell_edges.copy()
    cell_edges[0, 0] = grid.dims["edge"]
    edge_cells = grid.edge_cells.copy()
    edge_cells[0, 0] = grid.dims["cell"]

    check = check_grid(
        replace(
            grid,
            vertices=vertices,
            cells=cells,
            edges=edges,
            cell_edges=cell_edges,
            edge_cells=edge_cells,
        )
    )
    shape_check = check_grid(replace(grid, cell_edges=grid.cell_edges[:1]))

    assert not check.ok
    assert "vertices contains non-finite values" in check.errors
    assert "cells contain out-of-range vertex indices" in check.errors
    assert "edges contain out-of-range vertex indices" in check.errors
    assert "cell_edges contain out-of-range edge indices" in check.errors
    assert "edge_cells contain out-of-range cell indices" in check.errors
    assert not shape_check.ok
    assert any("cell_edges has shape" in error for error in shape_check.errors)


def test_check_grid_reports_reversed_duplicate_edges():
    grid = generate_grid(ChannelGridSpec(nx=3, ny=2, edge_length=1.0))
    edges = grid.edges.copy()
    edges[1] = edges[0][::-1]
    broken = replace(grid, edges=edges)

    check = check_grid(broken)

    assert not check.ok
    assert "edges contain duplicate vertex pairs" in check.errors
