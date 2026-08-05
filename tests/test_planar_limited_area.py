from __future__ import annotations

import numpy as np
import pytest

from grid_generator import (
    ChannelGridSpec,
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    Region,
    TorusGridSpec,
    generate_grid,
)
from grid_generator.cutting import CutGridSpec, cut_grid
from grid_generator.diagnostics import cell_divergence
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec
from grid_generator.transforms import (
    DiffusionOptions,
    OptimizationOptions,
    diffuse_grid,
    optimize_grid,
)


def test_torus_grid_has_periodic_topology_and_planar_metrics():
    edge_length = 1000.0
    grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=edge_length))

    assert grid.name == "TORUS4x3"
    assert grid.dims == {"cell": 24, "vertex": 12, "edge": 36}
    assert grid.metadata["grid_geometry"] == 2
    assert grid.metadata["domain_length"] == pytest.approx(4000.0)
    assert grid.metadata["domain_height"] == pytest.approx(3.0 * np.sqrt(3.0) * 500.0)
    assert np.all(grid.edge_cells >= 0)
    assert np.all(grid.edge_cells[:, 0] != grid.edge_cells[:, 1])
    assert np.allclose(grid.geometry["edge_length"], edge_length)
    assert np.allclose(grid.geometry["cell_area"], np.sqrt(3.0) * 0.25 * edge_length**2)
    assert np.allclose(grid.geometry["dual_edge_length"], edge_length / np.sqrt(3.0))
    assert np.allclose(
        grid.geometry["edgequad_area"],
        0.5 * edge_length**2 / np.sqrt(3.0),
    )
    assert np.all(np.isfinite(grid.vertices))
    assert np.all(np.isfinite(grid.cell_center_xyz))
    assert np.all(np.isfinite(grid.edge_center_xyz))
    assert np.all(np.isfinite(grid.geometry["edge_primal_normal_cartesian"]))


def test_torus_coordinates_metrics_normals_and_transforms_share_periodic_geometry():
    grid = generate_grid(TorusGridSpec(nx=32, ny=16, edge_length=1.0))
    vectors = _torus_minimum_image_vectors(
        grid.vertices[grid.edges[:, 1]] - grid.vertices[grid.edges[:, 0]],
        grid.spec,
    )

    assert np.allclose(np.linalg.norm(vectors[:, :2], axis=1), 1.0)
    assert np.allclose(grid.geometry["edge_length"], np.linalg.norm(vectors, axis=1))
    assert np.allclose(
        cell_divergence(
            grid,
            grid.geometry["edge_primal_normal_cartesian"][:, 0],
        ),
        0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        cell_divergence(
            grid,
            grid.geometry["edge_primal_normal_cartesian"][:, 1],
        ),
        0.0,
        atol=1.0e-12,
    )

    transformed = (
        optimize_grid(grid, OptimizationOptions(iterations=1, relaxation=0.25)),
        diffuse_grid(grid, DiffusionOptions(iterations=1, diffusion_constant=0.1)),
    )
    for result in transformed:
        assert np.allclose(result.geometry["edge_length"], 1.0)
        assert np.allclose(result.geometry["cell_area"], np.sqrt(3.0) * 0.25)


def _torus_minimum_image_vectors(vectors, spec):
    result = np.empty_like(vectors, dtype=np.float64)
    y_shift = (
        0.5
        * spec.ny
        * spec.edge_length
        * getattr(spec, "stretch_x", 1.0)
    )
    for index, vector in enumerate(vectors):
        best = None
        best_norm = np.inf
        y_center = int(np.rint(vector[1] / spec.domain_height))
        for y_wrap in range(y_center - 1, y_center + 2):
            candidate = vector.copy()
            candidate[1] -= y_wrap * spec.domain_height
            candidate[0] -= y_wrap * y_shift
            candidate[0] -= np.rint(candidate[0] / spec.domain_length) * spec.domain_length
            norm = np.linalg.norm(candidate[:2])
            if norm < best_norm:
                best = candidate
                best_norm = norm
        result[index] = best
    return result


def _geometric_dual_edge_lengths(grid):
    expected = np.empty(grid.dims["edge"], dtype=np.float64)
    interior = grid.edge_cells[:, 1] >= 0
    adjacent = grid.edge_cells[interior]
    vectors = (
        grid.cell_center_xyz[adjacent[:, 1]]
        - grid.cell_center_xyz[adjacent[:, 0]]
    )
    if getattr(grid.spec, "periodic", False):
        vectors = _torus_minimum_image_vectors(vectors, grid.spec)
    elif getattr(grid.spec, "periodic_x", False):
        period = grid.spec.nx * grid.spec.edge_length
        vectors[:, 0] -= np.rint(vectors[:, 0] / period) * period
    expected[interior] = np.linalg.norm(vectors, axis=1)

    boundary = ~interior
    vectors = (
        grid.edge_center_xyz[boundary]
        - grid.cell_center_xyz[grid.edge_cells[boundary, 0]]
    )
    if getattr(grid.spec, "periodic_x", False):
        period = grid.spec.nx * grid.spec.edge_length
        vectors[:, 0] -= np.rint(vectors[:, 0] / period) * period
    expected[boundary] = 2.0 * np.linalg.norm(vectors, axis=1)
    return expected


@pytest.mark.parametrize(
    "spec",
    [
        StretchedTorusGridSpec(
            nx=6,
            ny=5,
            edge_length=2.0,
            stretch_x=1.5,
            stretch_y=0.75,
        ),
        StretchedTorusGridSpec(
            nx=6,
            ny=5,
            edge_length=2.0,
            stretch_x=1.5,
            stretch_y=1.0,
        ),
        ParallelogramGridSpec(
            nx=6,
            ny=5,
            edge_length=2.0,
            shear=0.8,
        ),
        RaggedOrthogonalGridSpec(
            nx=6,
            ny=5,
            dx=2.0,
            dy=1.3,
            raggedness=0.3,
        ),
    ],
)
def test_planar_dual_edge_lengths_follow_generated_cell_centers(spec):
    grid = generate_grid(spec)

    assert np.allclose(
        grid.geometry["dual_edge_length"],
        _geometric_dual_edge_lengths(grid),
    )
    assert np.all(grid.geometry["edgequad_area"] > 0.0)
    assert np.all(
        grid.geometry["edgequad_area"]
        <= 0.5
        * grid.geometry["edge_length"]
        * grid.geometry["dual_edge_length"]
        * (1.0 + 1.0e-12)
    )


@pytest.mark.parametrize(
    "spec",
    [
        ChannelGridSpec(nx=6, ny=5, edge_length=2.0),
        ParallelogramGridSpec(nx=6, ny=5, edge_length=2.0, shear=0.4),
    ],
)
def test_open_planar_dual_areas_partition_primal_area(spec):
    grid = generate_grid(spec)

    assert grid.geometry["dual_area"].sum() == pytest.approx(
        grid.geometry["cell_area"].sum()
    )


@pytest.mark.parametrize(
    "spec",
    [
        StretchedTorusGridSpec(nx=4, ny=3, edge_length=2.0, stretch_x=1.5, stretch_y=0.75),
        ChannelGridSpec(nx=4, ny=3, edge_length=2.0),
        ParallelogramGridSpec(nx=4, ny=3, edge_length=2.0, shear=0.25),
        RaggedOrthogonalGridSpec(nx=4, ny=3, dx=2.0, dy=1.5),
    ],
)
def test_planar_grid_variants_have_consistent_triangular_topology(spec):
    grid = generate_grid(spec)

    assert grid.dims["cell"] == spec.expected_cells
    assert grid.dims["vertex"] == spec.expected_vertices
    assert grid.dims["edge"] == spec.expected_edges
    assert grid.metadata["grid_geometry"] == 2
    assert grid.cells.shape == (spec.expected_cells, 3)
    assert grid.edges.shape == (spec.expected_edges, 2)
    assert np.all((0 <= grid.cells) & (grid.cells < grid.dims["vertex"]))
    assert np.all((0 <= grid.cell_edges) & (grid.cell_edges < grid.dims["edge"]))
    assert np.all(np.isfinite(grid.vertices))
    assert np.all(grid.geometry["cell_area"] > 0.0)
    assert np.all(grid.geometry["edge_length"] > 0.0)
    if getattr(spec, "periodic", False):
        assert np.all(grid.edge_cells >= 0)
        assert grid.metadata["periodic"] == 1
    else:
        assert np.any(grid.edge_cells[:, 1] < 0)
        assert grid.metadata["periodic"] == 0


def test_stretched_torus_rejects_degenerate_periodic_dimensions():
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        StretchedTorusGridSpec(nx=2, ny=3, edge_length=1.0)
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        StretchedTorusGridSpec(nx=3, ny=2, edge_length=1.0)


def test_limited_area_grid_is_compact_boundary_ordered_and_parent_linked():
    spec = LimitedAreaGridSpec(
        parent="R02B01",
        region=Region.lonlat_box(
            lon_min=-20.0,
            lon_max=20.0,
            lat_min=-20.0,
            lat_max=20.0,
        ),
        boundary_depth=1,
    )
    grid = generate_grid(spec, options={"max_cells": None})
    parent = generate_grid("R02B01", options={"max_cells": None})
    parent_cells = grid.refinement["parent_cell_index"] - 1

    assert grid.name == "LAM_R02B01"
    assert grid.metadata["grid_geometry"] == 3
    assert grid.metadata["parent_grid_name"] == "R02B01"
    assert grid.metadata["boundary_depth_index"] == 1
    assert grid.dims["cell"] > 0
    assert grid.dims["vertex"] == len(np.unique(grid.cells))
    assert np.all((0 <= grid.cells) & (grid.cells < grid.dims["vertex"]))
    assert np.all((0 <= grid.cell_edges) & (grid.cell_edges < grid.dims["edge"]))
    assert np.any(grid.edge_cells[:, 1] < 0)
    assert np.all(grid.edge_cells[:, 0] >= 0)
    assert np.all(parent_cells >= 0)
    assert np.all(parent_cells < parent.dims["cell"])
    assert np.all(grid.refinement["parent_edge_index"] > 0)
    assert np.all(grid.refinement["parent_vertex_index"] > 0)
    assert np.all(np.diff(grid.refinement["refin_c_ctrl"]) >= 0)
    assert np.min(grid.refinement["refin_c_ctrl"]) == 1
    assert np.all(np.isfinite(grid.geometry["cell_area"]))
    assert np.all(np.isfinite(grid.geometry["edge_length"]))


def test_limited_area_default_uses_optimized_global_parent():
    spec = LimitedAreaGridSpec(
        parent="R02B01",
        region=Region.lonlat_box(
            lon_min=-20.0,
            lon_max=20.0,
            lat_min=-20.0,
            lat_max=20.0,
        ),
    )

    grid = generate_grid(spec, spring_iterations=5)
    parent = generate_grid("R02B01", spring_iterations=5)
    parent_cells = grid.refinement["parent_cell_index"] - 1

    assert grid.options.optimize_global is True
    assert np.allclose(grid.cell_center_xyz, parent.cell_center_xyz[parent_cells])


def test_limited_area_can_use_raw_or_explicitly_optimized_global_parent():
    spec = LimitedAreaGridSpec(
        parent="R02B01",
        region=Region.lonlat_box(
            lon_min=-20.0,
            lon_max=20.0,
            lat_min=-20.0,
            lat_max=20.0,
        ),
    )

    raw_grid = generate_grid(spec, optimize_global=False)
    raw_parent = generate_grid("R02B01", optimize_global=False)
    optimized_grid = generate_grid(spec, optimize_global=True, spring_iterations=5)

    raw_parent_cells = raw_grid.refinement["parent_cell_index"] - 1
    assert raw_grid.options.optimize_global is False
    assert np.allclose(raw_grid.cell_center_xyz, raw_parent.cell_center_xyz[raw_parent_cells])
    assert optimized_grid.options.optimize_global is True
    assert optimized_grid.dims["cell"] > 0


def test_limited_area_normals_follow_local_cell_order_and_parent_provenance():
    parent = generate_grid("R01B02", optimize_global=False)
    grid = generate_grid(
        LimitedAreaGridSpec(
            parent="R01B02",
            region=Region.lonlat_box(
                lon_min=-100.0,
                lon_max=100.0,
                lat_min=-60.0,
                lat_max=60.0,
            ),
            boundary_depth=1,
        ),
        optimize_global=False,
    )
    source_edges = grid.refinement["parent_edge_index"] - 1
    first_source_cells = (
        grid.refinement["parent_cell_index"][grid.edge_cells[:, 0]] - 1
    )
    expected_sign = np.where(
        first_source_cells == parent.edge_cells[source_edges, 0],
        1.0,
        -1.0,
    )
    normal_alignment = np.sum(
        grid.geometry["edge_primal_normal_cartesian"]
        * parent.geometry["edge_primal_normal_cartesian"][source_edges],
        axis=1,
    )

    assert np.allclose(normal_alignment, expected_sign)
    assert grid.metadata["uuidOfParHGrid"] == parent.metadata["uuidOfHGrid"]


@pytest.mark.parametrize(
    "spec",
    [
        ChannelGridSpec(nx=5, ny=3, edge_length=1.0),
        ParallelogramGridSpec(nx=4, ny=3, edge_length=1.0),
        RaggedOrthogonalGridSpec(nx=4, ny=3, dx=1.0, dy=0.8),
    ],
)
def test_open_planar_normals_give_zero_divergence_for_constant_vectors(spec):
    grid = generate_grid(spec)
    normals = grid.geometry["edge_primal_normal_cartesian"]

    assert np.allclose(cell_divergence(grid, normals[:, 0]), 0.0, atol=1.0e-12)
    assert np.allclose(cell_divergence(grid, normals[:, 1]), 0.0, atol=1.0e-12)


def test_cut_grid_supports_region_predicates_keep_remove_and_metadata():
    parent = generate_grid("R02B01", options={"max_cells": None})
    keep_spec = CutGridSpec(
        regions=(
            Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0),
            Region.lonlat_box(lon_min=-20.0, lon_max=20.0, lat_min=-15.0, lat_max=15.0),
            Region.rectangle(
                center_lon=0.0,
                center_lat=0.0,
                width_degrees=30.0,
                height_degrees=20.0,
                angle_degrees=20.0,
            ),
            Region.polygon(points=((-35.0, -5.0), (0.0, 30.0), (35.0, -5.0))),
        ),
        boundary_depth=1,
        smoothing_depth=2,
        name="CUT_KEEP",
    )
    cut = cut_grid(parent, keep_spec)
    remove = cut_grid(
        parent,
        CutGridSpec(
            regions=Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0),
            mode="remove",
        ),
    )

    assert cut.name == "CUT_KEEP"
    assert cut.dims["cell"] > 0
    assert cut.dims["cell"] < parent.dims["cell"]
    assert remove.dims["cell"] > 0
    assert remove.dims["cell"] < parent.dims["cell"]
    assert cut.metadata["source_grid_name"] == parent.name
    assert cut.metadata["boundary_depth_index"] == 1
    assert cut.metadata["smoothing_depth"] == 2
    assert np.any(cut.edge_cells[:, 1] < 0)
    assert np.all(cut.refinement["parent_cell_index"] > 0)
    assert np.all(cut.refinement["smooth_c_ctrl"] == 2)
    assert cut.metadata["uuidOfParHGrid"] == parent.metadata["uuidOfHGrid"]
    assert cut.metadata["grid_mapping_name"] == parent.metadata["grid_mapping_name"]
    assert cut.metadata["crs_name"] == parent.metadata["crs_name"]


def test_derived_grid_uuids_include_parent_identity_and_parent_options():
    region = Region.circle(lon=0.0, lat=0.0, radius_degrees=60.0)
    limited_spec = LimitedAreaGridSpec(parent="R01B02", region=region)
    unrotated = generate_grid(limited_spec, optimize_global=False)
    rotated = generate_grid(
        limited_spec,
        optimize_global=False,
        north_pole_lon=30.0,
        north_pole_lat=70.0,
        rotation_angle_degrees=10.0,
    )
    first_parent = generate_grid("R01B01", optimize_global=False)
    second_parent = generate_grid("R02B01", optimize_global=False)
    first_cut = cut_grid(first_parent, region)
    second_cut = cut_grid(second_parent, region)

    assert unrotated.metadata["uuidOfHGrid"] != rotated.metadata["uuidOfHGrid"]
    assert first_cut.metadata["uuidOfHGrid"] != second_cut.metadata["uuidOfHGrid"]
    assert first_cut.metadata["uuidOfParHGrid"] == first_parent.metadata["uuidOfHGrid"]
    assert second_cut.metadata["uuidOfParHGrid"] == second_parent.metadata["uuidOfHGrid"]


def test_cut_grid_accepts_region_directly_for_common_case():
    parent = generate_grid("R02B01", max_cells=None)
    cut = cut_grid(
        parent,
        Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0),
        boundary_depth=1,
        smoothing_depth=2,
        name="CUT_DIRECT",
    )

    assert cut.name == "CUT_DIRECT"
    assert cut.metadata["boundary_depth_index"] == 1
    assert cut.metadata["smoothing_depth"] == 2
    assert cut.dims["cell"] > 0
    assert cut.dims["cell"] < parent.dims["cell"]

    with pytest.raises(TypeError, match="CutGridSpec"):
        cut_grid(
            parent,
            CutGridSpec(regions=Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0)),
            boundary_depth=1,
        )


def test_cut_grid_boundary_expansion_ignores_open_grid_missing_neighbors():
    parent = generate_grid(ChannelGridSpec(nx=3, ny=2, edge_length=1.0))
    cut = cut_grid(
        parent,
        CutGridSpec(
            regions=Region.lonlat_box(
                lon_min=-180.0,
                lon_max=-60.0,
                lat_min=-90.0,
                lat_max=-60.0,
            ),
            boundary_depth=1,
        ),
    )

    assert np.all(cut.refinement["parent_cell_index"] > 0)
    assert np.all(cut.refinement["parent_cell_index"] <= parent.dims["cell"])


def test_cut_grid_spec_rejects_unsupported_region_objects():
    with pytest.raises(TypeError, match="Region"):
        CutGridSpec(regions=("not-a-region",))
