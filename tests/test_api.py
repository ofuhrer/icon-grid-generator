from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import grid_generator as grid_generator_package
from grid_generator import (
    ChannelGridSpec,
    IconGrid,
    IconGridOptions,
    GlobalGridSpec,
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    Region,
    TorusGridSpec,
    generate_grid,
)
from grid_generator import grid_generator as gg
from grid_generator.grid_generator import parse_grid_spec
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec


def assert_lon_lat_match_xyz(lon, lat, xyz):
    radius = np.linalg.norm(xyz, axis=1)
    expected_lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    expected_lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / radius, -1.0, 1.0)))
    assert np.allclose(lon, expected_lon)
    assert np.allclose(lat, expected_lat)


def assert_outward_cells(grid):
    vertices = grid.vertices
    triangles = vertices[grid.cells]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroids = triangles.sum(axis=1)
    assert np.all(np.sum(normals * centroids, axis=1) > 0.0)


def test_public_package_exports_only_supported_grid_api():
    assert grid_generator_package.__all__ == [
        "generate_grid",
        "IconGrid",
        "IconGridOptions",
        "GlobalGridSpec",
        "TorusGridSpec",
        "ChannelGridSpec",
        "ParallelogramGridSpec",
        "LimitedAreaGridSpec",
        "Region",
    ]
    assert "write_icon_grid" not in grid_generator_package.__all__
    assert not hasattr(grid_generator_package, "write_icon_grid")
    assert not hasattr(grid_generator_package, "GeneratedGrid")
    assert not hasattr(grid_generator_package, "GridOptions")
    assert not hasattr(grid_generator_package, "GridSpec")
    assert not hasattr(grid_generator_package, "IconGridSpec")
    assert not hasattr(grid_generator_package, "LimitedAreaSpec")
    assert not hasattr(grid_generator_package, "StretchedTorusGridSpec")
    assert not hasattr(grid_generator_package, "RaggedOrthogonalGridSpec")
    assert not hasattr(grid_generator_package, "GlobalGridOptions")
    assert not hasattr(grid_generator_package, "GlobalOptimizationOptions")
    assert not hasattr(grid_generator_package, "optimize_grid")
    assert not hasattr(grid_generator_package, "check_grid")
    assert not hasattr(grid_generator_package, "CutGridSpec")
    assert grid_generator_package.IconGrid is IconGrid
    assert grid_generator_package.IconGridOptions is IconGridOptions
    assert grid_generator_package.GlobalGridSpec is GlobalGridSpec
    assert grid_generator_package.LimitedAreaGridSpec is LimitedAreaGridSpec
    assert grid_generator_package.TorusGridSpec is TorusGridSpec
    assert grid_generator_package.generate_grid is generate_grid
    assert grid_generator_package.ChannelGridSpec is ChannelGridSpec
    assert grid_generator_package.Region is Region


def test_parse_grid_spec_normalizes_supported_names_and_expected_counts():
    assert parse_grid_spec("R01B01").name == "R01B01"
    assert parse_grid_spec("R1B1").name == "R01B01"
    assert parse_grid_spec("R2B6").name == "R02B06"
    assert parse_grid_spec(" r02b03 ").name == "R02B03"

    spec = parse_grid_spec("R02B03")

    assert isinstance(spec, GlobalGridSpec)
    assert spec.root == 2
    assert spec.bisections == 3
    assert spec.frequency == 16
    assert spec.expected_cells == 5120
    assert spec.expected_edges == 7680
    assert spec.expected_vertices == 2562


def test_global_grid_spec_derives_frequency_and_name():
    spec = GlobalGridSpec(root=2, bisections=3)

    assert spec.frequency == 16
    assert spec.name == "R02B03"
    assert generate_grid(spec).dims["cell"] == 5120


def test_global_grid_spec_normalizes_or_rejects_custom_name():
    spec = GlobalGridSpec(root=2, bisections=3, name=" r2b3 ")

    assert spec.name == "R02B03"
    assert gg.grid_uuid(spec.name) == gg.grid_uuid("R02B03")
    assert GlobalGridSpec(root=2, bisections=6, name="r2b6").name == "R02B06"

    with pytest.raises(ValueError, match="name must match"):
        GlobalGridSpec(root=2, bisections=3, name="R01B00")
    with pytest.raises(ValueError, match=r"form R<n>B<k>"):
        GlobalGridSpec(root=1, bisections=0, name="custom")


def test_global_grid_spec_rejects_inconsistent_frequency():
    with pytest.raises(ValueError, match=r"frequency must equal root \* 2\*\*bisections"):
        GlobalGridSpec(root=2, bisections=3, frequency=15)


@pytest.mark.parametrize(
    "grid_name",
    [None, "", "foo", "R00B01", "R01", "01B01", "R01B-1", "R1.0B1"],
)
def test_parse_grid_spec_rejects_invalid_names(grid_name):
    with pytest.raises((TypeError, ValueError)):
        parse_grid_spec(grid_name)


def test_parse_grid_spec_negative_bisection_defensive_guard(monkeypatch):
    class FakeMatch:
        def group(self, index):
            return {1: "1", 2: "-1"}[index]

    class FakeRegex:
        def fullmatch(self, value):
            assert value == "R01B-1"
            return FakeMatch()

    monkeypatch.setattr(gg, "GRID_NAME_RE", FakeRegex())

    with pytest.raises(ValueError, match="bisections must be non-negative"):
        parse_grid_spec("R01B-1")


def test_generate_grid_accepts_all_public_grid_specs():
    global_grid = generate_grid(parse_grid_spec("R01B00"))
    torus_grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=1.0))
    stretched_grid = generate_grid(StretchedTorusGridSpec(nx=4, ny=3, edge_length=1.0))
    channel_grid = generate_grid(ChannelGridSpec(nx=3, ny=2, edge_length=1.0))
    parallelogram_grid = generate_grid(ParallelogramGridSpec(nx=3, ny=2, edge_length=1.0))
    ragged_grid = generate_grid(RaggedOrthogonalGridSpec(nx=3, ny=2, dx=1.0, dy=1.0))
    limited_area_grid = generate_grid(
        LimitedAreaGridSpec(
            parent="R02B01",
            region=Region.lonlat_box(
                lon_min=-30.0,
                lon_max=30.0,
                lat_min=-30.0,
                lat_max=30.0,
            ),
            boundary_depth=1,
        ),
        options={"max_cells": None},
    )

    assert global_grid.metadata["grid_geometry"] == 1
    assert torus_grid.metadata["grid_geometry"] == 2
    assert stretched_grid.metadata["grid_geometry"] == 2
    assert channel_grid.metadata["grid_geometry"] == 2
    assert parallelogram_grid.metadata["grid_geometry"] == 2
    assert ragged_grid.metadata["grid_geometry"] == 2
    assert limited_area_grid.metadata["grid_geometry"] == 3


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        (object(), TypeError, "options must be"),
        ({"unknown": 1}, TypeError, "unknown grid option"),
        ({"radius": 0.0}, ValueError, "radius must be positive"),
        ({"radius": -1.0}, ValueError, "radius must be positive"),
        ({"radius": math.nan}, ValueError, "radius must be finite"),
        ({"radius": math.inf}, ValueError, "radius must be finite"),
        ({"sphere_radius": 0.0}, ValueError, "sphere_radius must be positive"),
        ({"sphere_radius": math.nan}, ValueError, "sphere_radius must be finite"),
        ({"sphere_radius": math.inf}, ValueError, "sphere_radius must be finite"),
        ({"max_cells": 3.14}, TypeError, "max_cells"),
        ({"max_cells": "100"}, TypeError, "max_cells"),
        ({"max_cells": 0}, ValueError, "max_cells must be positive"),
        ({"optimize_global": 1}, TypeError, "optimize_global"),
        ({"spring_beta": 0.0}, ValueError, "beta_spring"),
        (
            {"north_pole_lon": math.inf},
            ValueError,
            "north_pole_lon",
        ),
        (
            {"rotation_angle_degrees": "0.05"},
            TypeError,
            "rotation_angle_degrees",
        ),
        ({"accelerator": 1}, TypeError, "accelerator"),
        ({"accelerator": "fast"}, ValueError, "accelerator"),
        ({"spring_iterations": -1}, ValueError, "maxit"),
        ({"spring_iterations": 3.14}, TypeError, "maxit"),
        ({"indexing": "fast"}, ValueError, "indexing_algorithm"),
    ],
)
def test_generate_grid_rejects_invalid_options(options, error, message):
    with pytest.raises(error, match=message):
        generate_grid("R01B00", options=options)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sphere_radius": 0.0}, ValueError, "sphere_radius must be positive"),
        ({"sphere_radius": math.nan}, ValueError, "sphere_radius must be finite"),
        ({"sphere_radius": math.inf}, ValueError, "sphere_radius must be finite"),
        ({"options": {"north_pole_lat": math.inf}}, ValueError, "north_pole_lat"),
        (
            {"options": {"rotation_angle_degrees": "0.05"}},
            TypeError,
            "rotation_angle_degrees",
        ),
        ({"options": {"optimize_global": 1}}, TypeError, "optimize_global"),
    ],
)
def test_grid_uuid_rejects_invalid_numeric_inputs(kwargs, error, message):
    with pytest.raises(error, match=message):
        gg.grid_uuid("R01B00", **kwargs)


@pytest.mark.parametrize(
    ("grid_name", "cells", "edges", "vertices"),
    [
        ("R01B00", 20, 30, 12),
        ("R01B01", 80, 120, 42),
        ("R02B02", 1280, 1920, 642),
    ],
)
def test_known_grid_dimensions(grid_name, cells, edges, vertices):
    grid = generate_grid(grid_name)

    assert grid.dims == {"cell": cells, "vertex": vertices, "edge": edges}
    assert grid.cells.shape == (cells, 3)
    assert grid.edges.shape == (edges, 2)
    assert grid.vertices.shape == (vertices, 3)
    assert grid.cell_edges.shape == (cells, 3)
    assert grid.edge_cells.shape == (edges, 2)


def test_icon_grid_options_instance_produces_complete_grid():
    options = IconGridOptions(max_cells=None, radius=3.0, sphere_radius=4.0)
    grid = generate_grid("R01B01", options=options)
    grid_dict = grid.to_dict()
    dataset = grid.to_xarray()

    assert grid.options is options
    assert grid.dims == {"cell": 80, "vertex": 42, "edge": 120}
    assert grid.edges.shape == (120, 2)
    assert grid.cell_edges.shape == (80, 3)
    assert grid.edge_cells.shape == (120, 2)
    assert grid.edge_center_xyz.shape == (120, 3)
    assert grid.edge_lon.shape == (120,)
    assert grid.edge_lat.shape == (120,)
    assert grid.icon_connectivity
    assert grid.connectivity
    assert grid.neighbor_tables
    assert grid.geometry
    assert grid.refinement
    assert grid.metadata["grid_root"] == 1
    assert grid.metadata["sphere_radius"] == 4.0
    assert "mean_cell_area" in grid.metadata
    assert "edges" in grid_dict
    assert dataset.sizes["edge"] == 120
    assert dataset.attrs["name"] == "R01B01"
    assert dataset.attrs["radius"] == 3.0
    assert dataset.sizes["cell"] == 80
    assert dataset.sizes["vertex"] == 42


def test_generate_grid_accepts_keyword_option_overrides():
    mapping_grid = generate_grid("R01B01", options={"sphere_radius": 2.0})
    keyword_grid = generate_grid("R01B01", sphere_radius=2.0)
    overridden_grid = generate_grid(
        "R01B01",
        options=IconGridOptions(max_cells=1, sphere_radius=2.0),
        max_cells=None,
        sphere_radius=3.0,
    )

    assert keyword_grid.options.sphere_radius == 2.0
    assert np.array_equal(mapping_grid.cells, keyword_grid.cells)
    assert overridden_grid.options.max_cells is None
    assert overridden_grid.options.sphere_radius == 3.0
    with pytest.raises(TypeError, match="unexpected keyword"):
        generate_grid("R01B00", not_an_option=True)


@pytest.mark.parametrize(
    "call_kwargs",
    [
        {"options": {"optimize_global": True}},
        {"optimize_global": True},
        {"options": IconGridOptions(optimize_global=True)},
    ],
)
def test_planar_specs_reject_explicit_global_optimization(call_kwargs):
    with pytest.raises(ValueError, match="only supported for global grids"):
        generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=1.0), **call_kwargs)


@pytest.mark.parametrize(
    "call_kwargs",
    [
        {},
        {"options": {"optimize_global": False}},
        {"optimize_global": False},
        {"options": IconGridOptions(optimize_global=False)},
    ],
)
def test_planar_specs_accept_absent_or_disabled_global_optimization(call_kwargs):
    grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=1.0), **call_kwargs)

    assert grid.metadata["grid_geometry"] == 2
    assert grid.options.optimize_global is False


def test_generate_grid_signature_exposes_common_option_keywords():
    signature = inspect.signature(generate_grid)

    assert list(signature.parameters) == [
        "spec",
        "options",
        "max_cells",
        "accelerator",
        "radius",
        "sphere_radius",
        "optimize_global",
        "spring_beta",
        "spring_iterations",
        "fixed_boundary",
        "north_pole_lon",
        "north_pole_lat",
        "rotation_angle_degrees",
        "indexing",
        "centre",
        "subcentre",
        "number_of_grid_used",
    ]
    for name in list(signature.parameters)[2:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_icon_grid_core_geometry_arrays_are_consistent():
    radius = 6.0
    grid = generate_grid(
        "R01B01",
        options={"radius": radius},
    )

    vertex_radius = np.linalg.norm(grid.vertices, axis=1)
    center_radius = np.linalg.norm(grid.cell_center_xyz, axis=1)
    edge_center_radius = np.linalg.norm(grid.edge_center_xyz, axis=1)

    assert grid.name == "R01B01"
    assert grid.spec.name == grid.name
    assert np.allclose(vertex_radius, radius)
    assert np.allclose(center_radius, radius)
    assert np.allclose(edge_center_radius, radius)
    assert_lon_lat_match_xyz(grid.lon, grid.lat, grid.cell_center_xyz)
    assert_lon_lat_match_xyz(grid.vertex_lon, grid.vertex_lat, grid.vertices)
    assert_lon_lat_match_xyz(grid.edge_lon, grid.edge_lat, grid.edge_center_xyz)
    assert np.array_equal(grid.cell_vertex_lon, grid.vertex_lon[grid.cells])
    assert np.array_equal(grid.cell_vertex_lat, grid.vertex_lat[grid.cells])
    assert np.all((-180.0 <= grid.lon) & (grid.lon <= 180.0))
    assert np.all((-90.0 <= grid.lat) & (grid.lat <= 90.0))
    assert np.all((-180.0 <= grid.vertex_lon) & (grid.vertex_lon <= 180.0))
    assert np.all((-90.0 <= grid.vertex_lat) & (grid.vertex_lat <= 90.0))
    assert np.all((-180.0 <= grid.edge_lon) & (grid.edge_lon <= 180.0))
    assert np.all((-90.0 <= grid.edge_lat) & (grid.edge_lat <= 90.0))
    assert_outward_cells(grid)


def test_to_dict_contains_all_icon_grid_arrays_and_reuses_objects():
    grid = generate_grid("R01B00")
    data = grid.to_dict()

    assert data["name"] == "R01B00"
    assert data["kind"] == "R01B00"
    assert data["spec"] is grid.spec
    assert data["dims"] == grid.dims
    for key in [
        "vertices",
        "cells",
        "lon",
        "lat",
        "vertex_lon",
        "vertex_lat",
        "cell_center_xyz",
        "cell_vertex_lon",
        "cell_vertex_lat",
        "edges",
        "cell_edges",
        "edge_cells",
        "edge_center_xyz",
        "edge_lon",
        "edge_lat",
    ]:
        assert data[key] is getattr(grid, key)
    assert data["connectivity"] is grid.connectivity
    assert data["neighbor_tables"] is grid.neighbor_tables
    assert data["geometry"] is grid.geometry
    assert data["refinement"] is grid.refinement
    assert data["metadata"] is grid.metadata


def test_to_xarray_contains_coordinates_data_variables_and_attrs():
    grid = generate_grid("R01B00", options={"radius": 7.0})
    dataset = grid.to_xarray()

    assert dataset.sizes == {"vertex": 12, "xyz": 3, "cell": 20, "cell_vertex": 3, "edge": 30, "edge_vertex": 2, "edge_cell": 2}
    assert dataset.attrs["name"] == "R01B00"
    assert dataset.attrs["root"] == 1
    assert dataset.attrs["bisections"] == 0
    assert dataset.attrs["frequency"] == 1
    assert dataset.attrs["radius"] == 7.0
    assert np.array_equal(dataset["xyz"].values, np.array(["x", "y", "z"]))
    assert np.array_equal(dataset["cell_vertex"].values, np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(dataset["edge_vertex"].values, np.array([0, 1], dtype=np.int32))
    assert np.array_equal(dataset["edge_cell"].values, np.array([0, 1], dtype=np.int32))
    assert np.array_equal(dataset["vertices"].values, grid.vertices)
    assert np.array_equal(dataset["cells"].values, grid.cells)
    assert np.array_equal(dataset["edges"].values, grid.edges)
    assert np.array_equal(dataset["edge_center_xyz"].values, grid.edge_center_xyz)
    assert np.array_equal(dataset["edge_lon"].values, grid.edge_lon)
    assert np.array_equal(dataset["edge_lat"].values, grid.edge_lat)


def test_safety_cap_fails_clearly_and_can_be_changed_or_disabled():
    with pytest.raises(ValueError, match="exceeding max_cells"):
        generate_grid("R02B02", options={"max_cells": 10})

    assert generate_grid("R02B02", options={"max_cells": 2_000}).dims["cell"] == 1280
    assert IconGridOptions().max_cells == 2_000_000
    assert generate_grid("R01B01", options={"max_cells": None}).dims["cell"] == 80


def test_global_grid_generation_rejects_int32_index_overflow():
    with pytest.raises(ValueError, match="int32 index limit"):
        generate_grid("R02B13", options={"max_cells": None})
