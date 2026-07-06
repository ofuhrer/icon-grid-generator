from __future__ import annotations

import builtins
from dataclasses import replace
import math
import numpy as np
import pytest

import grid_generator as grid_generator_package
from grid_generator import _accelerated
from grid_generator import (
    ChannelGridSpec,
    CutGridSpec,
    IconGrid,
    IconGridOptions,
    GlobalGridSpec,
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    Region,
    TorusGridSpec,
    generate_grid,
)
from grid_generator.cutting import cut_grid
from grid_generator.diagnostics import (
    cell_divergence,
    cell_vorticity_fnorm,
    check_grid,
    grid_statistics,
    triangle_properties,
)
from grid_generator import grid_generator as gg
from grid_generator._geometry import SphericalIcosahedralGeometry
from grid_generator._metrics import SphericalMetricsBuilder
from grid_generator._ordering import IconOrderingBuilder
from grid_generator._refinement import GlobalRefinementBuilder
from grid_generator._topology import GlobalTopologyBuilder
from grid_generator.grid_generator import parse_grid_spec
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec
from grid_generator.transforms import (
    DiffusionOptions,
    OptimizationOptions,
    diffuse_grid,
    optimize_global_grid,
    optimize_grid,
)


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
        "CutGridSpec",
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
    assert grid_generator_package.IconGrid is IconGrid
    assert grid_generator_package.IconGridOptions is IconGridOptions
    assert grid_generator_package.GlobalGridSpec is GlobalGridSpec
    assert grid_generator_package.LimitedAreaGridSpec is LimitedAreaGridSpec
    assert grid_generator_package.TorusGridSpec is TorusGridSpec
    assert grid_generator_package.generate_grid is generate_grid
    assert grid_generator_package.ChannelGridSpec is ChannelGridSpec
    assert grid_generator_package.Region is Region


def assert_unit_sphere(points):
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)


def assert_outward_cells(grid):
    vertices = grid.vertices
    triangles = vertices[grid.cells]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroids = triangles.sum(axis=1)
    assert np.all(np.sum(normals * centroids, axis=1) > 0.0)


def assert_lon_lat_match_xyz(lon, lat, xyz):
    radius = np.linalg.norm(xyz, axis=1)
    expected_lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    expected_lat = np.degrees(np.arcsin(np.clip(xyz[:, 2] / radius, -1.0, 1.0)))
    assert np.allclose(lon, expected_lon)
    assert np.allclose(lat, expected_lat)


def unit_rows(points):
    return points / np.linalg.norm(points, axis=1)[:, np.newaxis]


def lon_unit_circle(lon):
    radians = np.radians(lon)
    return np.column_stack((np.cos(radians), np.sin(radians)))


def expected_edge_system_orientation(grid):
    vertices = unit_rows(grid.vertices)
    centers = unit_rows(grid.cell_center_xyz)
    edge_centers = unit_rows(grid.edge_center_xyz)
    vertex_direction = vertices[grid.edges[:, 1]] - vertices[grid.edges[:, 0]]
    cell_direction = centers[grid.edge_cells[:, 1]] - centers[grid.edge_cells[:, 0]]
    outward_component = np.sum(np.cross(vertex_direction, cell_direction) * edge_centers, axis=1)
    return np.where(outward_component > 0.0, 1, -1).astype(np.int32)


GLOBAL_RELAXATION_SNAPSHOTS = {
    "R02B02": {
        "coordinates": {
            "cells": [
                [-0.9972766427313585, 1.2213116483853766e-16, 0.07375159566050225],
                [-0.0007245100163799303, -0.7250902788999919, 0.6886534415291687],
                [0.9972766427257168, -1.226463717588987e-16, -0.0737515957367906],
            ],
            "edges": [
                [-0.9980032647133953, -0.03975230771934482, 0.04908398570197008],
                [-0.0017330106148963427, -0.9236895705201995, 0.383137800257842],
                [0.9980032646925895, 0.039752307641365794, -0.049083986188157874],
            ],
            "vertices": [
                [-0.9998905285231824, 1.2245127352536707e-16, -0.01479631608309715],
                [-5.851658108239274e-09, 1.0953727274925692e-16, 1.0],
                [0.9998905285183876, 4.176294833076613e-16, 0.014796316407119888],
            ],
        },
        "metrics": {
            "cell_area": [
                334455830974.8723,
                373499669297.1764,
                395086529658.12134,
                400748023387.92334,
                402345307710.89734,
                405184839120.001,
                410670878396.2804,
                413398864029.7295,
                423588523345.1393,
            ],
            "dual_edge_length": [
                398089.78549738013,
                471391.519156158,
                508539.20373929123,
                534028.1209502177,
                550110.3909770866,
                582849.0214484674,
                607945.7344405836,
                627791.9882128286,
                647510.7180244914,
            ],
            "edge_cell_distance": [
                161788.83798968507,
                243300.02701519613,
                256046.8527435893,
                265239.16919482104,
                282622.43340112607,
                288684.8784495073,
                303972.86772788863,
                319053.33939860546,
                323755.35907112143,
            ],
            "edge_length": [
                838004.9203659653,
                914946.1134868287,
                922771.4117992753,
                958450.2840412285,
                965223.3426980093,
                984220.5540255363,
                999964.4123921479,
                1003757.022946344,
                1011276.1609523273,
            ],
        },
    },
    "R02B04": {
        "coordinates": {
            "cells": [
                [-0.9999726130327524, 2.786923611539062e-15, 0.007400890787543893],
                [-3.7112756342732476e-05, -0.8121082181193181, 0.5835068471626612],
                [0.9999726130342036, 8.311396524546055e-17, -0.007400890591470153],
            ],
            "edges": [
                [-0.9999502825338646, -0.00988773966361991, 0.0012903738949580188],
                [-0.00010628419052694703, -0.83447960535123, -0.551038816197667],
                [0.9999502825341648, 0.009887739647363187, -0.001290373786913602],
            ],
            "vertices": [
                [-0.9998918416038874, 1.010533790078164e-15, -0.014707314302296805],
                [-5.302317535208647e-09, -7.412125961010718e-16, 1.0],
                [0.9998918415996828, -6.878558226253009e-16, 0.014707314588149377],
            ],
        },
        "metrics": {
            "cell_area": [
                18776260998.262985,
                23800601803.3689,
                24583400115.7571,
                25021682703.170013,
                25283642484.151394,
                25474817831.921032,
                25561939708.756954,
                25692650375.074226,
                25973307559.66683,
            ],
            "dual_edge_length": [
                91477.0435846128,
                121215.54128864895,
                128177.01938581436,
                132022.931804176,
                137611.61619351737,
                145012.18596433106,
                152462.67431252258,
                157997.06515464923,
                162013.55548335367,
            ],
            "edge_cell_distance": [
                37971.55139856784,
                60601.37924319611,
                64105.592291782465,
                66061.38046768436,
                69053.92384079934,
                72356.39201979684,
                76077.35330314182,
                79174.657822828,
                81006.77774523124,
            ],
            "edge_length": [
                198699.8369479145,
                228171.23421985135,
                232809.04562551054,
                237704.8107192587,
                242417.07603140824,
                246434.95949836235,
                248953.8877056851,
                250582.24354055093,
                252917.57302064548,
            ],
        },
    },
}


def sorted_value_samples(values, count):
    sorted_values = np.sort(np.asarray(values).ravel())
    return sorted_values[np.linspace(0, sorted_values.size - 1, count, dtype=int)]


def assert_global_relaxation_snapshot(grid):
    snapshot = GLOBAL_RELAXATION_SNAPSHOTS[grid.name]
    coordinate_sources = {
        "vertices": unit_rows(grid.vertices),
        "cells": unit_rows(grid.cell_center_xyz),
        "edges": unit_rows(grid.edge_center_xyz),
    }
    for name, rows in coordinate_sources.items():
        expected = np.asarray(snapshot["coordinates"][name], dtype=np.float64)
        distances = np.linalg.norm(
            rows[np.newaxis, :, :] - expected[:, np.newaxis, :],
            axis=2,
        )
        assert np.max(np.min(distances, axis=1)) <= 1.0e-8

    for name, expected_values in snapshot["metrics"].items():
        actual = sorted_value_samples(grid.geometry[name], len(expected_values))
        expected = np.asarray(expected_values, dtype=np.float64)
        scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))), 1.0)
        assert np.max(np.abs(actual - expected)) / scale <= 2.0e-8


def local_east_north(points):
    unit_points = unit_rows(points)
    lon = np.arctan2(unit_points[:, 1], unit_points[:, 0])
    lat = np.arcsin(np.clip(unit_points[:, 2], -1.0, 1.0))
    east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)))
    north = np.column_stack(
        (-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat))
    )
    return east, north


def spherical_triangle_areas_lhuilier(vertices, cells, sphere_radius):
    triangles = unit_rows(vertices)[cells]
    side_a = np.arccos(np.clip(np.sum(triangles[:, 1] * triangles[:, 2], axis=1), -1.0, 1.0))
    side_b = np.arccos(np.clip(np.sum(triangles[:, 0] * triangles[:, 2], axis=1), -1.0, 1.0))
    side_c = np.arccos(np.clip(np.sum(triangles[:, 0] * triangles[:, 1], axis=1), -1.0, 1.0))
    semiperimeter = 0.5 * (side_a + side_b + side_c)
    tan_quarter_excess = np.sqrt(
        np.tan(0.5 * semiperimeter)
        * np.tan(0.5 * (semiperimeter - side_a))
        * np.tan(0.5 * (semiperimeter - side_b))
        * np.tan(0.5 * (semiperimeter - side_c))
    )
    return 4.0 * np.arctan(tan_quarter_excess) * sphere_radius**2


def geometric_dual_areas_from_cell_centers(grid, sphere_radius):
    centers = unit_rows(grid.cell_center_xyz)
    dual_areas = np.zeros(grid.dims["vertex"], dtype=np.float64)
    for vertex_index, ordered_cells in enumerate(grid.icon_connectivity["v2c"]):
        cell_indices = ordered_cells[ordered_cells > 0] - 1
        if cell_indices.size < 3:
            continue
        polygon = centers[cell_indices]
        fan_cells = np.array(
            [[0, index, index + 1] for index in range(1, polygon.shape[0] - 1)],
            dtype=np.int32,
        )
        dual_areas[vertex_index] = spherical_triangle_areas_lhuilier(
            polygon,
            fan_cells,
            sphere_radius,
        ).sum()
    dual_areas *= grid.geometry["cell_area"].sum() / dual_areas.sum()
    return dual_areas


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


def test_global_spring_optimization_preserves_topology_and_improves_quality():
    raw = generate_grid(
        "R02B02",
        options={"max_cells": None, "optimize_global": False},
    )
    optimized = generate_grid(
        "R02B02",
        options={
            "max_cells": None,
            "spring_iterations": 250,
        },
    )

    assert optimized.dims == raw.dims
    assert np.array_equal(optimized.cells, raw.cells)
    assert np.array_equal(optimized.edges, raw.edges)
    assert np.array_equal(optimized.cell_edges, raw.cell_edges)
    assert np.array_equal(optimized.edge_cells, raw.edge_cells)
    assert_unit_sphere(optimized.vertices)
    assert optimized.metadata["global_optimization"] == "spring"
    assert optimized.metadata["global_optimization_iterations"] == 250
    assert optimized.metadata["uuidOfHGrid"] != raw.metadata["uuidOfHGrid"]

    raw_cell_cv = np.std(raw.geometry["cell_area"]) / np.mean(raw.geometry["cell_area"])
    optimized_cell_cv = np.std(optimized.geometry["cell_area"]) / np.mean(
        optimized.geometry["cell_area"]
    )
    raw_edge_cv = np.std(raw.geometry["edge_length"]) / np.mean(raw.geometry["edge_length"])
    optimized_edge_cv = np.std(optimized.geometry["edge_length"]) / np.mean(
        optimized.geometry["edge_length"]
    )

    assert optimized_cell_cv <= raw_cell_cv
    assert optimized_edge_cv <= raw_edge_cv


def test_global_optimization_options_can_be_configured_and_called_directly():
    raw = generate_grid(
        "R02B01",
        options={"max_cells": None, "optimize_global": False},
    )
    options = {"method": "spring", "iterations": 20}
    optimized = optimize_global_grid(raw, options)
    facade = generate_grid(
        "R02B01",
        options={"max_cells": None, "spring_iterations": 20},
    )

    assert optimized.dims == raw.dims
    assert facade.metadata["global_optimization_iterations"] == 20
    assert np.all(np.isfinite(optimized.geometry["cell_area"]))
    assert not np.allclose(optimized.vertices, raw.vertices)


@pytest.mark.parametrize(
    ("grid_name", "options"),
    [
        ("R02B02", {"centre": 215, "subcentre": 0}),
        ("R02B04", None),
    ],
)
def test_default_global_generation_uses_staged_spring_relaxation(grid_name, options):
    grid = generate_grid(grid_name, options=options)

    assert_global_relaxation_snapshot(grid)
    assert grid.metadata["global_optimization"] == "spring"


def test_raw_global_generation_bypasses_staged_spring_relaxation():
    raw = generate_grid("R02B02", options={"optimize_global": False})
    relaxed = generate_grid("R02B02")

    assert raw.metadata["global_optimization"] == "none"
    assert np.array_equal(raw.cells, relaxed.cells)
    assert np.array_equal(raw.edges, relaxed.edges)
    assert np.array_equal(raw.cell_edges, relaxed.cell_edges)
    assert np.array_equal(raw.edge_cells, relaxed.edge_cells)
    assert not np.allclose(raw.vertices, relaxed.vertices)

    raw_cell_cv = np.std(raw.geometry["cell_area"]) / np.mean(raw.geometry["cell_area"])
    relaxed_cell_cv = np.std(relaxed.geometry["cell_area"]) / np.mean(
        relaxed.geometry["cell_area"]
    )
    assert relaxed_cell_cv < raw_cell_cv


@pytest.mark.parametrize("grid_name", ["R02B02", "R02B04"])
def test_staged_global_generation_preserves_connectivity_contract(grid_name):
    grid = generate_grid(grid_name)

    assert grid.dims == {
        "cell": grid.spec.expected_cells,
        "vertex": grid.spec.expected_vertices,
        "edge": grid.spec.expected_edges,
    }
    assert np.array_equal(grid.connectivity["edge_of_cell"], grid.icon_connectivity["c2e"])
    assert np.array_equal(grid.connectivity["vertex_of_cell"], grid.cells)
    assert np.array_equal(grid.connectivity["edge_vertices"], grid.edges)
    assert np.array_equal(grid.connectivity["adjacent_cell_of_edge"], grid.edge_cells)
    assert np.array_equal(
        grid.geometry["edge_system_orientation"],
        np.ones(grid.dims["edge"], dtype=np.int32),
    )

    for cell_index, cell in enumerate(grid.cells):
        for local_index, pair in enumerate(
            ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0]))
        ):
            edge_index = grid.cell_edges[cell_index, local_index]
            assert set(map(int, pair)) == set(map(int, grid.edges[edge_index]))
            expected_orientation = 1 if grid.edge_cells[edge_index, 0] == cell_index else -1
            assert grid.geometry["orientation_of_normal"][cell_index, local_index] == (
                expected_orientation
            )

    parent = generate_grid(
        f"R{grid.spec.root:02d}B{grid.spec.bisections - 1:02d}",
    )
    assert np.all(
        (1 <= grid.refinement["parent_cell_index"])
        & (grid.refinement["parent_cell_index"] <= parent.dims["cell"])
    )
    assert np.all(
        (1 <= grid.refinement["parent_edge_index"])
        & (grid.refinement["parent_edge_index"] <= parent.dims["edge"])
    )


def test_global_optimization_is_rejected_for_planar_specs():
    with pytest.raises(ValueError, match="only supported for global grids"):
        generate_grid(
            TorusGridSpec(nx=4, ny=4, edge_length=1.0),
            options={"optimize_global": True},
        )


def test_numpy_accelerator_matches_auto_for_global_grid():
    auto_grid = generate_grid("R02B02", options={"accelerator": "auto"})
    numpy_grid = generate_grid("R02B02", options={"accelerator": "numpy"})

    assert np.array_equal(numpy_grid.cells, auto_grid.cells)
    assert np.array_equal(numpy_grid.edges, auto_grid.edges)
    assert np.array_equal(numpy_grid.cell_edges, auto_grid.cell_edges)
    assert np.array_equal(numpy_grid.edge_cells, auto_grid.edge_cells)
    assert np.array_equal(
        numpy_grid.icon_connectivity["v2c"],
        auto_grid.icon_connectivity["v2c"],
    )
    assert np.array_equal(
        numpy_grid.refinement["parent_cell_index"],
        auto_grid.refinement["parent_cell_index"],
    )
    assert np.array_equal(
        numpy_grid.refinement["parent_edge_index"],
        auto_grid.refinement["parent_edge_index"],
    )


def test_auto_accelerator_uses_numba_only_for_large_lookup_work():
    threshold = _accelerated.AUTO_NUMBA_MIN_LOOKUP_ROWS

    assert not _accelerated.should_use_numba("auto")
    assert not _accelerated.should_use_numba("auto", threshold - 1)
    assert _accelerated.should_use_numba("auto", threshold) == _accelerated.is_numba_available()


def test_numba_accelerator_is_optional_and_matches_numpy_when_available():
    if not _accelerated.is_numba_available():
        with pytest.raises(ModuleNotFoundError, match="accelerate"):
            generate_grid("R02B02", options={"accelerator": "numba"})
        return

    numpy_grid = generate_grid("R02B02", options={"accelerator": "numpy"})
    numba_grid = generate_grid("R02B02", options={"accelerator": "numba"})

    assert np.array_equal(numba_grid.cells, numpy_grid.cells)
    assert np.array_equal(numba_grid.edges, numpy_grid.edges)
    assert np.array_equal(
        numba_grid.refinement["parent_cell_type"],
        numpy_grid.refinement["parent_cell_type"],
    )
    assert np.array_equal(
        numba_grid.refinement["edge_parent_type"],
        numpy_grid.refinement["edge_parent_type"],
    )


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


def test_internal_generation_pipeline_matches_public_facade():
    spec = parse_grid_spec("R01B01")
    options = IconGridOptions(
        sphere_radius=3.0,
        rotation_angle_degrees=0.05,
        optimize_global=False,
    )
    grid = generate_grid(spec.name, options=options)
    context = gg._GlobalGenerationContext()

    geometry = SphericalIcosahedralGeometry().build(spec, options)
    geometry = IconOrderingBuilder(context).order_spherical_bisection(spec, options, geometry)
    topology = GlobalTopologyBuilder().build(spec, options, geometry)
    topology = gg._adjust_global_edge_orientation(spec, options, geometry, topology, context)
    metrics = SphericalMetricsBuilder().build(options, geometry, topology)
    refinement = GlobalRefinementBuilder(context).build(spec, options, geometry, topology)

    for name in [
        "vertices",
        "cells",
        "lon",
        "lat",
        "vertex_lon",
        "vertex_lat",
        "cell_center_xyz",
        "cell_vertex_lon",
        "cell_vertex_lat",
    ]:
        assert np.array_equal(getattr(geometry, name), getattr(grid, name))
    for name in [
        "edges",
        "cell_edges",
        "edge_cells",
        "edge_center_xyz",
        "edge_lon",
        "edge_lat",
    ]:
        assert np.array_equal(getattr(topology, name), getattr(grid, name))
    for name, value in topology.icon_connectivity.items():
        assert np.array_equal(value, grid.icon_connectivity[name])
    for name, value in topology.connectivity.items():
        assert np.array_equal(value, grid.connectivity[name])
    for name, value in topology.neighbor_tables.items():
        assert np.array_equal(value, grid.neighbor_tables[name])
    for name, value in metrics.fields.items():
        assert np.array_equal(value, grid.geometry[name])
    for name, value in refinement.fields.items():
        assert np.array_equal(value, grid.refinement[name])


def test_global_grid_rotation_defaults_to_unrotated_and_can_be_enabled():
    unrotated = generate_grid("R01B01")
    rotated = generate_grid(
        "R01B01",
        options={"rotation_angle_degrees": 0.05},
    )

    assert unrotated.options.rotation_angle_degrees == 0.0
    assert rotated.options.rotation_angle_degrees == 0.05
    assert np.array_equal(rotated.cells, unrotated.cells)
    assert np.array_equal(rotated.edges, unrotated.edges)
    assert not np.allclose(rotated.vertices, unrotated.vertices)
    assert np.allclose(np.linalg.norm(rotated.vertices, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(unrotated.vertices, axis=1), 1.0)


@pytest.mark.parametrize(
    ("grid_name", "parent_grid_name"),
    [("R01B02", "R01B01"), ("R02B03", "R02B02")],
)
def test_spherical_bisection_cells_follow_icon_child_ordering(
    grid_name,
    parent_grid_name,
):
    grid = generate_grid(grid_name)
    refinement = grid.refinement
    child_types = refinement["parent_cell_type"].reshape(-1, 4)
    parent_cells = refinement["parent_cell_index"].reshape(-1, 4)

    assert np.all(child_types == np.array([200, 203, 201, 202], dtype=np.int32))
    assert np.all(parent_cells == parent_cells[:, :1])
    assert np.array_equal(
        parent_cells[:, 0],
        np.arange(1, generate_grid(parent_grid_name).dims["cell"] + 1, dtype=np.int32),
    )


def test_spherical_bisection_parent_cell_types_match_child_geometry():
    grid = generate_grid("R02B03")
    parent = generate_grid("R02B02")
    parent_vertex_index = grid.refinement["parent_vertex_index"]
    type_to_parent_vertex = {
        201: 0,
        202: 1,
        203: 2,
    }

    for cell_index, child_type in enumerate(grid.refinement["parent_cell_type"]):
        parent_cell = grid.refinement["parent_cell_index"][cell_index] - 1
        child_parent_vertices = parent_vertex_index[grid.cells[cell_index]]
        inherited_vertices = child_parent_vertices[child_parent_vertices > 0]
        if child_type == 200:
            assert inherited_vertices.size == 0
            continue

        parent_vertex_position = type_to_parent_vertex[int(child_type)]
        assert inherited_vertices.size == 1
        assert inherited_vertices[0] == parent.cells[parent_cell, parent_vertex_position] + 1


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
    assert np.allclose(grid.geometry["edgequad_area"], 0.0)
    assert np.all(np.isfinite(grid.vertices))
    assert np.all(np.isfinite(grid.cell_center_xyz))
    assert np.all(np.isfinite(grid.edge_center_xyz))
    assert np.all(np.isfinite(grid.geometry["edge_primal_normal_cartesian"]))


def test_torus_netcdf_export_contains_complete_periodic_grid(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=2.0))
    path = grid.to_netcdf(tmp_path / "torus.nc")

    with netcdf4.Dataset(path) as dataset:
        assert dataset.dimensions["cell"].size == 24
        assert dataset.dimensions["vertex"].size == 12
        assert dataset.dimensions["edge"].size == 36
        assert dataset.getncattr("grid_geometry") == 2
        assert dataset.getncattr("torus_nx") == 4
        assert dataset.getncattr("torus_ny") == 3
        assert np.array_equal(dataset.variables["adjacent_cell_of_edge"][:], grid.edge_cells.T + 1)
        assert np.allclose(dataset.variables["cell_area"][:], grid.geometry["cell_area"])
        assert np.allclose(dataset.variables["edge_length"][:], grid.geometry["edge_length"])


@pytest.mark.parametrize(
    "spec",
    [
        StretchedTorusGridSpec(nx=4, ny=3, edge_length=2.0, stretch_x=1.5, stretch_y=0.75),
        ChannelGridSpec(nx=3, ny=2, edge_length=2.0),
        ParallelogramGridSpec(nx=3, ny=2, edge_length=2.0, shear=0.35),
        RaggedOrthogonalGridSpec(nx=3, ny=2, dx=2.0, dy=1.0, raggedness=0.1),
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


def test_check_grid_reports_reversed_duplicate_edges():
    grid = generate_grid(ChannelGridSpec(nx=3, ny=2, edge_length=1.0))
    edges = grid.edges.copy()
    edges[1] = edges[0][::-1]
    broken = replace(grid, edges=edges)

    check = check_grid(broken)

    assert not check.ok
    assert "edges contain duplicate vertex pairs" in check.errors


def test_cell_centers_are_true_spherical_circumcenters():
    grid = generate_grid("R01B01")
    vertices = unit_rows(grid.vertices)
    centers = unit_rows(grid.cell_center_xyz)
    center_vertex_cosines = np.sum(vertices[grid.cells] * centers[:, np.newaxis, :], axis=2)

    assert np.allclose(center_vertex_cosines[:, 0], center_vertex_cosines[:, 1])
    assert np.allclose(center_vertex_cosines[:, 0], center_vertex_cosines[:, 2])

    independent_centers = np.cross(
        vertices[grid.cells][:, 0] - vertices[grid.cells][:, 1],
        vertices[grid.cells][:, 0] - vertices[grid.cells][:, 2],
    )
    independent_centers = unit_rows(independent_centers)
    reference = unit_rows(vertices[grid.cells].sum(axis=1))
    independent_centers = np.where(
        np.sum(independent_centers * reference, axis=1)[:, np.newaxis] < 0.0,
        -independent_centers,
        independent_centers,
    )

    assert np.allclose(centers, independent_centers)


def test_all_icon_grid_numeric_fields_are_finite_and_integer_indices_are_bounded():
    grid = generate_grid("R02B01", options={"radius": 3.0, "sphere_radius": 7.0})

    floating_arrays = [
        grid.vertices,
        grid.lon,
        grid.lat,
        grid.vertex_lon,
        grid.vertex_lat,
        grid.cell_center_xyz,
        grid.cell_vertex_lon,
        grid.cell_vertex_lat,
        grid.edge_center_xyz,
        grid.edge_lon,
        grid.edge_lat,
        *grid.geometry.values(),
        *grid.refinement.values(),
    ]
    for array in floating_arrays:
        assert np.all(np.isfinite(array))

    for array in [
        grid.cells,
        grid.edges,
        grid.cell_edges,
        grid.edge_cells,
        *grid.icon_connectivity.values(),
        *grid.connectivity.values(),
        *grid.neighbor_tables.values(),
        *grid.refinement.values(),
    ]:
        assert array.dtype == np.int32

    assert np.all((0 <= grid.cells) & (grid.cells < grid.dims["vertex"]))
    assert np.all((0 <= grid.edges) & (grid.edges < grid.dims["vertex"]))
    assert np.all((0 <= grid.cell_edges) & (grid.cell_edges < grid.dims["edge"]))
    assert np.all((0 <= grid.edge_cells) & (grid.edge_cells < grid.dims["cell"]))
    assert np.all((0 <= grid.icon_connectivity["c2c"]) & (grid.icon_connectivity["c2c"] < grid.dims["cell"]))
    assert np.all((0 <= grid.icon_connectivity["v2c"][grid.icon_connectivity["v2c"] > 0]) & (grid.icon_connectivity["v2c"][grid.icon_connectivity["v2c"] > 0] <= grid.dims["cell"]))
    assert np.all((0 <= grid.icon_connectivity["v2e"][grid.icon_connectivity["v2e"] > 0]) & (grid.icon_connectivity["v2e"][grid.icon_connectivity["v2e"] > 0] <= grid.dims["edge"]))
    assert np.all((0 <= grid.icon_connectivity["v2v"][grid.icon_connectivity["v2v"] > 0]) & (grid.icon_connectivity["v2v"][grid.icon_connectivity["v2v"] > 0] <= grid.dims["vertex"]))
    parent = generate_grid("R02B00")
    assert np.all(
        (1 <= grid.refinement["parent_cell_index"])
        & (grid.refinement["parent_cell_index"] <= parent.dims["cell"])
    )
    assert set(np.unique(grid.refinement["parent_cell_type"])) == {200, 201, 202, 203}
    assert np.all(
        (1 <= grid.refinement["parent_edge_index"])
        & (grid.refinement["parent_edge_index"] <= parent.dims["edge"])
    )
    assert set(np.unique(grid.refinement["edge_parent_type"])) == {101, 102, 201, 202, 203}
    assert np.all(grid.refinement["parent_vertex_index"] != 0)
    assert np.all(grid.refinement["parent_vertex_index"] <= parent.dims["vertex"])
    assert np.all(grid.refinement["parent_vertex_index"] >= -parent.dims["edge"])


def test_grid_topology_is_closed_triangular_and_eulerian():
    grid = generate_grid("R02B02")

    assert np.all(grid.cells[:, 0] != grid.cells[:, 1])
    assert np.all(grid.cells[:, 1] != grid.cells[:, 2])
    assert np.all(grid.cells[:, 2] != grid.cells[:, 0])
    assert np.all(grid.edge_cells >= 0)
    assert np.all(grid.edge_cells[:, 0] != grid.edge_cells[:, 1])
    assert grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] == 2

    unique_edges = {tuple(edge) for edge in grid.edges}
    assert len(unique_edges) == grid.dims["edge"]
    for cell_index, cell in enumerate(grid.cells):
        for local_index, pair in enumerate(
            ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0]))
        ):
            edge_index = grid.cell_edges[cell_index, local_index]
            assert set(map(int, pair)) == set(map(int, grid.edges[edge_index]))
            assert cell_index in grid.edge_cells[edge_index]


@pytest.mark.parametrize("grid_name", ["R01B00", "R01B01", "R02B01", "R02B02", "R03B00"])
def test_spherical_manifold_degree_structure_and_area_conservation_across_resolutions(grid_name):
    sphere_radius = 6_371_229.0
    grid = generate_grid(grid_name, options={"sphere_radius": sphere_radius})
    cell_area = grid.geometry["cell_area"]
    dual_area = grid.geometry["dual_area"]
    vertex_degrees = np.count_nonzero(grid.connectivity["vertices_of_vertex"] >= 0, axis=1)

    assert np.count_nonzero(vertex_degrees == 5) == 12
    assert np.count_nonzero(vertex_degrees == 6) == grid.dims["vertex"] - 12
    assert set(np.unique(vertex_degrees)) <= {5, 6}
    assert np.allclose(cell_area.sum(), 4.0 * math.pi * sphere_radius**2, rtol=2.0e-14)
    assert np.allclose(dual_area.sum(), cell_area.sum(), rtol=2.0e-14)
    assert np.max(cell_area) / np.min(cell_area) < 1.35
    assert np.max(dual_area) / np.min(dual_area) < 1.7
    assert np.max(grid.geometry["edge_length"]) / np.min(grid.geometry["edge_length"]) < 1.25
    assert len({tuple(edge) for edge in grid.edges}) == grid.dims["edge"]
    assert len({tuple(np.round(vertex / grid.options.radius, 14)) for vertex in grid.vertices}) == grid.dims["vertex"]


def test_r01b00_is_exact_regular_icosahedron_grid():
    sphere_radius = 12.0
    grid = generate_grid("R01B00", options={"sphere_radius": sphere_radius})
    expected_edge_angle = math.acos(1.0 / math.sqrt(5.0))
    expected_edge_length = sphere_radius * expected_edge_angle
    expected_cell_area = 4.0 * math.pi * sphere_radius**2 / 20.0
    expected_dual_area = 4.0 * math.pi * sphere_radius**2 / 12.0

    assert np.allclose(grid.geometry["edge_length"], expected_edge_length)
    assert np.allclose(grid.geometry["cell_area"], expected_cell_area)
    assert np.allclose(grid.geometry["dual_area"], expected_dual_area)
    assert np.allclose(grid.geometry["cell_area"], grid.geometry["cell_area"][0])
    assert np.allclose(grid.geometry["dual_area"], grid.geometry["dual_area"][0])
    assert np.allclose(grid.geometry["edge_length"], grid.geometry["edge_length"][0])
    assert np.all(np.count_nonzero(grid.connectivity["cells_of_vertex"] >= 0, axis=1) == 5)
    assert np.all(np.count_nonzero(grid.connectivity["edges_of_vertex"] >= 0, axis=1) == 5)
    assert np.all(np.count_nonzero(grid.connectivity["vertices_of_vertex"] >= 0, axis=1) == 5)


def test_icon_connectivity_public_connectivity_and_neighbor_tables_are_consistent():
    grid = generate_grid("R01B01")
    icon = grid.icon_connectivity
    public = grid.connectivity
    tables = grid.neighbor_tables

    assert set(icon) == {
        "c2e",
        "c2c",
        "v2c",
        "v2e",
        "v2v",
        "orientation_of_normal",
        "edge_orientation",
    }
    assert set(public) == {
        "edge_of_cell",
        "vertex_of_cell",
        "neighbor_cell_index",
        "adjacent_cell_of_edge",
        "edge_vertices",
        "cells_of_vertex",
        "edges_of_vertex",
        "vertices_of_vertex",
    }
    assert set(tables) == {"c2e2c", "c2e", "e2c", "v2e", "v2c", "c2v", "v2e2v", "e2v"}

    assert np.array_equal(public["edge_of_cell"], icon["c2e"])
    assert np.array_equal(public["vertex_of_cell"], grid.cells)
    assert np.array_equal(public["neighbor_cell_index"], icon["c2c"])
    assert np.array_equal(public["adjacent_cell_of_edge"], grid.edge_cells)
    assert np.array_equal(public["edge_vertices"], grid.edges)
    assert np.array_equal(tables["c2e2c"], icon["c2c"])
    assert np.array_equal(tables["c2e"], icon["c2e"])
    assert np.array_equal(tables["e2c"], grid.edge_cells)
    assert np.array_equal(tables["c2v"], grid.cells)
    assert np.array_equal(tables["e2v"], grid.edges)
    assert np.array_equal(public["cells_of_vertex"], tables["v2c"])
    assert np.array_equal(public["edges_of_vertex"], tables["v2e"])
    assert np.array_equal(public["vertices_of_vertex"], tables["v2e2v"])

    assert set(np.unique(icon["orientation_of_normal"])) == {-1, 1}
    assert set(np.unique(icon["edge_orientation"])) == {-1, 0, 1}
    assert set(np.unique(public["cells_of_vertex"])) == set(range(-1, grid.dims["cell"]))
    assert np.any(public["cells_of_vertex"] == -1)

    for cell_index, edge_indices in enumerate(icon["c2e"]):
        for local_index, edge_index in enumerate(edge_indices):
            neighbors = grid.edge_cells[edge_index]
            expected_neighbor = neighbors[1] if neighbors[0] == cell_index else neighbors[0]
            assert icon["c2c"][cell_index, local_index] == expected_neighbor
            assert icon["orientation_of_normal"][cell_index, local_index] in {-1, 1}
            expected_orientation = 1 if neighbors[0] == cell_index else -1
            assert icon["orientation_of_normal"][cell_index, local_index] == expected_orientation
            assert set(grid.cells[cell_index]) & set(grid.cells[expected_neighbor]) == set(
                grid.edges[edge_index]
            )

    for vertex_index in range(grid.dims["vertex"]):
        incident_cells = {int(c) for c in public["cells_of_vertex"][vertex_index] if c >= 0}
        incident_edges = {int(e) for e in public["edges_of_vertex"][vertex_index] if e >= 0}
        incident_vertices = {int(v) for v in public["vertices_of_vertex"][vertex_index] if v >= 0}

        assert incident_cells == {
            cell_index for cell_index, cell in enumerate(grid.cells) if vertex_index in cell
        }
        assert incident_edges == {
            edge_index for edge_index, edge in enumerate(grid.edges) if vertex_index in edge
        }
        assert incident_vertices == {
            int(other)
            for edge in grid.edges
            if vertex_index in edge
            for other in edge
            if int(other) != vertex_index
        }


def test_cell_neighbor_relations_are_symmetric_across_shared_edges():
    grid = generate_grid("R02B01")

    for cell_index, neighbor_indices in enumerate(grid.icon_connectivity["c2c"]):
        for local_index, neighbor_index in enumerate(neighbor_indices):
            edge_index = grid.cell_edges[cell_index, local_index]
            neighbor_edge_ids = set(grid.cell_edges[neighbor_index])
            assert edge_index in neighbor_edge_ids
            assert cell_index in grid.icon_connectivity["c2c"][neighbor_index]
            assert set(grid.cells[cell_index]) & set(grid.cells[neighbor_index]) == set(
                grid.edges[edge_index]
            )


def test_vertex_sparse_tables_are_consistent_with_edges_cells_and_orientation():
    grid = generate_grid("R02B02")

    for vertex_index in range(grid.dims["vertex"]):
        row_edges = grid.connectivity["edges_of_vertex"][vertex_index]
        row_vertices = grid.connectivity["vertices_of_vertex"][vertex_index]
        row_cells = grid.connectivity["cells_of_vertex"][vertex_index]
        row_orientation = grid.icon_connectivity["edge_orientation"][vertex_index]

        active_edges = row_edges[row_edges >= 0]
        active_vertices = row_vertices[row_vertices >= 0]
        active_cells = row_cells[row_cells >= 0]
        assert len(active_edges) in {5, 6}
        assert len(active_edges) == len(active_vertices)
        assert len(active_edges) == len(active_cells)
        assert len(set(active_edges)) == len(active_edges)
        assert len(set(active_vertices)) == len(active_vertices)
        assert len(set(active_cells)) == len(active_cells)

        expected_neighbor_vertices = set()
        for pos, edge_index in enumerate(row_edges):
            if edge_index < 0:
                assert row_vertices[pos] == -1
                assert row_cells[pos] == -1
                assert row_orientation[pos] == 0
                continue
            edge = grid.edges[edge_index]
            assert vertex_index in edge
            expected_neighbor_vertices.add(int(edge[0] if edge[1] == vertex_index else edge[1]))
            assert row_orientation[pos] == (1 if edge[0] == vertex_index else -1)

        assert set(int(vertex) for vertex in active_vertices) == expected_neighbor_vertices
        for cell_index in active_cells:
            assert vertex_index in grid.cells[cell_index]


def test_geometry_metric_fields_are_positive_scaled_and_conservative():
    sphere_radius = 2.5
    grid = generate_grid("R01B01", options={"sphere_radius": sphere_radius})
    geometry = grid.geometry

    assert set(geometry) == {
        "cell_area",
        "dual_area",
        "edge_length",
        "dual_edge_length",
        "edge_cell_distance",
        "edge_vert_distance",
        "orientation_of_normal",
        "edge_system_orientation",
        "edge_orientation",
        "edgequad_area",
        "edge_primal_normal_cartesian",
        "edge_dual_normal_cartesian",
        "zonal_normal_primal_edge",
        "meridional_normal_primal_edge",
        "zonal_normal_dual_edge",
        "meridional_normal_dual_edge",
    }
    assert geometry["cell_area"].shape == (80,)
    assert geometry["dual_area"].shape == (42,)
    assert geometry["edge_length"].shape == (120,)
    assert geometry["dual_edge_length"].shape == (120,)
    assert geometry["edge_cell_distance"].shape == (120, 2)
    assert geometry["edge_vert_distance"].shape == (120, 2)
    assert geometry["edge_primal_normal_cartesian"].shape == (120, 3)
    assert geometry["edge_dual_normal_cartesian"].shape == (120, 3)
    assert np.all(geometry["cell_area"] > 0.0)
    assert np.all(geometry["dual_area"] > 0.0)
    assert np.all(geometry["edge_length"] > 0.0)
    assert np.all(geometry["dual_edge_length"] > 0.0)
    assert np.all(geometry["edge_cell_distance"] > 0.0)
    assert np.allclose(geometry["cell_area"].sum(), 4.0 * math.pi * sphere_radius**2)
    assert np.allclose(geometry["dual_area"].sum(), geometry["cell_area"].sum())
    assert np.allclose(geometry["edge_vert_distance"][:, 0], geometry["edge_length"] * 0.5)
    assert np.allclose(geometry["edge_vert_distance"][:, 1], geometry["edge_length"] * 0.5)
    assert np.allclose(
        geometry["edgequad_area"],
        0.5 * geometry["edge_length"] * geometry["dual_edge_length"],
    )
    assert np.array_equal(
        geometry["orientation_of_normal"],
        grid.icon_connectivity["orientation_of_normal"],
    )
    assert np.array_equal(geometry["edge_orientation"], grid.icon_connectivity["edge_orientation"])
    assert np.array_equal(
        geometry["edge_system_orientation"],
        expected_edge_system_orientation(grid),
    )
    assert set(np.unique(geometry["edge_system_orientation"])) <= {-1, 1}
    assert np.allclose(np.linalg.norm(geometry["edge_primal_normal_cartesian"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(geometry["edge_dual_normal_cartesian"], axis=1), 1.0)
    assert np.allclose(
        np.sum(
            geometry["edge_primal_normal_cartesian"]
            * geometry["edge_dual_normal_cartesian"],
            axis=1,
        ),
        0.0,
    )


def test_dual_areas_are_geometric_dual_cell_areas():
    sphere_radius = 3.0
    grid = generate_grid("R02B02", options={"sphere_radius": sphere_radius})
    expected_dual_area = geometric_dual_areas_from_cell_centers(grid, sphere_radius)

    assert np.allclose(grid.geometry["dual_area"], expected_dual_area)
    assert np.allclose(grid.geometry["dual_area"].sum(), grid.geometry["cell_area"].sum())


def test_edge_vectors_are_tangent_and_match_local_zonal_meridional_components():
    grid = generate_grid("R02B02")
    geometry = grid.geometry
    edge_centers = unit_rows(grid.edge_center_xyz)
    east, north = local_east_north(edge_centers)
    primal = geometry["edge_primal_normal_cartesian"]
    dual = geometry["edge_dual_normal_cartesian"]

    reconstructed_primal = (
        geometry["zonal_normal_primal_edge"][:, np.newaxis] * east
        + geometry["meridional_normal_primal_edge"][:, np.newaxis] * north
    )
    reconstructed_dual = (
        geometry["zonal_normal_dual_edge"][:, np.newaxis] * east
        + geometry["meridional_normal_dual_edge"][:, np.newaxis] * north
    )

    assert np.allclose(np.sum(primal * edge_centers, axis=1), 0.0)
    assert np.allclose(np.sum(dual * edge_centers, axis=1), 0.0)
    assert np.allclose(reconstructed_primal, primal)
    assert np.allclose(reconstructed_dual, dual)
    assert np.allclose(primal, unit_rows(np.cross(edge_centers, dual)))


def test_edge_system_orientation_makes_normals_point_from_first_to_second_cell():
    grid = generate_grid("R02B02")
    vertices = unit_rows(grid.vertices)
    centers = unit_rows(grid.cell_center_xyz)
    edge_centers = unit_rows(grid.edge_center_xyz)
    tangent = (
        grid.geometry["edge_system_orientation"][:, np.newaxis]
        * (vertices[grid.edges[:, 1]] - vertices[grid.edges[:, 0]])
    )
    tangent = unit_rows(tangent)
    normal = unit_rows(np.cross(edge_centers, tangent))
    first_to_second_cell = centers[grid.edge_cells[:, 1]] - centers[grid.edge_cells[:, 0]]

    assert np.all(np.sum(normal * first_to_second_cell, axis=1) > 0.0)


def test_global_convention_keeps_positive_edge_system_orientation():
    grid = generate_grid("R02B04", options={"spring_iterations": 1})
    vertices = unit_rows(grid.vertices)
    centers = unit_rows(grid.cell_center_xyz)
    edge_centers = unit_rows(grid.edge_center_xyz)
    tangent = unit_rows(vertices[grid.edges[:, 1]] - vertices[grid.edges[:, 0]])
    normal = unit_rows(np.cross(edge_centers, tangent))
    first_to_second_cell = centers[grid.edge_cells[:, 1]] - centers[grid.edge_cells[:, 0]]

    assert np.array_equal(
        grid.geometry["edge_system_orientation"],
        np.ones(grid.dims["edge"], dtype=np.int32),
    )
    assert set(np.unique(grid.icon_connectivity["orientation_of_normal"])) == {-1, 1}
    assert np.all(np.sum(normal * first_to_second_cell, axis=1) > 0.0)


def test_shifted_pole_rotation_matrix_matches_contract():
    options = IconGridOptions(
        north_pole_lon=15.0,
        north_pole_lat=75.0,
        rotation_angle_degrees=37.5,
    ).global_grid

    assert np.allclose(
        gg._global_grid_rotation_matrix(options),
        np.array(
            [
                [0.88088350, -0.36914665, 0.29626846],
                [0.38098700, 0.92438507, 0.01899687],
                [-0.28087877, 0.09614051, 0.95491576],
            ],
        ),
        atol=1.0e-8,
    )


def test_r02b03_refinement_parent_fields_match_previous_bisection_sizes():
    grid = generate_grid("R02B03")
    parent = generate_grid("R02B02")
    refinement = grid.refinement

    assert refinement["parent_cell_index"].shape == (grid.dims["cell"],)
    assert refinement["parent_cell_type"].shape == (grid.dims["cell"],)
    assert refinement["parent_edge_index"].shape == (grid.dims["edge"],)
    assert refinement["edge_parent_type"].shape == (grid.dims["edge"],)
    assert refinement["parent_vertex_index"].shape == (grid.dims["vertex"],)
    assert set(np.unique(refinement["parent_cell_type"])) == {200, 201, 202, 203}
    assert set(np.unique(refinement["edge_parent_type"])) == {101, 102, 201, 202, 203}
    assert np.array_equal(
        np.unique(refinement["parent_cell_index"]),
        np.arange(1, parent.dims["cell"] + 1, dtype=np.int32),
    )
    assert np.array_equal(
        np.unique(refinement["parent_edge_index"]),
        np.arange(1, parent.dims["edge"] + 1, dtype=np.int32),
    )
    assert np.max(refinement["parent_vertex_index"]) == parent.dims["vertex"]
    assert np.min(refinement["parent_vertex_index"]) == -parent.dims["edge"]
    assert len(np.unique(refinement["parent_vertex_index"])) == grid.dims["vertex"]


def test_global_bisection_uses_structural_parent_provenance(monkeypatch):
    def fail_coordinate_parent_lookup(*args, **kwargs):
        raise AssertionError("coordinate parent lookup should not be used")

    monkeypatch.setattr(gg, "_parent_vertex_indices", fail_coordinate_parent_lookup)

    grid = generate_grid("R02B03")

    assert grid.refinement["parent_vertex_index"].shape == (grid.dims["vertex"],)
    assert set(np.unique(grid.refinement["parent_cell_type"])) == {200, 201, 202, 203}
    assert set(np.unique(grid.refinement["edge_parent_type"])) == {101, 102, 201, 202, 203}


def test_geofac_n2s_coefficients_are_diffusive_for_topography_smoothing():
    grid = generate_grid("R02B02")
    geometry = grid.geometry
    c2e = grid.neighbor_tables["c2e"]
    e2c = grid.neighbor_tables["e2c"]
    c2e2c = grid.neighbor_tables["c2e2c"]
    cells = np.arange(grid.dims["cell"])[:, np.newaxis]

    geofac_div = (
        geometry["edge_length"][c2e]
        * geometry["orientation_of_normal"]
        / geometry["cell_area"][:, np.newaxis]
    )
    scaled_geofac = geofac_div / geometry["dual_edge_length"][c2e]
    geofac_n2s = np.zeros((grid.dims["cell"], 4), dtype=np.float64)

    geofac_n2s[:, 0] -= np.sum((e2c[c2e, 0] == cells) * scaled_geofac, axis=1)
    geofac_n2s[:, 0] += np.sum((e2c[c2e, 1] == cells) * scaled_geofac, axis=1)
    geofac_n2s[:, 1:] -= (e2c[c2e, 0] == c2e2c) * scaled_geofac
    geofac_n2s[:, 1:] += (e2c[c2e, 1] == c2e2c) * scaled_geofac

    smoothing_weights = 0.125 * geometry["cell_area"][:, np.newaxis] * geofac_n2s
    assert np.allclose(smoothing_weights.sum(axis=1), 0.0)
    assert np.all(smoothing_weights[:, 0] < 0.0)
    assert np.all(smoothing_weights[:, 1:] > 0.0)


def test_metric_fields_match_independent_spherical_recomputation():
    sphere_radius = 3.5
    grid = generate_grid("R01B01", options={"sphere_radius": sphere_radius})
    vertices = unit_rows(grid.vertices)
    centers = unit_rows(grid.cell_center_xyz)
    edge_centers = unit_rows(grid.edge_center_xyz)

    expected_edge_lengths = np.arccos(
        np.clip(
            np.sum(vertices[grid.edges][:, 0] * vertices[grid.edges][:, 1], axis=1),
            -1.0,
            1.0,
        )
    ) * sphere_radius
    expected_dual_edge_lengths = np.arccos(
        np.clip(
            np.sum(centers[grid.edge_cells][:, 0] * centers[grid.edge_cells][:, 1], axis=1),
            -1.0,
            1.0,
        )
    ) * sphere_radius
    expected_edge_cell_distances = np.arccos(
        np.clip(
            np.sum(centers[grid.edge_cells] * edge_centers[:, np.newaxis, :], axis=2),
            -1.0,
            1.0,
        )
    ) * sphere_radius
    expected_cell_areas = spherical_triangle_areas_lhuilier(grid.vertices, grid.cells, sphere_radius)
    expected_dual_area = geometric_dual_areas_from_cell_centers(grid, sphere_radius)

    assert np.allclose(grid.geometry["cell_area"], expected_cell_areas)
    assert np.allclose(grid.geometry["edge_length"], expected_edge_lengths)
    assert np.allclose(grid.geometry["dual_edge_length"], expected_dual_edge_lengths)
    assert np.allclose(grid.geometry["edge_cell_distance"], expected_edge_cell_distances)
    assert np.allclose(grid.geometry["dual_area"], expected_dual_area)


def test_geometry_scales_with_sphere_radius_squared_for_areas_and_linearly_for_lengths():
    small = generate_grid("R01B01", options={"sphere_radius": 2.0})
    large = generate_grid("R01B01", options={"sphere_radius": 5.0})
    area_scale = (5.0 / 2.0) ** 2
    length_scale = 5.0 / 2.0

    assert np.array_equal(small.cells, large.cells)
    assert np.array_equal(small.edges, large.edges)
    assert np.allclose(large.geometry["cell_area"], small.geometry["cell_area"] * area_scale)
    assert np.allclose(large.geometry["dual_area"], small.geometry["dual_area"] * area_scale)
    assert np.allclose(large.geometry["edgequad_area"], small.geometry["edgequad_area"] * area_scale)
    assert np.allclose(large.geometry["edge_length"], small.geometry["edge_length"] * length_scale)
    assert np.allclose(
        large.geometry["dual_edge_length"],
        small.geometry["dual_edge_length"] * length_scale,
    )
    assert np.allclose(
        large.geometry["edge_cell_distance"],
        small.geometry["edge_cell_distance"] * length_scale,
    )


def test_geometry_is_independent_of_display_radius_except_cartesian_scaling():
    unit_grid = generate_grid(
        "R01B01",
        options={
            "radius": 1.0,
            "sphere_radius": 4.0,
        },
    )
    scaled_grid = generate_grid(
        "R01B01",
        options={
            "radius": 10.0,
            "sphere_radius": 4.0,
        },
    )

    assert np.array_equal(unit_grid.cells, scaled_grid.cells)
    assert np.array_equal(unit_grid.edges, scaled_grid.edges)
    assert np.array_equal(unit_grid.cell_edges, scaled_grid.cell_edges)
    assert np.array_equal(unit_grid.edge_cells, scaled_grid.edge_cells)
    assert np.allclose(scaled_grid.vertices, unit_grid.vertices * 10.0)
    assert np.allclose(scaled_grid.cell_center_xyz, unit_grid.cell_center_xyz * 10.0)
    assert np.allclose(scaled_grid.edge_center_xyz, unit_grid.edge_center_xyz * 10.0)
    assert np.allclose(lon_unit_circle(scaled_grid.lon), lon_unit_circle(unit_grid.lon))
    assert np.allclose(scaled_grid.lat, unit_grid.lat)
    nonpolar_vertices = np.abs(unit_grid.vertex_lat) < 89.999
    assert np.allclose(
        lon_unit_circle(scaled_grid.vertex_lon[nonpolar_vertices]),
        lon_unit_circle(unit_grid.vertex_lon[nonpolar_vertices]),
    )
    assert np.allclose(scaled_grid.vertex_lat, unit_grid.vertex_lat)
    assert np.allclose(
        lon_unit_circle(scaled_grid.edge_lon),
        lon_unit_circle(unit_grid.edge_lon),
    )
    assert np.allclose(scaled_grid.edge_lat, unit_grid.edge_lat)
    for key in unit_grid.geometry:
        assert np.allclose(scaled_grid.geometry[key], unit_grid.geometry[key])


def test_metadata_uses_stable_uuid_and_metric_means():
    grid = generate_grid(
        "R02B01",
        options={"sphere_radius": 9.0},
    )
    rotated = generate_grid(
        "R02B01",
        options={
            "sphere_radius": 9.0,
            "rotation_angle_degrees": 0.05,
        },
    )
    display_scaled = generate_grid(
        "R02B01",
        options={
            "radius": 2.0,
            "sphere_radius": 9.0,
        },
    )
    metadata = grid.metadata

    assert metadata["uuidOfHGrid"] == gg.grid_uuid("R02B01", sphere_radius=9.0)
    assert metadata["uuidOfHGrid"] == gg.grid_uuid("r2b1", sphere_radius=9.0)
    assert metadata["uuidOfHGrid"] == display_scaled.metadata["uuidOfHGrid"]
    assert metadata["uuidOfHGrid"] != gg.grid_uuid("R02B01")
    assert metadata["uuidOfHGrid"] != rotated.metadata["uuidOfHGrid"]
    assert metadata["uuidOfParHGrid"] == "00000000-0000-0000-0000-000000000000"
    assert metadata["grid_root"] == 2
    assert metadata["grid_level"] == 1
    assert metadata["sphere_radius"] == 9.0
    assert metadata["semi_major_axis"] == 9.0
    assert metadata["inverse_flattening"] == 0.0
    assert metadata["grid_geometry"] == 1
    assert metadata["grid_cell_type"] == 3
    assert metadata["number_of_grid_used"] == 0
    assert metadata["center"] == 78
    assert metadata["subcenter"] == 255
    assert metadata["crs_id"] == 0
    assert metadata["crs_name"] == "Spherical Earth"
    assert metadata["grid_mapping_name"] == "lat_long_on_sphere"
    assert metadata["spring_beta"] == 0.9
    assert metadata["spring_maxit"] == 2000
    assert metadata["indexing_algorithm"] == "new"
    assert metadata["ellipsoid_name"] == "sphere"
    assert metadata["mean_edge_length"] == pytest.approx(np.mean(grid.geometry["edge_length"]))
    assert metadata["mean_dual_edge_length"] == pytest.approx(
        np.mean(grid.geometry["dual_edge_length"])
    )
    assert metadata["mean_cell_area"] == pytest.approx(np.mean(grid.geometry["cell_area"]))
    assert metadata["mean_dual_cell_area"] == pytest.approx(np.mean(grid.geometry["dual_area"]))


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


def test_to_netcdf_writes_expected_icon_grid_content(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid("R01B00")
    path = grid.to_netcdf(tmp_path / "nested" / "r01b00.nc")

    assert path == tmp_path / "nested" / "r01b00.nc"
    assert path.exists()

    second_path = grid.to_netcdf(tmp_path / "r01b00-via-method.nc")
    assert second_path.exists()

    with netcdf4.Dataset(path) as dataset:
        assert dataset.dimensions["cell"].size == 20
        assert dataset.dimensions["vertex"].size == 12
        assert dataset.dimensions["edge"].size == 30
        assert dataset.dimensions["nc"].size == 2
        assert dataset.dimensions["nv"].size == 3
        assert dataset.dimensions["ne"].size == 6
        assert dataset.dimensions["no"].size == 4
        assert dataset.dimensions["max_chdom"].size == 1
        assert dataset.dimensions["cell_grf"].size == 14
        assert dataset.dimensions["edge_grf"].size == 24
        assert dataset.dimensions["vert_grf"].size == 13
        assert dataset.getncattr("title") == "Pure Python ICON grid R01B00"
        assert dataset.getncattr("source") == "grid_generator Python ICON grid generator"
        assert dataset.getncattr("uuidOfHGrid") == grid.metadata["uuidOfHGrid"]
        assert dataset.getncattr("grid_root") == 1
        assert dataset.getncattr("grid_level") == 0
        assert dataset.getncattr("sphere_radius") == grid.options.sphere_radius
        assert dataset.getncattr("grid_ID") == 1
        assert dataset.getncattr("parent_grid_ID") == 0
        assert dataset.getncattr("no_of_subgrids") == 1
        assert dataset.getncattr("start_subgrid_id") == 0
        assert dataset.getncattr("max_childdom") == 1
        assert dataset.getncattr("boundary_depth_index") == 0
        assert np.array_equal(dataset.getncattr("rotation_vector"), np.zeros(3))
        assert np.array_equal(dataset.getncattr("domain_cartesian_center"), np.zeros(3))
        assert dataset.getncattr("domain_length") == pytest.approx(2.0 * np.pi * grid.options.sphere_radius)
        assert dataset.getncattr("domain_height") == pytest.approx(2.0 * np.pi * grid.options.sphere_radius)
        for attr in ("revision", "history", "date", "user_name", "os_name"):
            assert attr in dataset.ncattrs()
        assert dataset.variables["clon"].dimensions == ("cell",)
        assert dataset.variables["edge_of_cell"].dimensions == ("nv", "cell")
        assert dataset.variables["adjacent_cell_of_edge"].dimensions == ("nc", "edge")
        assert dataset.variables["cells_of_vertex"].dimensions == ("ne", "vertex")
        assert dataset.variables["child_edge_index"].dimensions == ("no", "edge")
        assert dataset.variables["elon_vertices"].dimensions == ("edge", "no")
        assert dataset.variables["elat_vertices"].dimensions == ("edge", "no")
        assert {
            "clon",
            "clat",
            "vlon",
            "vlat",
            "elon",
            "elat",
            "elon_vertices",
            "elat_vertices",
            "edge_of_cell",
            "vertex_of_cell",
            "neighbor_cell_index",
            "adjacent_cell_of_edge",
            "edge_vertices",
            "cells_of_vertex",
            "edges_of_vertex",
            "vertices_of_vertex",
            "cell_area",
            "dual_area",
            "edge_length",
            "dual_edge_length",
            "edge_cell_distance",
            "edge_vert_distance",
            "edgequad_area",
            "orientation_of_normal",
            "edge_system_orientation",
            "edge_orientation",
            "cell_circumcenter_cartesian_x",
            "cell_circumcenter_cartesian_y",
            "cell_circumcenter_cartesian_z",
            "edge_middle_cartesian_x",
            "edge_middle_cartesian_y",
            "edge_middle_cartesian_z",
            "edge_primal_normal_cartesian_x",
            "edge_primal_normal_cartesian_y",
            "edge_primal_normal_cartesian_z",
            "edge_dual_normal_cartesian_x",
            "edge_dual_normal_cartesian_y",
            "edge_dual_normal_cartesian_z",
            "zonal_normal_primal_edge",
            "meridional_normal_primal_edge",
            "zonal_normal_dual_edge",
            "meridional_normal_dual_edge",
        } <= set(dataset.variables)
        assert np.allclose(dataset.variables["clon"][:], np.radians(grid.lon))
        assert np.allclose(dataset.variables["vlon"][:], np.radians(grid.vertex_lon))
        assert np.allclose(dataset.variables["elon"][:], np.radians(grid.edge_lon))
        assert dataset.variables["elon_vertices"].shape == (30, 4)
        assert dataset.variables["elat_vertices"].shape == (30, 4)
        assert np.all(np.isfinite(dataset.variables["elon_vertices"][:]))
        assert np.all(np.isfinite(dataset.variables["elat_vertices"][:]))
        for variable_name, expected_attrs in gg.ICON_VARIABLE_ATTRS.items():
            variable = dataset.variables[variable_name]
            assert set(expected_attrs) <= set(variable.ncattrs())
            for attr_name, attr_value in expected_attrs.items():
                assert variable.getncattr(attr_name) == attr_value
        assert np.array_equal(dataset.variables["edge_of_cell"][:], grid.cell_edges.T + 1)
        assert np.array_equal(dataset.variables["vertex_of_cell"][:], grid.cells.T + 1)
        assert np.array_equal(dataset.variables["adjacent_cell_of_edge"][:], grid.edge_cells.T + 1)
        assert np.array_equal(dataset.variables["edge_vertices"][:], grid.edges.T + 1)
        assert np.allclose(dataset.variables["cell_area"][:], grid.geometry["cell_area"])
        assert np.allclose(dataset.variables["dual_area"][:], grid.geometry["dual_area"])
        assert np.allclose(dataset.variables["edge_length"][:], grid.geometry["edge_length"])
        assert np.allclose(dataset.variables["dual_edge_length"][:], grid.geometry["dual_edge_length"])
        assert np.allclose(
            dataset.variables["edge_cell_distance"][:],
            grid.geometry["edge_cell_distance"].T,
        )
        assert np.allclose(
            dataset.variables["edge_vert_distance"][:],
            grid.geometry["edge_vert_distance"].T,
        )
        assert np.allclose(
            dataset.variables["edgequad_area"][:],
            grid.geometry["edgequad_area"] / grid.options.sphere_radius**2,
        )
        assert np.array_equal(
            dataset.variables["orientation_of_normal"][:],
            grid.geometry["orientation_of_normal"].T,
        )
        assert np.array_equal(
            dataset.variables["edge_system_orientation"][:],
            grid.geometry["edge_system_orientation"],
        )
        assert np.array_equal(
            dataset.variables["edge_orientation"][:],
            grid.geometry["edge_orientation"].T,
        )
        unit_centers = unit_rows(grid.cell_center_xyz)
        unit_edge_centers = unit_rows(grid.edge_center_xyz)
        assert np.allclose(dataset.variables["cell_circumcenter_cartesian_x"][:], unit_centers[:, 0])
        assert np.allclose(dataset.variables["cell_circumcenter_cartesian_y"][:], unit_centers[:, 1])
        assert np.allclose(dataset.variables["cell_circumcenter_cartesian_z"][:], unit_centers[:, 2])
        assert np.allclose(dataset.variables["edge_middle_cartesian_x"][:], unit_edge_centers[:, 0])
        assert np.allclose(dataset.variables["edge_middle_cartesian_y"][:], unit_edge_centers[:, 1])
        assert np.allclose(dataset.variables["edge_middle_cartesian_z"][:], unit_edge_centers[:, 2])
        assert np.allclose(
            dataset.variables["edge_primal_normal_cartesian_x"][:],
            grid.geometry["edge_primal_normal_cartesian"][:, 0],
        )
        assert np.allclose(
            dataset.variables["edge_primal_normal_cartesian_y"][:],
            grid.geometry["edge_primal_normal_cartesian"][:, 1],
        )
        assert np.allclose(
            dataset.variables["edge_primal_normal_cartesian_z"][:],
            grid.geometry["edge_primal_normal_cartesian"][:, 2],
        )
        assert np.allclose(
            dataset.variables["edge_dual_normal_cartesian_x"][:],
            grid.geometry["edge_dual_normal_cartesian"][:, 0],
        )
        assert np.allclose(
            dataset.variables["edge_dual_normal_cartesian_y"][:],
            grid.geometry["edge_dual_normal_cartesian"][:, 1],
        )
        assert np.allclose(
            dataset.variables["edge_dual_normal_cartesian_z"][:],
            grid.geometry["edge_dual_normal_cartesian"][:, 2],
        )
        assert np.allclose(
            dataset.variables["zonal_normal_primal_edge"][:],
            grid.geometry["zonal_normal_primal_edge"],
        )
        assert np.allclose(
            dataset.variables["meridional_normal_primal_edge"][:],
            grid.geometry["meridional_normal_primal_edge"],
        )
        assert np.allclose(
            dataset.variables["zonal_normal_dual_edge"][:],
            grid.geometry["zonal_normal_dual_edge"],
        )
        assert np.allclose(
            dataset.variables["meridional_normal_dual_edge"][:],
            grid.geometry["meridional_normal_dual_edge"],
        )
        assert np.array_equal(dataset.variables["refin_c_ctrl"][:], np.full(20, -4))
        assert np.array_equal(dataset.variables["refin_e_ctrl"][:], np.full(30, -8))
        assert np.array_equal(dataset.variables["refin_v_ctrl"][:], np.zeros(12, dtype=np.int32))
        for name, values in grid.refinement.items():
            assert np.array_equal(dataset.variables[name][:], values)
        assert np.array_equal(
            dataset.variables["start_idx_c"][:],
            np.array([[21] * 9 + [1] * 5], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["end_idx_c"][:],
            np.array([[20] * 9 + [0] * 5], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["start_idx_e"][:],
            np.array([[31] * 14 + [1] * 10], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["end_idx_e"][:],
            np.array([[30] * 14 + [0] * 10], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["start_idx_v"][:],
            np.array([[13] * 8 + [1] * 5], dtype=np.int32),
        )
        assert np.array_equal(
            dataset.variables["end_idx_v"][:],
            np.array([[12] * 8 + [0] * 5], dtype=np.int32),
        )


def test_to_netcdf_rejects_radius_mismatch(tmp_path):
    grid = generate_grid("R01B00", options={"sphere_radius": 2.0})
    with pytest.raises(ValueError, match="sphere_radius must match"):
        grid.to_netcdf(tmp_path / "wrong-radius.nc", sphere_radius=3.0)


def test_to_netcdf_reports_missing_netcdf4(monkeypatch, tmp_path):
    grid = generate_grid("R01B00")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "netCDF4":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="NetCDF export requires"):
        grid.to_netcdf(tmp_path / "grid.nc")


def test_safety_cap_fails_clearly_and_can_be_changed_or_disabled():
    with pytest.raises(ValueError, match="exceeding max_cells"):
        generate_grid("R02B02", options={"max_cells": 10})

    assert generate_grid("R02B02", options={"max_cells": 2_000}).dims["cell"] == 1280
    assert IconGridOptions().max_cells == 2_000_000
    assert generate_grid("R01B01", options={"max_cells": None}).dims["cell"] == 80


def test_global_grid_generation_rejects_int32_index_overflow():
    with pytest.raises(ValueError, match="int32 index limit"):
        generate_grid("R02B13", options={"max_cells": None})


@pytest.mark.parametrize(
    "grid_name",
    [
        "R01B00",
        "R01B01",
        "R01B02",
        "R01B03",
        "R01B04",
        "R02B00",
        "R02B01",
        "R02B02",
        "R02B03",
        "R03B00",
        "R03B01",
        "R04B00",
        "R04B01",
    ],
)
def test_representative_grid_series_sanity(grid_name):
    sphere_radius = 6_371_229.0
    grid = generate_grid(grid_name, options={"sphere_radius": sphere_radius, "max_cells": 250_000})
    expected_dims = {
        "cell": grid.spec.expected_cells,
        "edge": grid.spec.expected_edges,
        "vertex": grid.spec.expected_vertices,
    }
    vertex_degrees = np.count_nonzero(grid.connectivity["vertices_of_vertex"] >= 0, axis=1)
    cell_area = grid.geometry["cell_area"]
    dual_area = grid.geometry["dual_area"]
    unit_vertices = unit_rows(grid.vertices)
    unit_centers = unit_rows(grid.cell_center_xyz)
    center_vertex_cosines = np.sum(unit_vertices[grid.cells] * unit_centers[:, np.newaxis, :], axis=2)

    assert grid.dims == expected_dims
    assert grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] == 2
    assert np.all(grid.edge_cells >= 0)
    assert np.all(grid.edge_cells[:, 0] != grid.edge_cells[:, 1])
    assert np.count_nonzero(vertex_degrees == 5) == 12
    assert np.count_nonzero(vertex_degrees == 6) == grid.dims["vertex"] - 12
    assert set(np.unique(vertex_degrees)) <= {5, 6}
    assert np.allclose(cell_area.sum(), 4.0 * math.pi * sphere_radius**2, rtol=5.0e-13)
    assert np.allclose(dual_area.sum(), cell_area.sum(), rtol=5.0e-15)
    assert np.allclose(
        np.max(center_vertex_cosines, axis=1),
        np.min(center_vertex_cosines, axis=1),
        atol=1.0e-12,
    )
    assert np.max(cell_area) / np.min(cell_area) < 1.6
    assert np.max(dual_area) / np.min(dual_area) < 1.9
    assert np.max(grid.geometry["edge_length"]) / np.min(grid.geometry["edge_length"]) < 1.35
    assert set(np.unique(grid.geometry["orientation_of_normal"])) <= {-1, 1}
    assert set(np.unique(grid.geometry["edge_orientation"])) <= {-1, 0, 1}

    for array in [
        grid.vertices,
        grid.cell_center_xyz,
        grid.edge_center_xyz,
        grid.lon,
        grid.lat,
        grid.vertex_lon,
        grid.vertex_lat,
        grid.edge_lon,
        grid.edge_lat,
        *grid.geometry.values(),
    ]:
        assert np.all(np.isfinite(array))


# The remaining tests intentionally exercise private defensive branches for coverage.
# They are not scientific validation or public API contracts.
def test_private_normalization_and_refinement_error_branches():
    with pytest.raises(RuntimeError, match="zero-length"):
        gg._normalize(np.zeros(3))

    vertices, cells = gg._icosahedron()
    refined_vertices, refined_cells = gg._refine_triangles_bisection(vertices, cells)

    assert refined_vertices.shape == (42, 3)
    assert refined_cells.shape == (80, 3)
    assert refined_vertices is not vertices
    assert refined_cells is not cells


def test_root_refinement_with_face_interior_vertices():
    grid = generate_grid("R03B00")

    assert grid.spec.root == 3
    assert grid.spec.frequency == 3
    assert grid.dims == {"cell": 180, "vertex": 92, "edge": 270}
    assert np.allclose(np.linalg.norm(grid.vertices, axis=1), 1.0)
    assert_outward_cells(grid)


def test_defensive_edge_count_mismatch_check(monkeypatch):
    def fake_build_edges(cells):
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((cells.shape[0], 3), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        )

    monkeypatch.setattr(gg, "_build_edges", fake_build_edges)

    with pytest.raises(RuntimeError, match="generated 0 edges, expected 30"):
        generate_grid("R01B00")


def test_private_orient_cell_swaps_inward_cells():
    vertices = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert gg._orient_cell((0, 1, 2), vertices) == (0, 1, 2)
    assert gg._orient_cell((0, 2, 1), vertices) == (0, 1, 2)


def test_private_build_edges_rejects_open_mesh():
    cells = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="adjacent cells"):
        gg._build_edges(cells)


def test_private_build_edges_preserves_first_seen_edge_order():
    cells = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )

    edges, cell_edges, edge_cells = gg._build_edges(cells)

    assert np.array_equal(
        edges,
        np.array([[1, 0], [2, 1], [0, 2], [3, 0], [1, 3], [2, 3]], dtype=np.int32),
    )
    assert np.array_equal(
        cell_edges,
        np.array([[0, 1, 2], [3, 4, 0], [4, 5, 1], [2, 5, 3]], dtype=np.int32),
    )
    assert np.array_equal(
        edge_cells,
        np.array([[0, 1], [0, 2], [0, 3], [1, 3], [1, 2], [2, 3]], dtype=np.int32),
    )


def test_private_spherical_triangle_areas_match_scalar_helper():
    points = unit_rows(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )
    a = points[[0, 0]]
    b = points[[1, 1]]
    c = points[[2, 3]]

    vectorized = gg._spherical_triangle_areas(a, b, c)
    scalar = np.array(
        [
            gg._spherical_triangle_area(a[0], b[0], c[0]),
            gg._spherical_triangle_area(a[1], b[1], c[1]),
        ]
    )

    assert np.allclose(vectorized, scalar)


def test_private_check_expected_counts_reports_mismatch():
    spec = parse_grid_spec("R01B00")

    with pytest.raises(RuntimeError, match="generated 19 cells"):
        gg._check_expected_counts(spec, np.zeros((12, 3)), np.zeros((19, 3), dtype=np.int32))
    with pytest.raises(RuntimeError, match="generated 11 vertices"):
        gg._check_expected_counts(spec, np.zeros((11, 3)), np.zeros((20, 3), dtype=np.int32))


def test_low_level_icosahedron_and_fixed_padding_helpers():
    vertices, faces = gg._icosahedron()

    assert vertices.shape == (12, 3)
    assert faces.shape == (20, 3)
    assert_unit_sphere(vertices)
    assert gg._sort_around_vertex(vertices, 0, []) == []
    assert np.array_equal(gg._zero_based_with_skip(np.array([[0, 1, 4]], dtype=np.int32)), [[-1, 0, 3]])
    assert np.array_equal(gg._zeros_fixed("cell_grf"), np.zeros((1, 14), dtype=np.int32))
