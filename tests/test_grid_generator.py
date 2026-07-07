from __future__ import annotations

import math
import sys
import types
import numpy as np
import pytest

from grid_generator import _accelerated
from grid_generator import (
    ChannelGridSpec,
    IconGridOptions,
    GlobalGridSpec,
    LimitedAreaGridSpec,
    Region,
    TorusGridSpec,
    generate_grid,
)
from grid_generator import grid_generator as gg
from grid_generator._geometry import SphericalIcosahedralGeometry
from grid_generator._metrics import SphericalMetricsBuilder
from grid_generator._ordering import IconOrderingBuilder, _permute_cells
from grid_generator._types import BisectionProvenance, GeometryData
from grid_generator._refinement import GlobalRefinementBuilder
from grid_generator._topology import GlobalTopologyBuilder
from grid_generator.grid_generator import parse_grid_spec
from grid_generator.transforms import (
    DiffusionOptions,
    OptimizationOptions,
    optimize_global_grid,
    optimize_grid,
)




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


def edge_quadrilateral_dual_areas(grid):
    dual_areas = np.zeros(grid.dims["vertex"], dtype=np.float64)
    for vertex_index, edge_row in enumerate(grid.icon_connectivity["v2e"]):
        edge_indices = edge_row[edge_row > 0] - 1
        dual_areas[vertex_index] = np.sum(
            0.25
            * grid.geometry["edge_length"][edge_indices]
            * grid.geometry["dual_edge_length"][edge_indices]
        )
    return dual_areas


















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


def test_raw_bisection_generation_matches_direct_pipeline():
    spec = GlobalGridSpec(root=2, bisections=3)
    options = IconGridOptions(max_cells=None, optimize_global=False)
    public = generate_grid(spec, options=options)
    direct = gg._generate_raw_global_grid(spec, options, gg._GlobalGenerationContext())

    assert public.metadata["global_optimization"] == "none"
    assert np.array_equal(public.vertices, direct.vertices)
    assert np.array_equal(public.cells, direct.cells)
    assert np.array_equal(public.edges, direct.edges)
    assert np.array_equal(public.cell_edges, direct.cell_edges)
    assert np.array_equal(public.edge_cells, direct.edge_cells)
    assert np.array_equal(
        public.geometry["orientation_of_normal"],
        direct.geometry["orientation_of_normal"],
    )


def test_staged_global_generation_evicts_completed_parent_caches():
    context = gg._GlobalGenerationContext()
    spec = GlobalGridSpec(root=2, bisections=2)
    grid = gg._generate_grid(spec, IconGridOptions(max_cells=None, spring_iterations=1), context)

    assert grid.name == "R02B02"
    assert set(context.grids) == {(2, 2)}
    assert not context.parent_data
    assert not context.parent_vertex_indices


def test_private_matching_unit_point_indices_matches_nearest_mapping():
    options = IconGridOptions(max_cells=None, spring_iterations=1)
    context = gg._GlobalGenerationContext()
    parent = gg._generate_grid(GlobalGridSpec(root=2, bisections=2), options, context)
    vertices, cells, provenance = gg._refine_triangles_bisection_with_provenance(
        parent.vertices,
        parent.cells,
    )
    geometry = gg._geometry_from_vertices(
        GlobalGridSpec(root=2, bisections=3),
        options,
        vertices * options.radius,
        cells,
        provenance,
    )
    parent_vertex_count = int(
        np.max(provenance.parent_vertex_index[provenance.parent_vertex_index > 0])
    )
    provenance_edge_centers = gg._edge_centers(
        geometry.vertices[:parent_vertex_count],
        provenance.edges,
        options.radius,
    )

    keyed = gg._matching_unit_point_indices(provenance_edge_centers, parent.edge_center_xyz)
    nearest = gg._nearest_unit_point_indices(provenance_edge_centers, parent.edge_center_xyz)

    assert np.array_equal(keyed, nearest)
    with pytest.raises(RuntimeError, match="matching target"):
        gg._matching_unit_point_indices(
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 1.0, 0.0]]),
        )


def test_private_matching_edge_indices_by_vertices_matches_spatial_mapping():
    options = IconGridOptions(max_cells=None, optimize_global=False)
    context = gg._GlobalGenerationContext()
    parent = gg._generate_grid(GlobalGridSpec(root=2, bisections=2), options, context)
    vertices, _, provenance = gg._refine_triangles_bisection_with_provenance(
        parent.vertices,
        parent.cells,
    )
    parent_vertex_count = int(
        np.max(provenance.parent_vertex_index[provenance.parent_vertex_index > 0])
    )
    provenance_edge_centers = gg._edge_centers(
        vertices[:parent_vertex_count],
        provenance.edges,
        options.radius,
    )

    keyed = gg._matching_edge_indices_by_vertices(provenance.edges, parent.edges)
    spatial = gg._matching_unit_point_indices(provenance_edge_centers, parent.edge_center_xyz)

    assert np.array_equal(keyed, spatial)
    with pytest.raises(RuntimeError, match="matching target edge"):
        gg._matching_edge_indices_by_vertices(
            np.asarray([[0, parent.dims["vertex"]]], dtype=np.int32),
            parent.edges,
        )


def test_bisection_provenance_stores_parent_edge_fields():
    parent = generate_grid("R02B02", options={"max_cells": None, "optimize_global": False})
    _, _, provenance = gg._refine_triangles_bisection_with_provenance(
        parent.vertices,
        parent.cells,
    )
    computed_parent_edge, computed_edge_type = gg._parent_edge_fields(
        provenance.child_edges,
        provenance.parent_vertex_index,
        provenance,
        "numpy",
    )

    assert np.array_equal(provenance.child_parent_edge_index, computed_parent_edge)
    assert np.array_equal(provenance.child_edge_parent_type, computed_edge_type)


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

    order_threshold = _accelerated.AUTO_NUMBA_MIN_ORDER_CELLS
    assert not _accelerated.should_use_numba_ordering("auto", order_threshold - 1)
    assert (
        _accelerated.should_use_numba_ordering("auto", order_threshold)
        == _accelerated.is_numba_available()
    )


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

    def assert_same_array(actual, expected):
        if np.issubdtype(np.asarray(expected).dtype, np.floating):
            assert np.allclose(actual, expected)
        else:
            assert np.array_equal(actual, expected)

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
        assert_same_array(getattr(grid, name), getattr(geometry, name))
    for name in [
        "edges",
        "cell_edges",
        "edge_cells",
        "edge_center_xyz",
        "edge_lon",
        "edge_lat",
    ]:
        assert_same_array(getattr(grid, name), getattr(topology, name))
    for name, value in topology.icon_connectivity.items():
        assert np.array_equal(value, grid.icon_connectivity[name])
    for name, value in topology.connectivity.items():
        assert np.array_equal(value, grid.connectivity[name])
    for name, value in topology.neighbor_tables.items():
        assert np.array_equal(value, grid.neighbor_tables[name])
    for name, value in metrics.fields.items():
        assert_same_array(grid.geometry[name], value)
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
    assert np.allclose(dual_area, edge_quadrilateral_dual_areas(grid))
    assert abs(dual_area.sum() - cell_area.sum()) / cell_area.sum() < 0.04
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
    expected_dual_area = edge_quadrilateral_dual_areas(grid)[0]

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
    assert np.allclose(geometry["dual_area"], edge_quadrilateral_dual_areas(grid))
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


def test_dual_areas_are_incident_edge_quadrilateral_areas():
    sphere_radius = 3.0
    grid = generate_grid("R02B02", options={"sphere_radius": sphere_radius})
    expected_dual_area = edge_quadrilateral_dual_areas(grid)

    assert np.allclose(grid.geometry["dual_area"], expected_dual_area)
    assert abs(grid.geometry["dual_area"].sum() - grid.geometry["cell_area"].sum()) / grid.geometry[
        "cell_area"
    ].sum() < 0.001


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
    expected_dual_area = edge_quadrilateral_dual_areas(grid)

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
    assert np.allclose(dual_area, edge_quadrilateral_dual_areas(grid))
    assert abs(dual_area.sum() - cell_area.sum()) / cell_area.sum() < 0.04
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
def _coverage_geometry(
    cells: np.ndarray | None = None,
    provenance: BisectionProvenance | None = None,
) -> GeometryData:
    if cells is None:
        cells = np.asarray(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ],
            dtype=np.int32,
        )
    cell_count = cells.shape[0]
    vertices = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return GeometryData(
        vertices=vertices,
        cells=cells,
        lon=np.arange(cell_count, dtype=np.float64),
        lat=np.arange(cell_count, dtype=np.float64) + 10.0,
        vertex_lon=np.arange(vertices.shape[0], dtype=np.float64),
        vertex_lat=np.arange(vertices.shape[0], dtype=np.float64) + 20.0,
        cell_center_xyz=np.arange(cell_count * 3, dtype=np.float64).reshape(cell_count, 3),
        cell_vertex_lon=np.arange(cell_count * 3, dtype=np.float64).reshape(cell_count, 3) + 30.0,
        cell_vertex_lat=np.arange(cell_count * 3, dtype=np.float64).reshape(cell_count, 3) + 40.0,
        source_cell_index=np.arange(cell_count, dtype=np.int32) + 100,
        source_vertex_index=np.arange(vertices.shape[0], dtype=np.int32) + 200,
        bisection_provenance=provenance,
    )


def test_private_accelerated_lookup_algorithms_with_identity_njit(monkeypatch):
    fake_numba = types.SimpleNamespace(njit=lambda function: function)
    monkeypatch.setitem(sys.modules, "numba", fake_numba)
    monkeypatch.setattr(_accelerated, "is_numba_available", lambda: True)
    grid = generate_grid("R01B00", options={"accelerator": "numpy"})
    _accelerated._compiled_lookup_width2.cache_clear()
    _accelerated._compiled_lookup_width3.cache_clear()
    _accelerated._compiled_order_cells_by_edges.cache_clear()
    _accelerated._compiled_fill_bisection_children.cache_clear()
    try:
        signature2 = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        query2 = np.asarray([[1, 2], [2, 3], [3, 4]], dtype=np.int64)
        parent2, type2 = _accelerated.lookup_width2_numba(
            signature2,
            np.asarray([11, 22], dtype=np.int32),
            np.asarray([201, 202], dtype=np.int32),
            query2,
        )

        signature3 = np.asarray([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
        query3 = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.int64)
        parent3, type3 = _accelerated.lookup_width3_numba(
            signature3,
            np.asarray([33, 44], dtype=np.int32),
            np.asarray([203, 204], dtype=np.int32),
            query3,
        )
        expected_cells, expected_cell_edges = gg._order_cells_by_edges(
            grid.vertices,
            grid.cells,
            grid.edges,
            grid.cell_edges,
            grid.edge_cells,
            "numpy",
        )
        edge_system_orientation = gg._edge_system_orientation(
            grid.vertices,
            grid.cell_center_xyz,
            grid.edges,
            grid.edge_cells,
            grid.edge_center_xyz,
        )
        ordered_cells, ordered_cell_edges, failure_cell, failure_kind = (
            _accelerated.order_cells_by_edges_numba(
                grid.edges,
                grid.cell_edges,
                grid.edge_cells,
                edge_system_orientation.astype(np.int32, copy=False),
            )
        )
        vertices, cells = gg._icosahedron()
        numpy_vertices, numpy_cells, numpy_provenance = (
            gg._refine_triangles_bisection_with_provenance(vertices, cells, "numpy")
        )
        numba_vertices, numba_cells, numba_provenance = (
            gg._refine_triangles_bisection_with_provenance(vertices, cells, "numba")
        )
    finally:
        _accelerated._compiled_lookup_width2.cache_clear()
        _accelerated._compiled_lookup_width3.cache_clear()
        _accelerated._compiled_order_cells_by_edges.cache_clear()
        _accelerated._compiled_fill_bisection_children.cache_clear()

    assert np.array_equal(parent2, np.asarray([11, 0, 22], dtype=np.int32))
    assert np.array_equal(type2, np.asarray([201, 0, 202], dtype=np.int32))
    assert np.array_equal(parent3, np.asarray([33, 0, 44], dtype=np.int32))
    assert np.array_equal(type3, np.asarray([203, 0, 204], dtype=np.int32))
    assert failure_cell == -1
    assert failure_kind == 0
    assert np.array_equal(ordered_cells, expected_cells)
    assert np.array_equal(ordered_cell_edges, expected_cell_edges)
    assert np.allclose(numba_vertices, numpy_vertices)
    assert np.array_equal(numba_cells, numpy_cells)
    assert np.array_equal(numba_provenance.child_edges, numpy_provenance.child_edges)
    assert np.array_equal(
        numba_provenance.child_cell_edges,
        numpy_provenance.child_cell_edges,
    )
    assert np.array_equal(
        numba_provenance.child_parent_edge_index,
        numpy_provenance.child_parent_edge_index,
    )
    assert np.array_equal(
        numba_provenance.child_edge_parent_type,
        numpy_provenance.child_edge_parent_type,
    )


def test_private_numba_accelerator_selection_branches(monkeypatch):
    monkeypatch.setattr(_accelerated, "is_numba_available", lambda: False)
    with pytest.raises(ModuleNotFoundError, match="accelerate"):
        _accelerated.should_use_numba("numba")

    monkeypatch.setattr(_accelerated, "is_numba_available", lambda: True)
    assert _accelerated.should_use_numba("numba")
    assert _accelerated.should_use_numba("numpy") is False
    assert _accelerated.should_use_numba("auto", _accelerated.AUTO_NUMBA_MIN_LOOKUP_ROWS)
    assert _accelerated.should_use_numba_ordering("numba", 1)
    assert _accelerated.should_use_numba_ordering("numpy", 10**9) is False
    assert _accelerated.should_use_numba_ordering(
        "auto",
        _accelerated.AUTO_NUMBA_MIN_ORDER_CELLS,
    )


def test_private_ordering_builder_sorts_children_by_parent_and_child_type(monkeypatch):
    geometry = _coverage_geometry()
    context = object()
    parent = object()

    def fake_parent_vertex_indices_cached(spec, options, vertices, cache_context):
        assert spec == GlobalGridSpec(root=1, bisections=1)
        assert cache_context is context
        assert vertices is geometry.vertices
        return np.arange(geometry.vertices.shape[0], dtype=np.int32), parent

    def fake_parent_cell_fields(cells, parent_vertex_index, parent_grid, accelerator):
        assert cells is geometry.cells
        assert parent_grid is parent
        assert accelerator == "numpy"
        return (
            np.asarray([2, 1, 1, 2], dtype=np.int32),
            np.asarray([202, 200, 203, 201], dtype=np.int32),
        )

    monkeypatch.setattr(gg, "_parent_vertex_indices_cached", fake_parent_vertex_indices_cached)
    monkeypatch.setattr(gg, "_parent_cell_fields", fake_parent_cell_fields)

    ordered = IconOrderingBuilder(context=context).order_spherical_bisection(
        GlobalGridSpec(root=1, bisections=1),
        IconGridOptions(accelerator="numpy", optimize_global=False),
        geometry,
    )

    permutation = np.asarray([1, 2, 3, 0], dtype=np.int64)
    assert np.array_equal(ordered.cells, geometry.cells[permutation])
    assert np.array_equal(ordered.lon, geometry.lon[permutation])
    assert np.array_equal(ordered.source_cell_index, geometry.source_cell_index[permutation])
    assert ordered.vertices is geometry.vertices
    assert ordered.source_vertex_index is geometry.source_vertex_index


def test_private_ordering_builder_and_permutation_early_branches():
    geometry = _coverage_geometry()
    builder = IconOrderingBuilder()

    assert (
        builder.order_spherical_bisection(
            GlobalGridSpec(root=1, bisections=0),
            IconGridOptions(optimize_global=False),
            geometry,
        )
        is geometry
    )

    provenance = BisectionProvenance(
        cells=geometry.cells,
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        cell_edges=np.asarray([[0, 1, 0]], dtype=np.int32),
        parent_vertex_index=np.arange(geometry.vertices.shape[0], dtype=np.int32),
        parent_cell_index=np.asarray([10, 20, 30, 40], dtype=np.int32),
        parent_cell_type=np.asarray([200, 201, 202, 203], dtype=np.int32),
    )
    geometry_with_provenance = _coverage_geometry(provenance=provenance)

    assert (
        builder.order_spherical_bisection(
            GlobalGridSpec(root=1, bisections=1),
            IconGridOptions(optimize_global=False),
            geometry_with_provenance,
        )
        is geometry_with_provenance
    )

    permutation = np.asarray([3, 1, 2, 0], dtype=np.int64)
    ordered = _permute_cells(geometry_with_provenance, permutation)
    assert np.array_equal(ordered.cells, geometry_with_provenance.cells[permutation])
    assert ordered.bisection_provenance is not None
    assert np.array_equal(
        ordered.bisection_provenance.parent_cell_index,
        provenance.parent_cell_index[permutation],
    )
    assert np.array_equal(
        ordered.bisection_provenance.parent_cell_type,
        provenance.parent_cell_type[permutation],
    )


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: IconGridOptions(fixed_boundary=1), TypeError, "fixed_boundary"),
        (lambda: IconGridOptions(centre="78"), TypeError, "centre"),
        (lambda: IconGridOptions(subcentre=-1), ValueError, "subcentre"),
        (lambda: IconGridOptions(number_of_grid_used=False), TypeError, "number_of_grid_used"),
        (lambda: OptimizationOptions(relaxation=-0.1), ValueError, "relaxation"),
        (lambda: OptimizationOptions(fixed_boundary=1), TypeError, "fixed_boundary"),
        (lambda: OptimizationOptions(target_edge_length=0.0), ValueError, "target_edge_length"),
        (lambda: DiffusionOptions(diffusion_constant=-0.1), ValueError, "diffusion_constant"),
        (lambda: DiffusionOptions(dt=-0.1), ValueError, "dt"),
        (lambda: DiffusionOptions(neighbor_weight=0.0), ValueError, "neighbor_weight"),
        (lambda: DiffusionOptions(fixed_boundary=1), TypeError, "fixed_boundary"),
    ],
)
def test_private_option_validation_error_branches(factory, error, message):
    with pytest.raises(error, match=message):
        factory()


def test_private_optimization_helpers_cover_boundary_and_projection_branches():
    from grid_generator import _optimization

    grid = generate_grid(ChannelGridSpec(nx=3, ny=2, edge_length=1.0))
    assert np.all(_optimization._movable_vertices(grid, fixed_boundary=False))

    assert OptimizationOptions(target_edge_length=1.5).target_edge_length == 1.5
    assert _optimization.resolve_global_optimization_options(None).method == "none"
    assert _optimization.resolve_global_optimization_options("spring").method == "spring"
    assert (
        _optimization.resolve_global_optimization_options(
            _optimization._GlobalOptimizationOptions(method="spring", iterations=1)
        ).method
        == "spring"
    )
    with pytest.raises(TypeError, match="method"):
        _optimization._GlobalOptimizationOptions(method=1)
    with pytest.raises(ValueError, match="method"):
        _optimization._GlobalOptimizationOptions(method="fast")
    with pytest.raises(TypeError, match="options"):
        _optimization.resolve_global_optimization_options(0)

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(
        _optimization._spring_target(vertices, 0, [0], target_edge_length=1.0),
        vertices[0],
    )
    assert np.allclose(
        _optimization._spring_target(vertices, 1, [0, 2], target_edge_length=2.0),
        np.asarray([1.0, 0.0, 0.0]),
    )

    spherical = generate_grid("R01B00", options={"optimize_global": False})
    broken_vertices = spherical.vertices.copy()
    broken_vertices[0] = 0.0
    with pytest.raises(RuntimeError, match="zero-length"):
        _optimization._project_vertices(spherical, broken_vertices)

    assert optimize_global_grid(spherical, {"method": "none"}) is spherical
    assert optimize_global_grid(spherical, {"method": "spring", "iterations": 0}) is spherical
    assert np.allclose(
        _optimization._spring_relaxed_vertices(
            spherical,
            _optimization._GlobalOptimizationOptions(method="spring", iterations=0),
        ),
        spherical.vertices,
    )
    with pytest.raises(ValueError, match="global"):
        optimize_global_grid(grid, {"method": "spring", "iterations": 1})

    torus = generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=1.0))
    optimized_torus = optimize_grid(
        torus,
        OptimizationOptions(iterations=1, fixed_boundary=False, target_edge_length=1.0),
    )
    assert np.all(optimized_torus.vertices[:, 0] >= 0.0)
    assert np.all(optimized_torus.vertices[:, 0] < torus.spec.domain_length)
    assert np.all(optimized_torus.vertices[:, 1] >= 0.0)
    assert np.all(optimized_torus.vertices[:, 1] < torus.spec.domain_height)

    limited = generate_grid(
        LimitedAreaGridSpec(
            parent="R02B01",
            region=Region.lonlat_box(lon_min=-30.0, lon_max=30.0, lat_min=-30.0, lat_max=30.0),
            boundary_depth=1,
        ),
        options={"max_cells": None},
    )
    optimized_limited = optimize_grid(
        limited,
        OptimizationOptions(iterations=1, fixed_boundary=False),
    )
    assert np.array_equal(optimized_limited.edge_cells, limited.edge_cells)
    assert np.all(np.isfinite(optimized_limited.geometry["edge_cell_distance"]))


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
