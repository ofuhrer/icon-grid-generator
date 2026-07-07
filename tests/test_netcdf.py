from __future__ import annotations

import builtins

import numpy as np
import pytest

from grid_generator import TorusGridSpec, generate_grid
from grid_generator import _netcdf


def unit_rows(points):
    return points / np.linalg.norm(points, axis=1)[:, np.newaxis]


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
        for variable_name, expected_attrs in _netcdf.ICON_VARIABLE_ATTRS.items():
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
