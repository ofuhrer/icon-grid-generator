from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

import grid_generator
from grid_generator import (
    ChannelGridSpec,
    GlobalGridSpec,
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    RaggedOrthogonalGridSpec,
    StretchedTorusGridSpec,
    TorusGridSpec,
    generate_grid,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed:RuntimeWarning"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
PYTHON_VERSION_MATRIX_RE = re.compile(r"python-version:\s*\[([^\]]+)\]")
PYTHON_BADGE_RE = re.compile(
    r"https://img\.shields\.io/badge/python-(3\.\d+)--(3\.\d+)-blue\.svg"
)
PYTHON_CLASSIFIER_RE = re.compile(
    r'"Programming Language :: Python :: (3\.\d+)"'
)

EXPECTED_NETCDF_DIMS = {
    "cell": 20,
    "vertex": 12,
    "edge": 30,
    "nc": 2,
    "nv": 3,
    "ne": 6,
    "no": 4,
    "max_chdom": 1,
    "cell_grf": 14,
    "edge_grf": 24,
    "vert_grf": 13,
}

EXPECTED_NETCDF_VARIABLE_DIMS = {
    "adjacent_cell_of_edge": ("nc", "edge"),
    "cartesian_x_vertices": ("vertex",),
    "cartesian_y_vertices": ("vertex",),
    "cartesian_z_vertices": ("vertex",),
    "cell_area": ("cell",),
    "cell_area_p": ("cell",),
    "cell_circumcenter_cartesian_x": ("cell",),
    "cell_circumcenter_cartesian_y": ("cell",),
    "cell_circumcenter_cartesian_z": ("cell",),
    "cell_elevation": ("cell",),
    "cell_index": ("cell",),
    "cell_sea_land_mask": ("cell",),
    "cells_of_vertex": ("ne", "vertex"),
    "child_cell_id": ("cell",),
    "child_cell_index": ("no", "cell"),
    "child_edge_id": ("edge",),
    "child_edge_index": ("no", "edge"),
    "clat": ("cell",),
    "clat_vertices": ("cell", "nv"),
    "clon": ("cell",),
    "clon_vertices": ("cell", "nv"),
    "dual_area": ("vertex",),
    "dual_area_p": ("vertex",),
    "dual_edge_length": ("edge",),
    "edge_cell_distance": ("nc", "edge"),
    "edge_dual_middle_cartesian_x": ("edge",),
    "edge_dual_middle_cartesian_y": ("edge",),
    "edge_dual_middle_cartesian_z": ("edge",),
    "edge_dual_normal_cartesian_x": ("edge",),
    "edge_dual_normal_cartesian_y": ("edge",),
    "edge_dual_normal_cartesian_z": ("edge",),
    "edge_elevation": ("edge",),
    "edge_index": ("edge",),
    "edge_length": ("edge",),
    "edge_middle_cartesian_x": ("edge",),
    "edge_middle_cartesian_y": ("edge",),
    "edge_middle_cartesian_z": ("edge",),
    "edge_of_cell": ("nv", "cell"),
    "edge_orientation": ("ne", "vertex"),
    "edge_parent_type": ("edge",),
    "edge_primal_normal_cartesian_x": ("edge",),
    "edge_primal_normal_cartesian_y": ("edge",),
    "edge_primal_normal_cartesian_z": ("edge",),
    "edge_sea_land_mask": ("edge",),
    "edge_system_orientation": ("edge",),
    "edge_vert_distance": ("nc", "edge"),
    "edge_vertices": ("nc", "edge"),
    "edgequad_area": ("edge",),
    "edges_of_vertex": ("ne", "vertex"),
    "elat": ("edge",),
    "elat_vertices": ("edge", "no"),
    "elon": ("edge",),
    "elon_vertices": ("edge", "no"),
    "end_idx_c": ("max_chdom", "cell_grf"),
    "end_idx_e": ("max_chdom", "edge_grf"),
    "end_idx_v": ("max_chdom", "vert_grf"),
    "lat_cell_centre": ("cell",),
    "lat_edge_centre": ("edge",),
    "latitude_vertices": ("vertex",),
    "lon_cell_centre": ("cell",),
    "lon_edge_centre": ("edge",),
    "longitude_vertices": ("vertex",),
    "meridional_normal_dual_edge": ("edge",),
    "meridional_normal_primal_edge": ("edge",),
    "neighbor_cell_index": ("nv", "cell"),
    "orientation_of_normal": ("nv", "cell"),
    "parent_cell_index": ("cell",),
    "parent_cell_type": ("cell",),
    "parent_edge_index": ("edge",),
    "parent_vertex_index": ("vertex",),
    "phys_cell_id": ("cell",),
    "phys_edge_id": ("edge",),
    "refin_c_ctrl": ("cell",),
    "refin_e_ctrl": ("edge",),
    "refin_v_ctrl": ("vertex",),
    "start_idx_c": ("max_chdom", "cell_grf"),
    "start_idx_e": ("max_chdom", "edge_grf"),
    "start_idx_v": ("max_chdom", "vert_grf"),
    "vertex_index": ("vertex",),
    "vertex_of_cell": ("nv", "cell"),
    "vertices_of_vertex": ("ne", "vertex"),
    "vlat": ("vertex",),
    "vlon": ("vertex",),
    "zonal_normal_dual_edge": ("edge",),
    "zonal_normal_primal_edge": ("edge",),
}


def test_markdown_python_examples_execute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    markdown_files = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]

    for markdown_file in markdown_files:
        namespace: dict[str, object] = {"__name__": "__main__"}
        for match in PYTHON_BLOCK_RE.finditer(markdown_file.read_text()):
            code = match.group(1)
            exec(compile(code, str(markdown_file), "exec"), namespace)


def test_public_api_inventory_matches_documented_exports():
    api_text = (PROJECT_ROOT / "docs" / "api.md").read_text()
    documented = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", api_text))

    assert set(grid_generator.__all__) <= documented


def test_readme_python_badge_matches_test_workflow_matrix():
    workflow_text = (PROJECT_ROOT / ".github" / "workflows" / "test.yml").read_text()
    matrix_versions = sorted(
        {
            version
            for matrix in PYTHON_VERSION_MATRIX_RE.findall(workflow_text)
            for version in re.findall(r'"([^"]+)"', matrix)
        },
        key=lambda version: tuple(int(part) for part in version.split(".")),
    )
    assert matrix_versions

    minors = [int(version.split(".")[1]) for version in matrix_versions]
    assert minors == list(range(min(minors), max(minors) + 1))

    readme_text = (PROJECT_ROOT / "README.md").read_text()
    badge_match = PYTHON_BADGE_RE.search(readme_text)
    assert badge_match is not None
    assert badge_match.groups() == (matrix_versions[0], matrix_versions[-1])

    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text()
    classifier_versions = sorted(
        set(PYTHON_CLASSIFIER_RE.findall(pyproject_text)),
        key=lambda version: tuple(int(part) for part in version.split(".")),
    )
    assert classifier_versions == matrix_versions


@pytest.mark.parametrize(
    "spec",
    [
        GlobalGridSpec(root=1, bisections=0),
        TorusGridSpec(nx=4, ny=3, edge_length=1.0),
        StretchedTorusGridSpec(nx=4, ny=3, edge_length=1.0, stretch_y=1.1),
        ChannelGridSpec(nx=3, ny=2, edge_length=1.0),
        ParallelogramGridSpec(nx=3, ny=2, edge_length=1.0),
        RaggedOrthogonalGridSpec(nx=3, ny=2, dx=1.0, dy=1.0),
        LimitedAreaGridSpec(
            "R02B01",
            lon_min=-30.0,
            lon_max=30.0,
            lat_min=-30.0,
            lat_max=30.0,
            boundary_depth=1,
        ),
    ],
)
def test_all_public_grid_specs_export_to_netcdf(spec, tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid(spec, options={"max_cells": None})
    path = grid.to_netcdf(tmp_path / f"{grid.name}.nc")

    with netcdf4.Dataset(path) as dataset:
        assert dataset.dimensions["cell"].size == grid.dims["cell"]
        assert dataset.dimensions["edge"].size == grid.dims["edge"]
        assert dataset.dimensions["vertex"].size == grid.dims["vertex"]
        for variable_name in EXPECTED_NETCDF_VARIABLE_DIMS:
            assert variable_name in dataset.variables
        for variable_name in ("clon", "clat", "cell_area", "edge_length", "dual_area"):
            assert np.all(np.isfinite(dataset.variables[variable_name][:]))


def test_netcdf_schema_snapshot_for_core_grid(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    grid = generate_grid("R01B00")
    path = grid.to_netcdf(tmp_path / "r01b00.nc")

    with netcdf4.Dataset(path) as dataset:
        dims = {name: len(dimension) for name, dimension in dataset.dimensions.items()}
        variable_dims = {
            name: variable.dimensions for name, variable in dataset.variables.items()
        }

        assert dims == EXPECTED_NETCDF_DIMS
        assert variable_dims == EXPECTED_NETCDF_VARIABLE_DIMS
        assert dataset.getncattr("title") == "Pure Python ICON grid R01B00"
        assert dataset.getncattr("grid_geometry") == 1
        assert dataset.getncattr("grid_cell_type") == 3
        assert dataset.getncattr("uuidOfHGrid") == grid.metadata["uuidOfHGrid"]
        for name in ("uuidOfHGrid", "grid_root", "grid_level", "sphere_radius"):
            assert name in dataset.ncattrs()


def test_comparison_artifacts_remain_ignored():
    tracked_artifact_prefixes = ("dist/", "build/", "site/", "tmp/")
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("tracked-file check requires a Git checkout")
    tracked_files = result.stdout.splitlines()

    assert not any(
        path.startswith(tracked_artifact_prefixes) for path in tracked_files
    )
