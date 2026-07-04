"""Grid diagnostics and lightweight postprocessing operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridCheckResult:
    """Structural grid validation result."""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GridStatistics:
    """Common scalar grid statistics."""

    cells: int
    edges: int
    vertices: int
    boundary_edges: int
    min_cell_area: float
    max_cell_area: float
    mean_cell_area: float
    min_edge_length: float
    max_edge_length: float
    mean_edge_length: float


@dataclass(frozen=True)
class TriangleProperties:
    """Per-cell triangle geometry."""

    area: np.ndarray
    edge_lengths: np.ndarray
    min_angle_degrees: np.ndarray
    max_angle_degrees: np.ndarray


def check_grid(grid: Any) -> GridCheckResult:
    """Return structural validation errors and warnings for an ICON grid."""
    errors: list[str] = []
    warnings: list[str] = []
    _check_shape(errors, "cells", grid.cells, (grid.dims["cell"], 3))
    _check_shape(errors, "edges", grid.edges, (grid.dims["edge"], 2))
    _check_shape(errors, "cell_edges", grid.cell_edges, (grid.dims["cell"], 3))
    _check_shape(errors, "edge_cells", grid.edge_cells, (grid.dims["edge"], 2))
    for name in ["vertices", "cell_center_xyz", "edge_center_xyz"]:
        if not np.all(np.isfinite(getattr(grid, name))):
            errors.append(f"{name} contains non-finite values")
    if np.any((grid.cells < 0) | (grid.cells >= grid.dims["vertex"])):
        errors.append("cells contain out-of-range vertex indices")
    if np.any((grid.edges < 0) | (grid.edges >= grid.dims["vertex"])):
        errors.append("edges contain out-of-range vertex indices")
    if np.any((grid.cell_edges < 0) | (grid.cell_edges >= grid.dims["edge"])):
        errors.append("cell_edges contain out-of-range edge indices")
    active_edge_cells = grid.edge_cells[grid.edge_cells >= 0]
    if np.any(active_edge_cells >= grid.dims["cell"]):
        errors.append("edge_cells contain out-of-range cell indices")
    if len({tuple(sorted(map(int, edge))) for edge in grid.edges}) != grid.dims["edge"]:
        errors.append("edges contain duplicate vertex pairs")
    if np.any(grid.edge_cells[:, 1] < 0):
        warnings.append("grid has open boundary edges")
    elif grid.dims["vertex"] - grid.dims["edge"] + grid.dims["cell"] != 2:
        warnings.append("closed grid Euler characteristic is not spherical")
    return GridCheckResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def grid_statistics(grid: Any) -> GridStatistics:
    """Return count, boundary, area, and edge-length statistics."""
    area = np.asarray(grid.geometry["cell_area"], dtype=np.float64)
    edge_length = np.asarray(grid.geometry["edge_length"], dtype=np.float64)
    return GridStatistics(
        cells=grid.dims["cell"],
        edges=grid.dims["edge"],
        vertices=grid.dims["vertex"],
        boundary_edges=int(np.count_nonzero(grid.edge_cells[:, 1] < 0)),
        min_cell_area=float(area.min()),
        max_cell_area=float(area.max()),
        mean_cell_area=float(area.mean()),
        min_edge_length=float(edge_length.min()),
        max_edge_length=float(edge_length.max()),
        mean_edge_length=float(edge_length.mean()),
    )


def triangle_properties(grid: Any) -> TriangleProperties:
    """Return per-cell triangle area, side lengths, and Euclidean angles."""
    side_lengths = np.asarray(grid.geometry["edge_length"], dtype=np.float64)[grid.cell_edges]
    a = side_lengths[:, 0]
    b = side_lengths[:, 1]
    c = side_lengths[:, 2]
    angles = np.column_stack(
        (
            _angle_from_sides(c, a, b),
            _angle_from_sides(a, b, c),
            _angle_from_sides(b, c, a),
        )
    )
    return TriangleProperties(
        area=np.asarray(grid.geometry["cell_area"], dtype=np.float64).copy(),
        edge_lengths=side_lengths,
        min_angle_degrees=np.degrees(angles.min(axis=1)),
        max_angle_degrees=np.degrees(angles.max(axis=1)),
    )


def cell_divergence(grid: Any, edge_flux: np.ndarray) -> np.ndarray:
    """Compute cell divergence from edge-normal fluxes."""
    flux = np.asarray(edge_flux, dtype=np.float64)
    if flux.shape != (grid.dims["edge"],):
        raise ValueError("edge_flux must have shape (edge,)")
    c2e = grid.icon_connectivity["c2e"]
    orientation = grid.icon_connectivity["orientation_of_normal"]
    edge_length = np.asarray(grid.geometry["edge_length"], dtype=np.float64)
    area = np.asarray(grid.geometry["cell_area"], dtype=np.float64)
    return np.sum(orientation * flux[c2e] * edge_length[c2e], axis=1) / area


def cell_vorticity_fnorm(
    grid: Any,
    vertex_vorticity: np.ndarray,
    coriolis: np.ndarray | float,
) -> np.ndarray:
    """Average vertex vorticity to cells and normalize by the Coriolis value."""
    values = np.asarray(vertex_vorticity, dtype=np.float64)
    if values.shape != (grid.dims["vertex"],):
        raise ValueError("vertex_vorticity must have shape (vertex,)")
    averaged = values[grid.cells].mean(axis=1)
    coriolis_array = np.asarray(coriolis, dtype=np.float64)
    if coriolis_array.shape == ():
        if float(coriolis_array) == 0.0:
            raise ValueError("coriolis must be non-zero")
        return averaged / float(coriolis_array)
    if coriolis_array.shape != (grid.dims["cell"],):
        raise ValueError("coriolis must be scalar or have shape (cell,)")
    if np.any(coriolis_array == 0.0):
        raise ValueError("coriolis must be non-zero")
    return averaged / coriolis_array


def _check_shape(
    errors: list[str],
    name: str,
    array: np.ndarray,
    expected: tuple[int, ...],
) -> None:
    if array.shape != expected:
        errors.append(f"{name} has shape {array.shape}, expected {expected}")


def _angle_from_sides(left: np.ndarray, right: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    denominator = 2.0 * left * right
    cosine = np.divide(
        left**2 + right**2 - opposite**2,
        denominator,
        out=np.ones_like(opposite),
        where=denominator != 0.0,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))
