"""Diagnostics and postprocessing helpers for generated grids."""

from ._diagnostics import (
    GridCheckResult,
    GridStatistics,
    TriangleProperties,
    cell_divergence,
    cell_vorticity_fnorm,
    check_grid,
    grid_statistics,
    triangle_properties,
)

__all__ = [
    "GridCheckResult",
    "GridStatistics",
    "TriangleProperties",
    "cell_divergence",
    "cell_vorticity_fnorm",
    "check_grid",
    "grid_statistics",
    "triangle_properties",
]
