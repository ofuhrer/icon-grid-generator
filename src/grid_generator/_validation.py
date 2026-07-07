"""Validation helpers for public grid generation options."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._accelerated import SUPPORTED_ACCELERATORS, should_use_numba

MAX_INDEXED_GRID_ELEMENTS = np.iinfo(np.int32).max


def finite_float_option(name: str, value: Any) -> float:
    """Return `value` as float after rejecting booleans, non-numbers, and NaNs."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_grid_options(spec: Any, options: Any) -> None:
    """Validate options that are common to spherical ICON grid generation."""

    radius = finite_float_option("radius", options.radius)
    sphere_radius = finite_float_option("sphere_radius", options.sphere_radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    if sphere_radius <= 0:
        raise ValueError("sphere_radius must be positive")

    if not isinstance(options.accelerator, str):
        raise TypeError("accelerator must be a string")
    if options.accelerator not in SUPPORTED_ACCELERATORS:
        names = ", ".join(sorted(SUPPORTED_ACCELERATORS))
        raise ValueError(f"accelerator must be one of: {names}")
    should_use_numba(options.accelerator)

    global_optimization = getattr(options, "global_optimization", None)
    if global_optimization is None:
        raise TypeError("options must include resolved global optimization settings")
    if (
        global_optimization.method != "none"
        and not hasattr(spec, "bisections")
        and not hasattr(spec, "parent_grid_name")
    ):
        raise ValueError("optimize_global is only supported for global grids")

    if options.max_cells is not None:
        if not isinstance(options.max_cells, int) or isinstance(options.max_cells, bool):
            raise TypeError("max_cells must be None or a positive integer")
        if options.max_cells <= 0:
            raise ValueError("max_cells must be positive")
    if options.max_cells is not None and spec.expected_cells > options.max_cells:
        raise ValueError(
            f"{spec.name} has {spec.expected_cells} cells, exceeding max_cells="
            f"{options.max_cells}"
        )
    for name, size in (
        ("cells", spec.expected_cells),
        ("edges", spec.expected_edges),
        ("vertices", spec.expected_vertices),
    ):
        if size > MAX_INDEXED_GRID_ELEMENTS:
            raise ValueError(
                f"{spec.name} has {size} {name}, exceeding the int32 index limit "
                f"of {MAX_INDEXED_GRID_ELEMENTS}"
            )
