"""Validation helpers for public grid generation options."""

from __future__ import annotations

from typing import Any

import numpy as np


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

    rotation_axis = np.asarray(options.rotation_axis, dtype=np.float64)
    if rotation_axis.shape != (3,) or not np.all(np.isfinite(rotation_axis)):
        raise ValueError("rotation_axis must contain three finite numbers")
    rotation_angle_degrees = finite_float_option(
        "rotation_angle_degrees",
        options.rotation_angle_degrees,
    )
    if np.linalg.norm(rotation_axis) == 0.0 and rotation_angle_degrees != 0.0:
        raise ValueError("rotation_axis must be non-zero when rotation_angle_degrees is non-zero")

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
