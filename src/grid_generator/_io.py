"""ICON NetCDF writing boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class IconNetcdfWriter:
    """Write complete ICON grid objects to NetCDF."""

    def write(
        self,
        grid: Any,
        path: str | Path,
        *,
        sphere_radius: float | None = None,
    ) -> Path:
        from . import grid_generator as gg

        return gg._write_icon_grid(grid, path, sphere_radius=sphere_radius)
