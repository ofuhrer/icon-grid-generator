"""Lightweight SVG visualization for generated grids."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np


def write_svg(
    grid: Any,
    path: str | Path,
    *,
    width: int = 900,
    height: int = 450,
    max_edges: int = 20_000,
    stroke: str = "#1f2937",
    background: str = "#ffffff",
) -> Path:
    """Write a simple SVG edge plot for a generated grid.

    The helper intentionally has no plotting dependency. It projects vertex
    longitude/latitude coordinates into a compact SVG viewport and draws grid
    edges as line segments. For global grids, edges that cross the dateline are
    omitted to avoid wraparound artifacts in the equirectangular view.
    """
    path = Path(path)
    if width <= 0 or height <= 0:
        raise ValueError("SVG width and height must be positive")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")

    lon = np.asarray(grid.vertex_lon, dtype=np.float64)
    lat = np.asarray(grid.vertex_lat, dtype=np.float64)
    edges = np.asarray(grid.edges, dtype=np.int64)
    if lon.ndim != 1 or lat.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("grid vertex longitude and latitude arrays must be one-dimensional")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("grid edge array must have shape (edge, 2)")

    if edges.shape[0] > max_edges:
        step = int(np.ceil(edges.shape[0] / max_edges))
        edges = edges[::step]

    segments = _projected_segments(grid, lon, lat, edges, width=width, height=height)
    path.write_text(
        _svg_document(
            grid_name=str(getattr(grid, "name", "grid")),
            segments=segments,
            width=width,
            height=height,
            stroke=stroke,
            background=background,
        )
    )
    return path


def _projected_segments(
    grid: Any,
    lon: np.ndarray,
    lat: np.ndarray,
    edges: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    x0 = lon[edges[:, 0]]
    y0 = lat[edges[:, 0]]
    x1 = lon[edges[:, 1]]
    y1 = lat[edges[:, 1]]

    if int(getattr(grid, "metadata", {}).get("grid_geometry", 0)) == 1:
        keep = np.abs(x0 - x1) <= 180.0
        x0 = x0[keep]
        y0 = y0[keep]
        x1 = x1[keep]
        y1 = y1[keep]

    x_values = np.concatenate([x0, x1])
    y_values = np.concatenate([y0, y1])
    if x_values.size == 0 or y_values.size == 0:
        return []

    padding = 16.0
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    x_scale = (width - 2.0 * padding) / max(x_max - x_min, 1.0)
    y_scale = (height - 2.0 * padding) / max(y_max - y_min, 1.0)
    scale = min(x_scale, y_scale)

    plot_width = (x_max - x_min) * scale
    plot_height = (y_max - y_min) * scale
    x_offset = (width - plot_width) * 0.5
    y_offset = (height - plot_height) * 0.5

    sx0 = x_offset + (x0 - x_min) * scale
    sx1 = x_offset + (x1 - x_min) * scale
    sy0 = height - (y_offset + (y0 - y_min) * scale)
    sy1 = height - (y_offset + (y1 - y_min) * scale)
    return list(zip(sx0.tolist(), sy0.tolist(), sx1.tolist(), sy1.tolist(), strict=True))


def _svg_document(
    *,
    grid_name: str,
    segments: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    stroke: str,
    background: str,
) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{escape(grid_name)} grid">'
        ),
        f'<rect width="100%" height="100%" fill="{escape(background)}"/>',
        f'<title>{escape(grid_name)} grid</title>',
        (
            f'<g fill="none" stroke="{escape(stroke)}" stroke-width="0.8" '
            'stroke-linecap="round" stroke-linejoin="round">'
        ),
    ]
    lines.extend(
        f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"/>'
        for x0, y0, x1, y1 in segments
    )
    lines.extend(["</g>", "</svg>", ""])
    return "\n".join(lines)
