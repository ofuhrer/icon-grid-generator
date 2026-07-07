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
    projection: str = "map",
    azimuth_degrees: float = -35.0,
    elevation_degrees: float = 25.0,
    stroke: str = "#1f2937",
    background: str = "#ffffff",
) -> Path:
    """Write a simple SVG grid plot for a generated grid.

    The helper intentionally has no plotting dependency. Use
    ``projection="map"`` for a longitude/latitude or planar map view, or
    ``projection="3d"`` for a static pseudo-3D view useful in documentation.
    For global map-view grids, edges that cross the dateline are omitted to
    avoid wraparound artifacts in the equirectangular view.
    """
    path = Path(path)
    if width <= 0 or height <= 0:
        raise ValueError("SVG width and height must be positive")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")
    if projection not in {"map", "3d"}:
        raise ValueError('SVG projection must be "map" or "3d"')

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

    if projection == "3d":
        segments = _projected_segments_3d(
            grid,
            lon,
            lat,
            edges,
            width=width,
            height=height,
            azimuth_degrees=azimuth_degrees,
            elevation_degrees=elevation_degrees,
        )
        frame = "sphere" if _is_spherical(grid) else "plane"
    else:
        segments = _projected_segments(grid, lon, lat, edges, width=width, height=height)
        frame = "map"
    path.write_text(
        _svg_document(
            grid_name=str(getattr(grid, "name", "grid")),
            segments=segments,
            width=width,
            height=height,
            frame=frame,
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


def _projected_segments_3d(
    grid: Any,
    lon: np.ndarray,
    lat: np.ndarray,
    edges: np.ndarray,
    *,
    width: int,
    height: int,
    azimuth_degrees: float,
    elevation_degrees: float,
) -> list[tuple[float, float, float, float, float]]:
    if _is_spherical(grid):
        x, y, z = _spherical_xyz(lon, lat)
        x, y, depth = _rotated_view(
            x,
            y,
            z,
            azimuth_degrees=azimuth_degrees,
            elevation_degrees=elevation_degrees,
        )
        return _scaled_3d_segments(x, y, depth, edges, width=width, height=height, hide_back=True)

    x, y = _normalized_xy(lon, lat)
    x, y, depth = _rotated_view(
        x,
        y,
        np.zeros_like(x),
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
    )
    return _scaled_3d_segments(x, y, depth, edges, width=width, height=height, hide_back=False)


def _is_spherical(grid: Any) -> bool:
    return int(getattr(grid, "metadata", {}).get("grid_geometry", 0)) in {1, 3}


def _spherical_xyz(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    return cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)


def _normalized_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_mid = (float(np.min(x)) + float(np.max(x))) * 0.5
    y_mid = (float(np.min(y)) + float(np.max(y))) * 0.5
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    return (x - x_mid) / scale, (y - y_mid) / scale


def _rotated_view(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    azimuth_degrees: float,
    elevation_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)
    x_az = cos_az * x - sin_az * y
    y_az = sin_az * x + cos_az * y

    cos_el = np.cos(elevation)
    sin_el = np.sin(elevation)
    y_view = cos_el * y_az - sin_el * z
    depth = sin_el * y_az + cos_el * z
    return x_az, y_view, depth


def _scaled_3d_segments(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    edges: np.ndarray,
    *,
    width: int,
    height: int,
    hide_back: bool,
) -> list[tuple[float, float, float, float, float]]:
    x0 = x[edges[:, 0]]
    y0 = y[edges[:, 0]]
    d0 = depth[edges[:, 0]]
    x1 = x[edges[:, 1]]
    y1 = y[edges[:, 1]]
    d1 = depth[edges[:, 1]]

    if hide_back:
        keep = np.maximum(d0, d1) >= -0.05
        x0 = x0[keep]
        y0 = y0[keep]
        d0 = d0[keep]
        x1 = x1[keep]
        y1 = y1[keep]
        d1 = d1[keep]
    if x0.size == 0:
        return []

    sx0, sy0, sx1, sy1 = _scale_to_view(x0, y0, x1, y1, width=width, height=height)
    avg_depth = (d0 + d1) * 0.5
    depth_min = float(np.min(avg_depth))
    depth_range = max(float(np.ptp(avg_depth)), 1.0e-12)
    shade = 0.35 + 0.65 * ((avg_depth - depth_min) / depth_range)
    order = np.argsort(avg_depth)
    return [
        (
            float(sx0[index]),
            float(sy0[index]),
            float(sx1[index]),
            float(sy1[index]),
            float(shade[index]),
        )
        for index in order
    ]


def _scale_to_view(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.concatenate([x0, x1])
    y_values = np.concatenate([y0, y1])
    padding = 24.0
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    x_scale = (width - 2.0 * padding) / max(x_max - x_min, 1.0e-12)
    y_scale = (height - 2.0 * padding) / max(y_max - y_min, 1.0e-12)
    scale = min(x_scale, y_scale)

    plot_width = (x_max - x_min) * scale
    plot_height = (y_max - y_min) * scale
    x_offset = (width - plot_width) * 0.5
    y_offset = (height - plot_height) * 0.5

    sx0 = x_offset + (x0 - x_min) * scale
    sx1 = x_offset + (x1 - x_min) * scale
    sy0 = height - (y_offset + (y0 - y_min) * scale)
    sy1 = height - (y_offset + (y1 - y_min) * scale)
    return sx0, sy0, sx1, sy1


def _svg_document(
    *,
    grid_name: str,
    segments: list[tuple[float, float, float, float] | tuple[float, float, float, float, float]],
    width: int,
    height: int,
    frame: str,
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
    ]
    if frame == "sphere":
        radius = min(width, height) * 0.44
        lines.append(
            f'<circle cx="{width * 0.5:.3f}" cy="{height * 0.5:.3f}" '
            f'r="{radius:.3f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.0"/>'
        )
    elif frame == "plane":
        lines.append(
            f'<ellipse cx="{width * 0.5:.3f}" cy="{height * 0.72:.3f}" '
            f'rx="{width * 0.32:.3f}" ry="{height * 0.08:.3f}" fill="#e5e7eb" '
            'opacity="0.55"/>'
        )
    lines.append(
        f'<g fill="none" stroke="{escape(stroke)}" stroke-width="0.8" '
        'stroke-linecap="round" stroke-linejoin="round">'
    )
    lines.extend(
        _svg_line(segment)
        for segment in segments
    )
    lines.extend(["</g>", "</svg>", ""])
    return "\n".join(lines)


def _svg_line(
    segment: tuple[float, float, float, float] | tuple[float, float, float, float, float],
) -> str:
    x0, y0, x1, y1 = segment[:4]
    if len(segment) == 5:
        return (
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
            f'stroke-opacity="{segment[4]:.3f}"/>'
        )
    return f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}"/>'
