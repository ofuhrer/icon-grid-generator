from __future__ import annotations

import numpy as np
import pytest

from grid_generator import ChannelGridSpec, LimitedAreaGridSpec, Region, TorusGridSpec, generate_grid
from grid_generator.cutting import cut_grid
from grid_generator.visualization import _projected_segments, write_svg


@pytest.mark.parametrize(
    "grid",
    [
        generate_grid("R01B01", spring_iterations=5),
        generate_grid("R01B01", optimize_global=False),
        generate_grid(TorusGridSpec(nx=4, ny=3, edge_length=1.0)),
        generate_grid(ChannelGridSpec(nx=4, ny=3, edge_length=1.0)),
        generate_grid(
            LimitedAreaGridSpec(
                parent="R02B01",
                region=Region.lonlat_box(
                    lon_min=-20.0,
                    lon_max=20.0,
                    lat_min=-20.0,
                    lat_max=20.0,
                ),
            ),
            spring_iterations=5,
        ),
    ],
)
def test_write_svg_creates_edge_plot_for_main_grid_modes(grid, tmp_path):
    path = write_svg(grid, tmp_path / f"{grid.name}.svg", max_edges=200)
    text = path.read_text()

    assert path == tmp_path / f"{grid.name}.svg"
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<svg " in text
    assert "<line " in text
    assert grid.name in text


def test_write_svg_creates_edge_plot_for_cut_grid(tmp_path):
    parent = generate_grid("R02B01", spring_iterations=5)
    cut = cut_grid(parent, Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0))

    path = write_svg(cut, tmp_path / "cut.svg")
    text = path.read_text()

    assert path.name == "cut.svg"
    assert "CUT_GRID grid" in text
    assert text.count("<line ") > 0


def test_write_svg_rejects_invalid_render_options(tmp_path):
    grid = generate_grid("R01B00")

    with pytest.raises(ValueError, match="width and height"):
        write_svg(grid, tmp_path / "bad.svg", width=0)
    with pytest.raises(ValueError, match="max_edges"):
        write_svg(grid, tmp_path / "bad.svg", max_edges=0)


@pytest.mark.parametrize(
    "grid",
    [
        generate_grid(TorusGridSpec(nx=8, ny=5, edge_length=1.0)),
        generate_grid(ChannelGridSpec(nx=8, ny=5, edge_length=1.0)),
    ],
)
def test_periodic_svg_projection_omits_wraparound_segments(grid):

    segments = _projected_segments(
        grid,
        grid.vertex_lon,
        grid.vertex_lat,
        grid.edges,
        width=800,
        height=500,
    )
    lengths = [
        ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        for x0, y0, x1, y1 in segments
    ]

    assert max(lengths) <= 2.0 * float(np.median(lengths))
