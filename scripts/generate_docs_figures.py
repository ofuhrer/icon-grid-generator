"""Generate checked-in SVG figures for the documentation examples."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grid_generator import (  # noqa: E402
    ChannelGridSpec,
    LimitedAreaGridSpec,
    ParallelogramGridSpec,
    Region,
    TorusGridSpec,
    generate_grid,
)
from grid_generator.cutting import CutGridSpec, cut_grid  # noqa: E402
from grid_generator.diagnostics import check_grid  # noqa: E402
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec  # noqa: E402
from grid_generator.transforms import OptimizationOptions, optimize_grid  # noqa: E402
from grid_generator.visualization import write_svg  # noqa: E402


FIGURE_DIR = PROJECT_ROOT / "docs" / "assets" / "examples"
FigureBuilder = Callable[[Path], None]


def _global(output: Path) -> None:
    grid = generate_grid("R1B1", spring_iterations=20)
    write_svg(grid, output / "global_r1b1.svg")


def _global_netcdf(output: Path) -> None:
    grid = generate_grid("R1B1", spring_iterations=20)
    write_svg(grid, output / "global_r1b1_netcdf.svg")


def _global_raw(output: Path) -> None:
    grid = generate_grid("R1B1", optimize_global=False)
    write_svg(grid, output / "global_r1b1_raw.svg")


def _planar_torus(output: Path) -> None:
    grid = generate_grid(TorusGridSpec(nx=12, ny=6, edge_length=1_000.0))
    write_svg(grid, output / "planar_torus.svg")


def _open_planar(output: Path) -> None:
    channel = generate_grid(ChannelGridSpec(nx=8, ny=5, edge_length=1_000.0))
    parallelogram = generate_grid(
        ParallelogramGridSpec(nx=8, ny=5, edge_length=1_000.0, shear=0.25)
    )
    write_svg(channel, output / "planar_channel.svg")
    write_svg(parallelogram, output / "planar_parallelogram.svg")


def _advanced_planar(output: Path) -> None:
    stretched = generate_grid(
        StretchedTorusGridSpec(nx=8, ny=5, edge_length=1_000.0, stretch_x=1.4)
    )
    ragged = generate_grid(RaggedOrthogonalGridSpec(nx=8, ny=5, dx=1_000.0, dy=800.0))
    write_svg(stretched, output / "planar_stretched_torus.svg")
    write_svg(ragged, output / "planar_ragged_orthogonal.svg")


def _limited_area(output: Path) -> None:
    spec = LimitedAreaGridSpec(
        parent="R2B1",
        region=Region.lonlat_box(
            lon_min=-30.0,
            lon_max=30.0,
            lat_min=-20.0,
            lat_max=35.0,
        ),
        boundary_depth=1,
    )
    grid = generate_grid(spec, spring_iterations=20)
    write_svg(grid, output / "limited_area.svg")


def _cut_circle(output: Path) -> None:
    parent = generate_grid("R2B1", spring_iterations=20)
    cut = cut_grid(parent, Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0))
    write_svg(cut, output / "cut_circle.svg")


def _cut_multi_region(output: Path) -> None:
    parent = generate_grid("R2B1", spring_iterations=20)
    cut = cut_grid(
        parent,
        CutGridSpec(
            regions=(
                Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0),
                Region.lonlat_box(
                    lon_min=-20.0,
                    lon_max=20.0,
                    lat_min=-15.0,
                    lat_max=15.0,
                ),
            ),
            boundary_depth=1,
            smoothing_depth=1,
            name="CUT_MULTI",
        ),
    )
    write_svg(cut, output / "cut_multi_region.svg")


def _optimized_channel(output: Path) -> None:
    grid = generate_grid(ChannelGridSpec(nx=8, ny=5, edge_length=1_000.0))
    assert check_grid(grid).ok
    optimized = optimize_grid(grid, OptimizationOptions(iterations=2, relaxation=0.1))
    write_svg(optimized, output / "optimized_channel.svg")


BUILDERS: tuple[FigureBuilder, ...] = (
    _global,
    _global_netcdf,
    _global_raw,
    _planar_torus,
    _open_planar,
    _advanced_planar,
    _limited_area,
    _cut_circle,
    _cut_multi_region,
    _optimized_channel,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in docs figures differ from generated figures",
    )
    args = parser.parse_args()

    if args.check:
        with tempfile.TemporaryDirectory() as tmpdir:
            generated = Path(tmpdir) / "examples"
            _generate(generated)
            return _check(generated, FIGURE_DIR)

    if FIGURE_DIR.exists():
        shutil.rmtree(FIGURE_DIR)
    _generate(FIGURE_DIR)
    print(f"wrote {len(list(FIGURE_DIR.glob('*.svg')))} figures to {FIGURE_DIR}")
    return 0


def _generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for builder in BUILDERS:
        builder(output)


def _check(generated: Path, committed: Path) -> int:
    generated_files = {path.name for path in generated.glob("*.svg")}
    committed_files = {path.name for path in committed.glob("*.svg")}
    if generated_files != committed_files:
        missing = sorted(generated_files - committed_files)
        extra = sorted(committed_files - generated_files)
        if missing:
            print(f"missing docs figures: {', '.join(missing)}", file=sys.stderr)
        if extra:
            print(f"stale extra docs figures: {', '.join(extra)}", file=sys.stderr)
        return 1

    changed = [
        name
        for name in sorted(generated_files)
        if not filecmp.cmp(generated / name, committed / name, shallow=False)
    ]
    if changed:
        print(
            "stale docs figures: "
            + ", ".join(changed)
            + "\nrun `make docs-figures` and commit the result",
            file=sys.stderr,
        )
        return 1
    print(f"{len(generated_files)} docs figures are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
