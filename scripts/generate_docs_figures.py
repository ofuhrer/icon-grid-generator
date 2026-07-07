"""Generate checked-in SVG figures for the documentation examples."""

from __future__ import annotations

import argparse
import filecmp
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree


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
SVG_COORDINATE_TOLERANCE = 0.1
FigureBuilder = Callable[[Path], None]


def _generate_grid(*args: object, **kwargs: object) -> Any:
    kwargs.setdefault("accelerator", "numpy")
    return generate_grid(*args, **kwargs)


def _global(output: Path) -> None:
    grid = _generate_grid("R1B1", spring_iterations=20)
    write_svg(grid, output / "global_r1b1.svg")


def _global_netcdf(output: Path) -> None:
    grid = _generate_grid("R1B1", spring_iterations=20)
    write_svg(grid, output / "global_r1b1_netcdf.svg")


def _global_raw(output: Path) -> None:
    grid = _generate_grid("R1B1", optimize_global=False)
    write_svg(grid, output / "global_r1b1_raw.svg")


def _planar_torus(output: Path) -> None:
    grid = _generate_grid(TorusGridSpec(nx=12, ny=6, edge_length=1_000.0))
    write_svg(grid, output / "planar_torus.svg")


def _open_planar(output: Path) -> None:
    channel = _generate_grid(ChannelGridSpec(nx=8, ny=5, edge_length=1_000.0))
    parallelogram = _generate_grid(
        ParallelogramGridSpec(nx=8, ny=5, edge_length=1_000.0, shear=0.25)
    )
    write_svg(channel, output / "planar_channel.svg")
    write_svg(parallelogram, output / "planar_parallelogram.svg")


def _advanced_planar(output: Path) -> None:
    stretched = _generate_grid(
        StretchedTorusGridSpec(nx=8, ny=5, edge_length=1_000.0, stretch_x=1.4)
    )
    ragged = _generate_grid(
        RaggedOrthogonalGridSpec(nx=8, ny=5, dx=1_000.0, dy=800.0)
    )
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
    grid = _generate_grid(spec, spring_iterations=20)
    write_svg(grid, output / "limited_area.svg")


def _cut_circle(output: Path) -> None:
    parent = _generate_grid("R2B1", spring_iterations=20)
    cut = cut_grid(parent, Region.circle(lon=0.0, lat=0.0, radius_degrees=35.0))
    write_svg(cut, output / "cut_circle.svg")


def _cut_multi_region(output: Path) -> None:
    parent = _generate_grid("R2B1", spring_iterations=20)
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
    grid = _generate_grid(ChannelGridSpec(nx=8, ny=5, edge_length=1_000.0))
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

    changed: list[tuple[str, str]] = []
    for name in sorted(generated_files):
        matches, reason = _svg_files_match(generated / name, committed / name)
        if not matches:
            changed.append((name, reason))

    if changed:
        print(
            "stale docs figures: "
            + ", ".join(name for name, _reason in changed)
            + "\nrun `make docs-figures` and commit the result",
            file=sys.stderr,
        )
        for name, reason in changed:
            print(f"  {name}: {reason}", file=sys.stderr)
        return 1
    print(f"{len(generated_files)} docs figures are current")
    return 0


def _svg_files_match(generated: Path, committed: Path) -> tuple[bool, str]:
    if filecmp.cmp(generated, committed, shallow=False):
        return True, "exact match"
    return _equivalent_svg(generated, committed)


def _equivalent_svg(generated: Path, committed: Path) -> tuple[bool, str]:
    try:
        generated_root = ElementTree.parse(generated).getroot()
        committed_root = ElementTree.parse(committed).getroot()
    except ElementTree.ParseError as error:
        return False, f"invalid SVG XML: {error}"

    if _svg_static_signature(generated_root) != _svg_static_signature(committed_root):
        return False, "non-line SVG structure differs"

    generated_lines = _svg_lines(generated_root)
    committed_lines = _svg_lines(committed_root)
    if len(generated_lines) != len(committed_lines):
        return (
            False,
            f"line count differs: generated={len(generated_lines)}, "
            f"committed={len(committed_lines)}",
        )

    max_delta = 0.0
    max_location = (0, 0)
    for line_index, (generated_line, committed_line) in enumerate(
        zip(generated_lines, committed_lines, strict=True)
    ):
        for coordinate_index, (generated_value, committed_value) in enumerate(
            zip(generated_line, committed_line, strict=True)
        ):
            delta = abs(generated_value - committed_value)
            if delta > max_delta:
                max_delta = delta
                max_location = (line_index, coordinate_index)

    if math.isclose(max_delta, 0.0, abs_tol=SVG_COORDINATE_TOLERANCE):
        return True, "semantic match"
    return (
        False,
        (
            f"max coordinate delta {max_delta:.3f}px at line "
            f"{max_location[0]} coordinate {max_location[1]} "
            f"(tolerance {SVG_COORDINATE_TOLERANCE:.3f}px)"
        )
    )


def _svg_static_signature(
    root: ElementTree.Element,
) -> tuple[tuple[str, tuple[tuple[str, str], ...], str], ...]:
    signature: list[tuple[str, tuple[tuple[str, str], ...], str]] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", maxsplit=1)[-1]
        if tag == "line":
            continue
        signature.append(
            (
                tag,
                tuple(sorted(element.attrib.items())),
                (element.text or "").strip(),
            )
        )
    return tuple(signature)


def _svg_lines(root: ElementTree.Element) -> list[tuple[float, float, float, float]]:
    lines: list[tuple[float, float, float, float]] = []
    for element in root.iter():
        if element.tag.rsplit("}", maxsplit=1)[-1] != "line":
            continue
        endpoint_a = (float(element.attrib["x1"]), float(element.attrib["y1"]))
        endpoint_b = (float(element.attrib["x2"]), float(element.attrib["y2"]))
        start, end = sorted((endpoint_a, endpoint_b))
        lines.append((*start, *end))
    return sorted(
        lines,
        key=lambda line: (
            round(line[0], 3),
            round(line[1], 3),
            round(line[2], 3),
            round(line[3], 3),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
