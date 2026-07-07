# ICON Grid Generator

[![Tests](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/test.yml/badge.svg)](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/test.yml)
[![Docs](https://github.com/ofuhrer/icon-grid-generator/actions/workflows/docs.yml/badge.svg)](https://ofuhrer.github.io/icon-grid-generator/)
[![PyPI](https://img.shields.io/pypi/v/icon-grid-generator.svg)](https://pypi.org/project/icon-grid-generator/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](.github/workflows/test.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

Pure Python generation of ICON-style triangular grids.

![Global ICON grid resolutions](docs/assets/global-icon-grid-series.png)

ICON Grid Generator creates spherical ICON `R<n>B<k>` grids, planar triangular
grids, limited-area extracts, and ICON-style NetCDF files without depending on
ICON model runtimes or stencil frameworks.

## Quick Start

Most users only need `generate_grid()`:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
print(grid.name)
print(grid.dims)
grid.to_netcdf("icon_grid_R02B04.nc")
```

Example output:

```text
R02B04
{'cell': 20480, 'vertex': 10242, 'edge': 30720}
```

Global grids are optimized by default and are suitable for normal ICON-style
grid-file use. Use `optimize_global=False` only when you explicitly need the
raw bisection topology for diagnostics or tests.

## What You Can Generate

- Global spherical ICON grids from names such as `R2B4` or `R02B04`.
- Planar triangular torus, channel, and parallelogram grids for experiments.
- Limited-area grids extracted from generated global parent grids.
- ICON-style NetCDF grid files when the optional `netCDF4` dependency is
  installed.
- In-memory topology, geometry, metric, refinement, and metadata arrays for
  plotting, diagnostics, and downstream conversion.

## Which Grid Should I Use?

| Goal | Use |
| --- | --- |
| Standard spherical grid file | `generate_grid("R2B4")` |
| Raw topology checks | `generate_grid("R2B4", optimize_global=False)` |
| Periodic planar experiment | `TorusGridSpec(...)` |
| Open planar experiment | `ChannelGridSpec(...)` or `ParallelogramGridSpec(...)` |
| Regional extract from a global parent | `LimitedAreaGridSpec(...)` |
| Cut an existing in-memory grid | `grid_generator.cutting.cut_grid(...)` |

## Installation

Install from PyPI:

```bash
python -m pip install "icon-grid-generator[netcdf]"
```

From a local checkout:

```bash
python -m pip install -e .
```

Install optional NetCDF and xarray support with:

```bash
python -m pip install -e ".[netcdf,xarray]"
```

Install optional Numba acceleration support with:

```bash
python -m pip install -e ".[accelerate]"
```

Install development dependencies with:

```bash
python -m pip install -e ".[test,docs]"
```

## Grid Naming

The ICON documentation describes grid file names with the generic nomenclature
[`R<n>B<k>`](https://docs.icon-model.org/documentation/buildrun/buildrun_input_data.html),
where `n` is the number of root divisions and `k` is the number of subsequent
bisections. ICON examples also commonly use zero-padded grid file names such as
`R02B06`. This package accepts both compact names (`R2B6`) and zero-padded names
(`R02B06`), then stores labels and metadata in the zero-padded form.

## Resource Expectations

Global grid size grows by a factor of four with each bisection:

| Grid | Cells | Edges | Vertices |
| --- | ---: | ---: | ---: |
| `R1B0` | 20 | 30 | 12 |
| `R1B1` | 80 | 120 | 42 |
| `R2B3` | 5,120 | 7,680 | 2,562 |
| `R2B4` | 20,480 | 30,720 | 10,242 |
| `R2B6` | 327,680 | 491,520 | 163,842 |

`generate_grid()` has a default safety limit of 2,000,000 cells. Set
`max_cells=None` only when the allocation is intentional.

## Common Recipes

Disable the default safety limit when a large allocation is intentional:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4", max_cells=None)
```

Generate a raw diagnostic grid without global optimization:

```python
raw_grid = generate_grid("R2B4", optimize_global=False)
```

Generate a planar torus grid:

```python
from grid_generator import TorusGridSpec, generate_grid

grid = generate_grid(TorusGridSpec(nx=32, ny=16, edge_length=1_000.0))
print(grid.metadata["grid_geometry"])
print(grid.metadata["domain_length"])
```

Extract a limited-area grid from a generated global parent:

```python
from grid_generator import LimitedAreaGridSpec, Region, generate_grid

spec = LimitedAreaGridSpec(
    parent="R02B03",
    region=Region.lonlat_box(lon_min=-20.0, lon_max=20.0, lat_min=35.0, lat_max=60.0),
    boundary_depth=2,
)
grid = generate_grid(spec, max_cells=None)
print(grid.dims)
```

Cut an existing grid with advanced region predicates:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import CutGridSpec, cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(
    parent,
    CutGridSpec(regions=Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0)),
)
```

For the common single-region case, pass the region directly:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(parent, Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0))
```

NetCDF export requires the `netcdf` optional extra. See
[`examples/write_global_grid.py`](examples/write_global_grid.py),
[`examples/write_limited_area.py`](examples/write_limited_area.py), and
[`examples/planar_torus.py`](examples/planar_torus.py) for runnable scripts.

## Documentation

The minimal documentation lives in [docs](docs):

- [Overview](docs/index.md)
- [Examples](docs/examples.md)
- [API overview](docs/api.md)
- [Design notes](docs/design.md)
- [Repository metadata](docs/repository-metadata.md)

To preview the docs locally:

```bash
mkdocs serve
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, review
expectations, and domain-specific requirements for grid math and NetCDF changes.

Run the checks used by CI:

```bash
make check
```

The package is laid out as a standalone Python project. If this directory is
split out of a larger checkout, keep `.github/`, `docs/`, `CITATION.cff`,
`CHANGELOG.md`, `LICENSE`, `README.md`, `mkdocs.yml`, `pyproject.toml`, `src/`,
and `tests/` at the new repository root.

## Citation

If you use ICON Grid Generator in published work, cite it using
[CITATION.cff](CITATION.cff). For research releases, connect the public GitHub
repository to Zenodo before creating a GitHub Release so a DOI can be minted.

## Release History

See [CHANGELOG.md](CHANGELOG.md).

## License

ICON Grid Generator is distributed under the [BSD 3-Clause License](LICENSE).
