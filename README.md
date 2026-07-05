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

## Features

- Generate global spherical ICON grids from names such as `R2B3`, canonicalized
  to zero-padded names such as `R02B03`.
- Generate spring-relaxed global grids for improved metric quality.
- Generate planar doubly periodic torus grids and additional planar variants.
- Extract limited-area grids from generated global parent grids.
- Export ICON-style NetCDF grid files with optional `netCDF4` support.
- Inspect in-memory topology, connectivity, geometry, refinement, and metadata
  arrays for plotting or downstream conversion.
- Run lightweight diagnostics and deterministic geometry postprocessing.

## Installation

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

## Quick Start

Generate a global spherical grid:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B3")
print(grid.name)
print(grid.dims)
```

Example output:

```text
R02B03
{'cell': 5120, 'vertex': 2562, 'edge': 7680}
```

Generate a spring-relaxed global grid:

```python
grid = generate_grid("R2B3", options={"max_cells": None, "global_optimization": "spring"})
print(grid.metadata["global_optimization"])
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
from grid_generator import LimitedAreaGridSpec, generate_grid

spec = LimitedAreaGridSpec(
    "R02B03",
    lon_min=-20.0,
    lon_max=20.0,
    lat_min=35.0,
    lat_max=60.0,
    boundary_depth=2,
)
grid = generate_grid(spec, options={"max_cells": None})
print(grid.dims)
```

Write an ICON-style NetCDF file:

```python
grid.to_netcdf("grid.nc")
```

NetCDF export requires the `netcdf` optional extra.

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
python -m ruff check .
python -m pytest
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
