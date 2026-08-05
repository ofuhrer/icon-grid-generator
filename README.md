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

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
print(grid.name)
print(grid.dims)
grid.to_netcdf("icon_grid_R02B04.nc")
```

```text
R02B04
{'cell': 20480, 'vertex': 10242, 'edge': 30720}
```

Global grids are optimized by default. Pass `optimize_global=False` only when
raw bisection geometry is required for diagnostics or comparisons.

## What You Can Generate

- Global spherical ICON grids from standard `R<n>B<k>` names.
- Planar triangular torus, channel, parallelogram, stretched, and ragged grids.
- Limited-area grids extracted from generated global parents.
- Region-based cuts of existing spherical or planar grids.
- ICON-style NetCDF files and complete in-memory xarray datasets.
- Geometry, topology, metric, refinement, diagnostic, and visualization data.

## Installation

Python 3.10 or newer is required. Install the common NetCDF and xarray support
from PyPI with:

```bash
python -m pip install "icon-grid-generator[netcdf,xarray]"
```

The base package depends only on NumPy and can instead be installed with
`python -m pip install icon-grid-generator`.

## Documentation

The complete documentation is published at
[ofuhrer.github.io/icon-grid-generator](https://ofuhrer.github.io/icon-grid-generator/):

- [Examples](https://ofuhrer.github.io/icon-grid-generator/examples/)
- [API and usage guide](https://ofuhrer.github.io/icon-grid-generator/api/)
- [Design notes and limitations](https://ofuhrer.github.io/icon-grid-generator/design/)
- [Changelog](CHANGELOG.md)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution guidance.
Run the complete local check with:

```bash
python -m pip install -e ".[test,docs,netcdf,xarray]"
make check
```

## Citation

If you use ICON Grid Generator in published work, cite it using
[CITATION.cff](CITATION.cff).

## License

ICON Grid Generator is distributed under the [BSD 3-Clause License](LICENSE).
