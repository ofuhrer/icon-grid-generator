# ICON Grid Generator

Pure Python generation of ICON-style triangular grids.

The package provides spherical RxxByy grids, planar doubly periodic torus grids,
limited-area grids extracted from generated global grids, and optional NetCDF
export. It has no dependency on model runtimes or stencil frameworks.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

Install optional NetCDF and xarray support with:

```bash
python -m pip install -e ".[netcdf,xarray]"
```

## Usage

Generate a global spherical grid:

```python
from grid_generator import GlobalGridSpec, generate_grid

grid = generate_grid(GlobalGridSpec(root=2, bisections=3))
print(grid.dims)
```

Generate a planar torus grid:

```python
from grid_generator import TorusGridSpec, generate_grid

grid = generate_grid(TorusGridSpec(nx=32, ny=16, edge_length=1_000.0))
print(grid.metadata["grid_geometry"])
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
```

For global grids, the compact RxxByy string remains available as shorthand:

```python
grid = generate_grid("R02B03")
```

Write an ICON-style NetCDF file:

```python
grid.to_netcdf("grid.nc")
```

NetCDF export requires the `netcdf` optional extra.

## Release History

See [CHANGELOG.md](CHANGELOG.md).
