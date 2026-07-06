# Examples

## Global Grid

```python
from grid_generator import generate_grid

grid = generate_grid("R01B03")
print(grid.name)
print(grid.dims)
```

## Planar Torus

```python
from grid_generator import TorusGridSpec, generate_grid

grid = generate_grid(TorusGridSpec(nx=32, ny=16, edge_length=1_000.0))
print(grid.metadata["grid_geometry"])
print(grid.metadata["domain_length"])
```

## Limited Area

```python
from grid_generator import LimitedAreaGridSpec, Region, generate_grid

spec = LimitedAreaGridSpec(
    parent="R02B03",
    region=Region.lonlat_box(lon_min=-20.0, lon_max=20.0, lat_min=35.0, lat_max=60.0),
    boundary_depth=2,
)
grid = generate_grid(spec, options={"max_cells": None})
print(grid.dims)
```

## NetCDF Export

```python
from grid_generator import generate_grid

grid = generate_grid("R02B02")
grid.to_netcdf("icon-grid-R02B02.nc")
```

NetCDF export requires installing the optional extra:

```bash
python -m pip install -e ".[netcdf]"
```
