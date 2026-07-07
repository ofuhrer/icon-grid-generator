# Examples

## Write A Global Grid

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
grid.to_netcdf("icon_grid_R02B04.nc")
```

This is the default path for standard spherical grid files. Global grids are
optimized by default.

## Pick The Right Spec

| Need | Use |
| --- | --- |
| Standard spherical grid | `generate_grid("R2B4")` |
| Raw topology diagnostic | `generate_grid("R2B4", optimize_global=False)` |
| Periodic planar grid | `TorusGridSpec(...)` |
| Regional extract | `LimitedAreaGridSpec(...)` |
| Cut an existing grid | `cut_grid(...)` from `grid_generator.cutting` |

## Inspect An In-Memory Grid

```python
from grid_generator import generate_grid

grid = generate_grid("R2B3")
print(grid.name)
print(grid.dims)
print(grid.metadata["mean_edge_length"])
print(grid.geometry["cell_area"].shape)
```

## Disable The Safety Limit

`generate_grid()` has a safety limit to avoid accidental large allocations. Set
`max_cells=None` when a large grid is intentional.

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4", max_cells=None)
print(grid.dims)
```

## Generate A Raw Diagnostic Grid

Global grids are optimized by default. Raw grids are useful for topology tests
and diagnostics.

```python
from grid_generator import generate_grid

raw_grid = generate_grid("R2B4", optimize_global=False)
print(raw_grid.metadata["global_optimization"])
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
grid = generate_grid(spec, max_cells=None)
print(grid.dims)
```

## Cut An Existing Grid

For one region, pass the region directly:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(parent, Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0))
print(cut.dims)
```

Use `CutGridSpec` when you need multiple regions or non-default cut options:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import CutGridSpec, cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(
    parent,
    CutGridSpec(regions=Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0)),
)
print(cut.dims)
```

## Runnable Scripts

The `examples/` directory contains short scripts for the main user workflows:

- `examples/write_global_grid.py`
- `examples/write_limited_area.py`
- `examples/planar_torus.py`

NetCDF export requires installing the optional extra:

```bash
python -m pip install "icon-grid-generator[netcdf]"
```
