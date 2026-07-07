# API Overview

## Everyday API

Most users import only `generate_grid`:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4")
grid.to_netcdf("icon_grid_R02B04.nc")
```

The root import surface is intentionally small:

```python
from grid_generator import (
    generate_grid,
    IconGrid,
    IconGridOptions,
    GlobalGridSpec,
    TorusGridSpec,
    ChannelGridSpec,
    ParallelogramGridSpec,
    LimitedAreaGridSpec,
    Region,
)
```

## Grid Specifications

- `GlobalGridSpec` describes spherical ICON `R<n>B<k>` grids. Strings such as
  `"R2B4"` are shorthand for this common path.
- `LimitedAreaGridSpec(parent=..., region=..., boundary_depth=...)` extracts a
  compact regional grid from a generated global parent.
- `TorusGridSpec` describes planar doubly periodic triangular torus grids.
- `ChannelGridSpec` describes a planar triangular channel with open boundaries
  in one direction and periodic boundaries in the other.
- `ParallelogramGridSpec` describes a skewed planar triangular parallelogram.

Advanced but supported planar variants live in `grid_generator.planar`:

```python
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec
```

### Specification Signatures

| Object | Parameters |
| --- | --- |
| `GlobalGridSpec` | `root`, `bisections`, optional `name` |
| `TorusGridSpec` | `nx`, `ny`, `edge_length`, optional `name` |
| `ChannelGridSpec` | `nx`, `ny`, `edge_length`, optional `name` |
| `ParallelogramGridSpec` | `nx`, `ny`, `edge_length`, optional `shear`, `name` |
| `LimitedAreaGridSpec` | `parent`, `region`, optional `boundary_depth`, `name` |

Use `generate_grid("R2B4")` for the common global-grid case. Use explicit spec
objects when the grid family has parameters beyond the standard `R<n>B<k>`
name.

## Regions

Use `Region` constructors for limited-area extraction and cutting:

- `Region.lonlat_box(lon_min=..., lon_max=..., lat_min=..., lat_max=...)`
- `Region.circle(lon=..., lat=..., radius_degrees=...)`
- `Region.rectangle(center_lon=..., center_lat=..., width_degrees=..., height_degrees=..., angle_degrees=...)`
- `Region.polygon(((lon0, lat0), (lon1, lat1), ...))`

Longitudes and latitudes are in degrees. Region predicates select cells by cell
center.

## Options

Pass common options directly to `generate_grid()`:

```python
from grid_generator import generate_grid

grid = generate_grid("R2B4", sphere_radius=6_371_229.0)
raw_grid = generate_grid("R2B4", optimize_global=False)
large_grid = generate_grid("R2B4", max_cells=None)
```

Use `IconGridOptions` when the same configuration is reused:

```python
from grid_generator import IconGridOptions, generate_grid

options = IconGridOptions(sphere_radius=6_371_229.0, spring_iterations=2_000)
grid = generate_grid("R2B4", options=options)
```

Common options:

- `max_cells`: generation safety limit. Set to `None` for intentional large
  grids.
- `sphere_radius`: radius used for spherical metric fields.
- `optimize_global`: global grids are optimized by default. Limited-area grids
  use the same setting for their generated global parent. Planar grids do not
  support global optimization; omit the option or pass `False` for planar specs.
- `north_pole_lon`, `north_pole_lat`, and `rotation_angle_degrees`: spherical
  orientation controls.

Advanced options:

- `accelerator`: `"auto"`, `"numpy"`, or `"numba"`.
- `spring_beta` and `spring_iterations`: global spring relaxation controls.
- `indexing`: global indexing convention.
- `centre`, `subcentre`, and `number_of_grid_used`: exported metadata fields.

Prefer `sphere_radius` for physical grid metrics. The lower-level `radius`
option controls the displayed Cartesian coordinate radius and is mainly useful
for tests and visualization.

### Resource Expectations

| Grid | Cells | Edges | Vertices |
| --- | ---: | ---: | ---: |
| `R1B0` | 20 | 30 | 12 |
| `R1B1` | 80 | 120 | 42 |
| `R2B3` | 5,120 | 7,680 | 2,562 |
| `R2B4` | 20,480 | 30,720 | 10,242 |
| `R2B6` | 327,680 | 491,520 | 163,842 |

## Grid Object

`IconGrid` is the in-memory object returned by all generators. It exposes:

- `dims`: cell, edge, and vertex counts.
- `vertices`, `cells`, `edges`, `cell_edges`, and `edge_cells`: core topology.
- `lon`, `lat`, `vertex_lon`, `vertex_lat`, `edge_lon`, and `edge_lat`:
  geographic or projected coordinates.
- `geometry`: metric fields such as cell area and edge length.
- `refinement`: ICON refinement and parent-index fields.
- `metadata`: scalar grid attributes used for export and provenance.
- `to_dict()`, `to_xarray()`, and `to_netcdf(path)`: conversion helpers.

## Cutting

Cutting an existing grid is an advanced workflow kept in a focused module:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import CutGridSpec, cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(
    parent,
    CutGridSpec(regions=Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0)),
)
```

For a single-region cut, pass the region directly:

```python
from grid_generator import Region, generate_grid
from grid_generator.cutting import cut_grid

parent = generate_grid("R2B4")
cut = cut_grid(parent, Region.circle(lon=8.0, lat=47.0, radius_degrees=10.0))
```

## Diagnostics And Transforms

Diagnostics and postprocessing utilities are available from focused modules:

```python
from grid_generator.diagnostics import (
    check_grid,
    grid_statistics,
    triangle_properties,
    cell_divergence,
    cell_vorticity_fnorm,
)
from grid_generator.transforms import diffuse_grid, optimize_grid
```

Their result and option dataclasses are exported from the same submodules.

## Public API Inventory

Every name exported from `grid_generator.__all__` should appear here so public
documentation moves with API changes:

- `ChannelGridSpec`
- `GlobalGridSpec`
- `IconGrid`
- `IconGridOptions`
- `LimitedAreaGridSpec`
- `ParallelogramGridSpec`
- `Region`
- `TorusGridSpec`
- `generate_grid`
