# API Overview

## Root API

- `generate_grid(spec, options=None)` creates an `IconGrid` from an ICON
  `R<n>B<k>` string or a grid specification object.
- `IconGridOptions(...)` configures generation limits, acceleration, spherical
  radius, spring relaxation, rotation, indexing, and exported metadata.
- `IconGrid` is the in-memory grid object returned by all generators.

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
    CutGridSpec,
    Region,
)
```

## Grid Specifications

- `GlobalGridSpec` describes spherical ICON `R<n>B<k>` grids. Strings such as
  `"R2B4"` are shorthand for this common path.
- `TorusGridSpec` describes planar doubly periodic triangular torus grids.
- `ChannelGridSpec` describes a planar triangular channel with open boundaries
  in one direction and periodic boundaries in the other.
- `ParallelogramGridSpec` describes a skewed planar triangular parallelogram.
- `LimitedAreaGridSpec(parent=..., region=..., boundary_depth=...)` extracts a
  compact regional grid from a generated global parent.
- `CutGridSpec(regions=..., mode=..., boundary_depth=...)` configures cutting
  of an existing grid using one or more region predicates. Use it with
  `grid_generator.cutting.cut_grid`.

Advanced but supported planar variants live in `grid_generator.planar`:

```python
from grid_generator.planar import RaggedOrthogonalGridSpec, StretchedTorusGridSpec
```

## Regions

Use `Region` constructors for limited-area extraction and cutting:

- `Region.lonlat_box(lon_min=..., lon_max=..., lat_min=..., lat_max=...)`
- `Region.circle(lon=..., lat=..., radius_degrees=...)`
- `Region.rectangle(lon=..., lat=..., width_degrees=..., height_degrees=..., angle_degrees=...)`
- `Region.polygon(((lon0, lat0), (lon1, lat1), ...))`

## Options

`IconGridOptions` is a flat dataclass. Common fields are:

- `max_cells`: generation safety limit.
- `accelerator`: `"auto"`, `"numpy"`, or `"numba"`.
- `sphere_radius`: spherical grid radius used for metric fields.
- `optimize_global`: enable staged spring relaxation for global grids.
- `spring_beta` and `spring_iterations`: global spring relaxation controls.
- `north_pole_lon`, `north_pole_lat`, and `rotation_angle_degrees`: spherical
  orientation controls.
- `indexing`: global indexing convention.
- `centre`, `subcentre`, and `number_of_grid_used`: exported metadata fields.

Use `optimize_global=False` only for raw topology diagnostics or tests.

## Grid Object

`IconGrid` exposes:

- `dims`: cell, edge, and vertex counts.
- `vertices`, `cells`, `edges`, `cell_edges`, and `edge_cells`: core topology.
- `lon`, `lat`, `vertex_lon`, `vertex_lat`, `edge_lon`, and `edge_lat`:
  geographic or projected coordinates.
- `geometry`: metric fields such as cell area and edge length.
- `refinement`: ICON refinement and parent-index fields.
- `metadata`: scalar grid attributes used for export and provenance.
- `to_dict()`, `to_xarray()`, and `to_netcdf(path)`: conversion helpers.

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
- `CutGridSpec`
- `GlobalGridSpec`
- `IconGrid`
- `IconGridOptions`
- `LimitedAreaGridSpec`
- `ParallelogramGridSpec`
- `Region`
- `TorusGridSpec`
- `generate_grid`
