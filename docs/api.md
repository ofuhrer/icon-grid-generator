# API Overview

## Public Entry Point

- `generate_grid(spec, options=None)` creates an `IconGrid` from an ICON
  `R<n>B<k>` string or a grid specification object.
- `parse_grid_spec(grid_name)` parses compact global grid names such as
  `R2B3` and returns canonical zero-padded names such as `R02B03`.
- `grid_uuid(grid_name, ...)` returns a stable UUID for supported grid
  parameters.

## Grid Specifications

- `GlobalGridSpec` describes spherical ICON `R<n>B<k>` grids.
- `TorusGridSpec` describes planar doubly periodic triangular torus grids.
- `LimitedAreaGridSpec` extracts a region from a generated global parent grid.
- `StretchedTorusGridSpec`, `ChannelGridSpec`, `ParallelogramGridSpec`, and
  `RaggedOrthogonalGridSpec` cover additional planar variants.

## Options

- `IconGridOptions(accelerator="auto")` controls optional acceleration. Use
  `"numpy"` for the reference NumPy implementation or `"numba"` to require
  experimental Numba acceleration. The default `"auto"` uses the reference path
  for small grids and may use Numba for large parent-provenance lookups when
  Numba is installed.
- `GlobalGridOptions(...)` configures global spherical grid construction,
  including spring beta, iteration limit, pole placement, rotation, indexing
  mode, and exported centre/subcentre metadata.
- `IconGridOptions(global_optimization="spring")` or
  `GlobalOptimizationOptions(method="spring", ...)` enables spring-relaxed
  global spherical grids with unchanged topology and recomputed metrics. This is
  the default for global spherical grids; use `global_optimization="none"` only
  for diagnostics or raw topology checks.

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

## Diagnostics And Postprocessing

- `check_grid(grid)` validates basic topology and geometry consistency.
- `grid_statistics(grid)` summarizes counts, boundary edges, areas, and edge
  lengths.
- `triangle_properties(grid)` returns per-cell triangle metrics.
- `optimize_grid(grid, options=None)` and `diffuse_grid(grid, options=None)`
  return geometry-transformed copies with unchanged topology.

## Public API Inventory

Every name exported from `grid_generator.__all__` should appear here so public
documentation moves with API changes:

- `ChannelGridSpec`
- `CircleRegion`
- `CutGridSpec`
- `DiffusionOptions`
- `GridCheckResult`
- `GridStatistics`
- `GlobalGridOptions`
- `GlobalOptimizationOptions`
- `IconGrid`
- `IconGridOptions`
- `GlobalGridSpec`
- `LimitedAreaGridSpec`
- `LonLatBoxRegion`
- `OptimizationOptions`
- `OrientedRectangleRegion`
- `ParallelogramGridSpec`
- `PolygonRegion`
- `RaggedOrthogonalGridSpec`
- `StretchedTorusGridSpec`
- `TriangleProperties`
- `TorusGridSpec`
- `cell_divergence`
- `cell_vorticity_fnorm`
- `check_grid`
- `cut_grid`
- `diffuse_grid`
- `generate_grid`
- `grid_statistics`
- `optimize_global_grid`
- `optimize_grid`
- `triangle_properties`
