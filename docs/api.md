# API Overview

## Public Entry Point

- `generate_grid(spec, options=None)` creates an `IconGrid` from an RxxByy string
  or a grid specification object.
- `parse_grid_spec(grid_name)` parses compact global grid names such as
  `R02B03`.
- `grid_uuid(grid_name, ...)` returns a stable UUID for supported grid
  parameters.

## Grid Specifications

- `GlobalGridSpec` describes spherical ICON RxxByy grids.
- `TorusGridSpec` describes planar doubly periodic triangular torus grids.
- `LimitedAreaGridSpec` extracts a region from a generated global parent grid.
- `StretchedTorusGridSpec`, `ChannelGridSpec`, `ParallelogramGridSpec`, and
  `RaggedOrthogonalGridSpec` cover additional planar variants.

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
