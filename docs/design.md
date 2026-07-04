# Design Notes

ICON Grid Generator builds in-memory `IconGrid` objects through a small,
deterministic pipeline:

1. Parse and validate a grid spec plus `IconGridOptions`.
2. Build geometry: vertices, cells, centers, and lon/lat coordinates.
3. Build topology: edges, cell-edge relations, edge-cell relations, and ICON
   connectivity tables.
4. Build metrics: cell areas, edge lengths, dual quantities, and normal vectors.
5. Build refinement/provenance fields.
6. Assemble metadata, UUIDs, conversion helpers, and optional NetCDF output.

## Compatibility Contracts

- Public grid specs and `generate_grid()` are the main API.
- `IconGrid.dims` and array shapes must remain predictable from the spec.
- Internal topology arrays are zero-based; exported NetCDF index fields are
  one-based where ICON expects that convention.
- Metadata keys used by UUIDs, NetCDF export, and examples should not drift
  accidentally.
- Grid UUIDs must stay stable for unchanged canonical inputs.

## Feature Boundaries

- The package is Python API first. Keep command wrappers and workflow glue out
  unless they support an existing public API use case.
- Global, planar, limited-area, optimization, diffusion, diagnostics, and
  NetCDF export features should share the `IconGrid` data model.
- Triangular grids are the supported cell family. Add other cell families only
  with explicit public API, NetCDF, and diagnostic contracts.
- Ragged planar grids are deterministic Python variants; test structural
  validity and exported contracts rather than assuming metric identity with
  regular planar grids.
- Parent/provenance indices belong in `IconGrid.refinement`; metadata should
  carry descriptive scalar attributes only.

## Testing Expectations

Changes to geometry, topology, metrics, refinement, limited-area extraction, or
NetCDF output should include tests for the relevant contract:

- expected cell, edge, and vertex counts
- index bounds and missing-neighbor sentinels
- finite numeric geometry and positive areas/lengths where applicable
- parent/provenance index validity
- exported NetCDF dimensions, variables, and metadata

Use the smallest grid that proves the behavior. Larger grids are useful only for
representative sanity checks.
