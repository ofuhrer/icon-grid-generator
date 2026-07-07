# Changelog

## Unreleased

## 0.3.0 - 2026-07-07

- Keep the root import surface focused while moving advanced cutting helpers to
  `grid_generator.cutting`.
- Allow common `generate_grid()` options to be passed directly as keyword
  arguments.
- Add lightweight SVG grid visualization via `grid_generator.visualization`.
- Expand copy-pasteable examples for global, raw, planar, limited-area,
  cutting, diagnostics, transforms, NetCDF export, and visualization workflows.
- Clarify and test `optimize_global` behavior across global, limited-area, and
  planar grid specs.
- Preserve planar geometry when optimizing or diffusing cut grids from planar
  parents.
- Add fast mathematical correctness matrix coverage across all supported grid
  families, region predicates, cut modes, transforms, metric scaling, and
  representative NetCDF exports.
- Add example scripts, type-marker packaging, and documentation updates for the
  refactored public API.

## 0.2.1 - 2026-07-05

Patch release for relaxed global grid generation.

- Add optional spring relaxation for global spherical grids with unchanged
  topology and recomputed metrics.
- Expose `GlobalOptimizationOptions` and `optimize_global_grid()` in the public
  API.

## 0.2.0 - 2026-07-05

Expanded grid generation, validation, and release automation.

- Add triangular planar variants for stretched periodic, channel, parallelogram,
  and ragged orthogonal grids.
- Add geometry optimization and diffusion transforms.
- Add region-based local-area cutting with parent-index metadata.
- Add grid diagnostics, statistics, triangle properties, divergence, and
  normalized vorticity helpers.
- Improve ICON-style NetCDF metadata, refinement fields, ordering, and
  large-grid safety checks.
- Add optional Numba acceleration support and CI coverage for accelerated and
  non-accelerated execution paths.
- Add documentation publishing, contributor guidance, and drift checks for
  documentation, badges, API exports, and the Python test matrix.

## 0.1.0 - 2026-07-04

Initial public release.

- Generate global spherical ICON `R<n>B<k>` grids.
- Generate planar doubly periodic torus grids.
- Extract limited-area grids from generated global parent grids.
- Export ICON-style NetCDF grid files with optional `netCDF4` support.
- Provide the public grid-spec API: `GlobalGridSpec`, `LimitedAreaGridSpec`,
  `TorusGridSpec`, and `generate_grid()`.
