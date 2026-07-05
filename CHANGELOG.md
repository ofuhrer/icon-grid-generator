# Changelog

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
