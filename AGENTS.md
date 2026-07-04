# Agent Notes

This repository is a standalone Python package for deterministic generation of
ICON-style triangular grids. Keep changes small, tested, and compatible with the
public API documented in `README.md` and `docs/api.md`.

## Code Map

- `src/grid_generator/grid_generator.py`: public specs, `IconGrid`, generation
  facade, metadata, UUIDs, and NetCDF field assembly.
- `src/grid_generator/_geometry.py`, `_topology.py`, `_metrics.py`,
  `_refinement.py`: spherical grid pipeline.
- `src/grid_generator/_torus.py`, `_planar.py`, `_limited_area.py`: planar and
  regional variants.
- `src/grid_generator/_diagnostics.py`, `_optimization.py`: optional utilities.
- `tests/test_grid_generator.py`: contract and regression coverage.

## Design Constraints

- Preserve deterministic output for identical specs and options.
- Validate public inputs before expensive generation.
- Do not add runtime dependencies lightly; `numpy` is the core dependency.
- Keep NetCDF variable names, shapes, and one-based exported indices stable
  unless the change is intentional and documented.
- Geometry/topology changes must test counts, bounds, adjacency, finite numeric
  fields, and relevant metadata.
- UUID behavior is a compatibility contract.
- Do not commit generated `dist/`, `build/`, `site/`, cache, or `tmp/` content.

## Required Checks

Run the focused checks for normal code changes:

```bash
make check
```

For packaging or release-facing changes, also run:

```bash
make package
```

For docs-only changes, `make docs` is sufficient. If `make` is unavailable, use
the commands in the `Makefile` directly.

## Contribution Policy

Contributions are BSD-3-Clause and require Developer Certificate of Origin
sign-off. Use `git commit -s` and keep PR descriptions explicit for changes to
grid math, topology, metrics, refinement, UUIDs, or NetCDF output.
