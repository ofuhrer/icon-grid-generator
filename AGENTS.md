# Agent Notes

This repository is a standalone Python package for deterministic generation of
ICON-style triangular grids. Keep changes small, tested, and compatible with the
public API documented in `README.md` and `docs/api.md`.

## Code Map

- `src/grid_generator/grid_generator.py`: public specs, `IconGrid`, generation
  facade, metadata, UUIDs, and shared geometry/connectivity helpers.
- `src/grid_generator/_global.py`, `_geometry.py`, `_topology.py`,
  `_metrics.py`, `_refinement.py`: spherical grid pipeline.
- `src/grid_generator/_netcdf.py`, `_io.py`: ICON-style NetCDF field assembly
  and writing boundary.
- `src/grid_generator/_torus.py`, `_planar.py`, `_limited_area.py`: planar and
  regional variants.
- `src/grid_generator/cutting.py`: public cutting import surface; implementation
  remains in `grid_generator.py` and `_limited_area.py`.
- `src/grid_generator/_diagnostics.py`, `_optimization.py`: optional utilities.
- `tests/test_grid_generator.py`: global geometry/topology/refinement contract
  coverage.
- `tests/test_api.py`, `test_netcdf.py`, `test_planar_limited_area.py`,
  `test_diagnostics_transforms.py`, `test_performance.py`: focused public API,
  export, regional/planar, utility, and performance coverage.

## Repository Intelligence

- A local repowise service may be running at `http://127.0.0.1:7337`. When it
  is reachable, prefer it over codebase-memory and use it as the first-choice
  repo intelligence layer before broad manual exploration, especially for
  onboarding, generated documentation, semantic search, symbol metadata,
  dependency/call graphs, git risk, hotspots, health findings, ownership,
  decisions, and refactoring context.
- Repowise exposes FastAPI docs at `/docs` and `/redoc`, OpenAPI at
  `/openapi.json`, health at `/health`, registered repos at `/api/repos`, and
  its Codex/MCP-oriented tool surface at `/api/mcp/tools`.
- Start by calling `/api/repos` to find the active `repo_id`, then scope
  queries with that ID. Useful endpoints include `/api/search`, `/api/symbols`,
  `/api/symbols/detail`, `/api/pages`, `/api/graph/{repo_id}/callers-callees`,
  `/api/graph/{repo_id}/path`, `/api/repos/{repo_id}/overview-summary`,
  `/api/repos/{repo_id}/health/overview`, and
  `/api/repos/{repo_id}/risk/range`.
- Before trusting repowise results, compare the repo's `head_commit` from
  `/api/repos` with `git rev-parse HEAD`. If they differ, refresh repowise with
  `POST /api/repos/{repo_id}/index` for source/API changes, or
  `POST /api/repos/{repo_id}/sync` for lighter documentation syncs, then query
  again. Treat stale repowise output as orientation only.
- Use codebase-memory MCP tools (`search_graph`, `trace_path`,
  `get_code_snippet`, `query_graph`, `search_code`) only when repowise is not
  reachable, when repowise lacks the needed detail, or when a task explicitly
  needs the codebase-memory graph. Be aware that codebase-memory may be stale;
  call `index_repository` first if freshness matters or new symbols/files were
  added.
- Treat repowise as contextual intelligence, not a replacement for source
  verification. For exact behavior, verify with code snippets/local files and
  the required checks.

## Design Constraints

- Preserve deterministic output for identical specs and options.
- Validate public inputs before expensive generation.
- Keep the root import surface small. Advanced helpers such as `CutGridSpec` and
  `cut_grid` are exported from focused submodules, not from `grid_generator`.
- `generate_grid()` accepts direct keyword option overrides. Keep mapping,
  `IconGridOptions`, and keyword override semantics consistent.
- Global grids are optimized by default. Limited-area grids pass that setting to
  their generated global parent. Planar grids must reject explicit
  `optimize_global=True` and auto-disable only when the option is absent.
- Do not add runtime dependencies lightly; `numpy` is the core dependency.
- Keep NetCDF variable names, shapes, and one-based exported indices stable
  unless the change is intentional and documented.
- Geometry/topology changes must test counts, bounds, adjacency, finite numeric
  fields, and relevant metadata.
- UUID behavior is a compatibility contract.
- Keep local generated artifacts, exploratory comparison outputs, and cloned
  external sources under `tmp/`; that directory is ignored and must stay out of
  tracked docs, code, and assets.
- Do not add references to deprecated implementation paths or names in tracked
  docs, code, images, or generated assets.
- Do not commit generated `dist/`, `build/`, `site/`, cache, or `tmp/` content.

## Required Checks

Install local development extras before running checks:

```bash
python -m pip install -e ".[test,docs,netcdf,xarray]"
python -m pip install build twine
```

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

Before handing work back from an agent session, run the full local check:

```bash
make check
```

Run `make perf-check` after performance-sensitive grid-generation changes. It
uses ignored local subprocess benchmarks and is intentionally not part of the
default check because timings are hardware- and load-dependent.

If a change touches public specs, `generate_grid()`, `IconGrid`, NetCDF export,
metadata, UUIDs, or examples, update the matching README/docs/API text and tests
in the same change. Do not leave generated `site/` output or comparison files as
tracked changes.

Use `make contract-compare REF_EXE=/path/to/reference-command` only for manual
local contract checks; it depends on ignored files under `tmp/` and is not a CI
target.

## Contribution Policy

Contributions are BSD-3-Clause and require Developer Certificate of Origin
sign-off. Use `git commit -s` and keep PR descriptions explicit for changes to
grid math, topology, metrics, refinement, UUIDs, or NetCDF output.
