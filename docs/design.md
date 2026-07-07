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
- `grid_generator.py` is the public facade. Keep large implementation concerns
  in focused private modules such as `_global.py`, `_netcdf.py`, `_planar.py`,
  and `_limited_area.py`; preserve thin private aliases only where internal
  builders/tests still rely on them.
- Triangular grids are the supported cell family. Add other cell families only
  with explicit public API, NetCDF, and diagnostic contracts.
- Ragged planar grids are deterministic Python variants; test structural
  validity and exported contracts rather than assuming metric identity with
  regular planar grids.
- Parent/provenance indices belong in `IconGrid.refinement`; metadata should
  carry descriptive scalar attributes only.

## Architectural Decisions

- Global grid generation uses staged spring relaxation by default; raw
  bisection remains available with `optimize_global=False` for diagnostics and
  topology checks.
- Optional Numba acceleration is an implementation detail selected through
  `IconGridOptions.accelerator`; NumPy remains the required baseline.
- UUIDs use deterministic UUIDv5 payloads derived from canonical specs and
  options. Any payload change is a compatibility change.
- NetCDF export is an internal module boundary. Public users should call
  `IconGrid.to_netcdf(path)`.
- Performance checks live behind `make perf-check` and are intentionally
  separate from default CI-style checks because runtime varies with local load.

## Limitations

- Connectivity and NetCDF index fields use signed 32-bit integer arrays. Global
  grids up to current large operational scales such as `R02B11` are within that
  range; generation fails early when cells, edges, or vertices would exceed the
  int32 index limit.
- Global bisection parent/provenance fields are tracked structurally during
  refinement. Some defensive fallback paths can still use rounded coordinate
  matching when geometry is constructed outside the normal global pipeline.
- Spherical metrics use double-precision trigonometric formulas. They are
  appropriate for supported resolutions, but extremely small triangles can make
  angle-sum area formulas and `arccos`-based distances more sensitive to
  floating-point cancellation.
- The implementation assumes closed global triangular meshes have vertex
  valence at most six. Limited-area and planar grids use separate open-mesh
  paths where boundary sentinels are expected.

## Performance and Scaling

For large global `R<n>B<k>` grids, the useful scaling variable is the effective
refinement frequency

```text
f = n * 2^k
```

The main asymptotic behavior follows directly from `f`:

```text
cells    = 20 * f^2      = 20 * n^2 * 4^k
edges    = 30 * f^2      = 30 * n^2 * 4^k
vertices = 10 * f^2 + 2  = 10 * n^2 * 4^k + 2
```

Generation time, peak memory, and NetCDF file size are therefore all expected
to scale approximately as `O(n^2 * 4^k)` for sufficiently large global grids.
Equivalently, each additional bisection level roughly multiplies work and
output size by four.

The measured single-process generation-time model on the benchmark machine is:

```text
generation_seconds ~= 9.5e-5 * f^2
                   ~= 4.8e-6 * cells
```

Peak memory is less exact because it includes temporary arrays, Python/NumPy
allocator behavior, and whether NetCDF export is running. For large measured
global grids, generation peak RSS was roughly:

```text
peak_generation_rss_gb ~= (2.5e-5 to 3.6e-5) * f^2
                       ~= (1.3e-6 to 1.8e-6) * cells
```

NetCDF file size is the most predictable of the three:

```text
netcdf_size_mb ~= 0.0168 * f^2
               ~= 0.000838 * cells
```

These constants were calibrated on an Apple M1 laptop with 16 GB RAM, macOS
26.5.1, and Python 3.11.11. Generation timings exclude NetCDF export; file size
is for the standard ICON-style NetCDF output. Treat runtime and memory constants
as hardware-specific estimates, not guarantees. The asymptotic `n^2 * 4^k`
scaling is the portable part of the model.

| Grid | `f` | Cells | Generation | Peak RSS | NetCDF size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `R02B06` | 128 | 327,680 | 1.55 s | 0.58 GB | 275 MB |
| `R02B07` | 256 | 1,310,720 | 5.98 s | 2.34 GB | 1.1 GB |
| `R02B08` | 512 | 5,242,880 | 25.34 s | 6.57 GB | 4.4 GB |

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
