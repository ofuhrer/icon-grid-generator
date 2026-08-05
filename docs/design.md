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
- Planar triangular variants share one builder pipeline. The spec object
  carries variant flags such as `periodic` and `periodic_x`; `_planar.py`
  dispatches geometry, topology, and metric behavior from those flags instead
  of adding separate public generation entry points.
- Doubly periodic triangular grids use the two coupled lattice vectors of the
  skew fundamental domain. Crossing the periodic y boundary therefore applies
  both a y wrap and the corresponding x shift. Geometry transforms use the same
  minimum-image convention when averaging neighboring vertices.
- Planar dual-edge lengths are computed from generated cell centers through
  edge centers rather than assumed equilateral formulas. Planar vertex dual
  areas use one third of each incident triangle, so open-grid dual areas
  partition the total primal area without assigning full interior control
  volumes to boundary vertices. Planar `edgequad_area` uses the cross product
  of primal and dual edge vectors, so sheared grids include their actual
  intersection angle rather than assuming orthogonality.
- Limited-area and cut grids are compacted views of an `IconGrid` selected by
  region predicates plus optional boundary expansion. They deliberately reuse
  the open-mesh topology, metrics, and refinement reconstruction path so
  regional extraction does not fork the grid contract.
- Geometry optimization and diffusion are post-generation transforms that
  preserve topology and rebuild geometry-derived fields. Cut grids retain the
  source spec needed for periodic coordinate reconstruction. A nontrivial
  public transform receives a deterministic UUID derived from the input grid
  UUID and canonical transform options while preserving `uuidOfParHGrid`;
  `geometry_transform_source_uuid` records that immediate input and zero-step
  transforms return the input grid unchanged. Global spring
  relaxation shares the same module but remains part of global generation when
  `optimize_global=True`.
- Spherical `dual_area` follows the ICON grid-file contract: each vertex area
  is the sum of `0.25 * edge_length * dual_edge_length` over incident edges.
  It is an edge-quadrilateral metric field and is not forced to sum exactly to
  the spherical cell-area total.
- Optional Numba acceleration is an implementation detail selected through
  `IconGridOptions.accelerator`; NumPy remains the required baseline.
- UUIDs use deterministic UUIDv5 payloads derived from canonical specs and
  options. Limited-area and cut-grid payloads include the source parent UUID,
  and `uuidOfParHGrid` records that source. Any payload change is a
  compatibility change.
- NetCDF export is an internal module boundary. Public users should call
  `IconGrid.to_netcdf(path)`.
- For compatibility with established ICON grid files, spherical NetCDF
  `edgequad_area` values are normalized by `sphere_radius**2`. The exported
  variable is therefore dimensionless (`units = "1"`) and carries a
  `normalization` attribute. In-memory `IconGrid.geometry` retains physical
  square-metre values. This compatibility convention is not applied to planar
  grids.
- Xarray public connectivity is zero-based with `-1` for missing neighbors;
  parent-provenance arrays remain one-based with `0` for no parent. Variables
  carry explicit `start_index` and `missing_value` attributes.
- Pipeline stage results use frozen dataclasses to keep builder boundaries
  explicit. Arrays remain mutable NumPy buffers during construction; callers
  should treat completed `IconGrid` objects as immutable values.
- `grid_generator.py` owns public specs, validation-facing helpers, metadata,
  UUID payloads, and the `generate_grid()` facade. Large implementation
  concerns should stay in focused private modules rather than growing the
  facade again.
- Performance checks live behind `make perf-check` and are intentionally
  separate from default CI-style checks because runtime varies with local load.

## Optimization Algorithms

### Staged Global Spring System

Default global generation operates recursively across bisection levels. A
parent stage is generated and relaxed before its child is refined, and the child
is then relaxed as another stage. The spring kernel normalizes coordinates to
the unit sphere, computes the initial mean angular edge length, and uses
`1.164 * beta_spring` times that mean as its rest angle. Incident edge forces
are accumulated per vertex and integrated with a damped velocity. Every
position and velocity update is projected into the sphere tangent geometry.

The time step is `0.016` for the first 50 steps, increases through step 150, and
is `0.08` thereafter. Iteration is bounded by the stage cap and may terminate
early when the force statistic reaches its observed maximum or kinetic energy
falls below `0.001` of its observed maximum. For stages whose parent has fewer
than 100,000 cells, the internal cap is ten times the requested
`spring_iterations`; the public metadata records the requested setting rather
than the realized number of integration steps.

A `B0` specification returns the completed root geometry directly because no
bisection stage exists; consequently the staged relaxation path is not entered.

This path aims to reduce edge-length and cell-area variation while retaining
the exact refined topology. It is a compatibility-oriented heuristic, not an
optimization with a reported scalar objective or formal convergence proof.

### General Smoothing and Diffusion

`optimize_grid()` and `diffuse_grid()` build undirected vertex adjacency from
the immutable edge table. Each iteration is Jacobi-style: all targets are
computed from the previous vertex array and applied simultaneously.

Without `target_edge_length`, `optimize_grid()` moves a fraction `relaxation`
toward the mean neighbor displacement. With a target length, each active edge
direction is normalized, scaled to the requested length, and translated back to
the current vertex before the incident proposals are averaged. `diffuse_grid()`
uses the same mean displacement multiplied by
`diffusion_constant * dt * neighbor_weight`.

Periodic displacements use the same coupled lattice minimum-image operation as
metric generation. Spherical vertices are renormalized to `radius`; planar z
coordinates are restored; periodic planar coordinates are wrapped to the
fundamental domain. Open-boundary vertices are excluded from updates when
`fixed_boundary=True`.

After movement, the grid rebuilds centers, projected coordinates, metric fields,
normal vectors, and summary metadata. Topology and refinement arrays are
unchanged. Public transforms derive a new UUID from the source UUID, operation,
and canonical option payload. No inversion detector or line search is part of
the update, so callers remain responsible for checking scientific quality.

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
- Longitude/latitude rectangles and polygons are center-selection predicates
  evaluated in a wrapped equirectangular lon/lat plane. Circles use great-circle
  angular distance. Use small regional rectangles/polygons away from the poles,
  or provide an explicit scientific comparison for polar and very large areas.
- Planar `lon`/`lat` fields are normalized compatibility and visualization
  coordinates, not a geographic CRS. Scientific planar calculations should use
  Cartesian coordinates and metric arrays.
- `smoothing_depth` on a cut is an ICON control-field value written uniformly
  to `smooth_c_ctrl`; it does not invoke the geometry smoothing algorithms.
- `check_grid()` is a structural check. It does not independently rederive all
  metric fields, test cell inversion, or certify a grid for a numerical model.
- `write_svg()` is an equirectangular-style diagnostic preview. It omits
  periodic seam segments and can subsample edges, so it must not be used to
  assess metric distances, areas, or complete seam topology.

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

Raw topology/metric generation time, peak memory, and NetCDF file size are
therefore all expected to scale approximately as `O(n^2 * 4^k)` for
sufficiently large global grids. Equivalently, each additional bisection level
roughly multiplies work and output size by four.

The measured single-process generation-time model below is calibrated for raw
global generation with `optimize_global=False`. Default optimized global
generation has the same asymptotic grid-size scaling, but its runtime constant
depends on the spring-relaxation settings and is not represented by this model.

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

Private helper tests may exercise defensive branches for coverage when the
branch protects a public contract. These tests are regression guards, not
scientific validation or additional public API.
