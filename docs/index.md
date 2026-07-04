# ICON Grid Generator

ICON Grid Generator is a pure Python package for creating ICON-style triangular
grids without depending on ICON model runtimes or stencil frameworks.

![Global ICON grid resolutions](assets/global-icon-grid-series.png)

## What It Provides

- Global spherical ICON `R<n>B<k>` grids, including compact string parsing such
  as `R2B3` and canonical zero-padded names such as `R02B03`.
- Planar torus and open planar triangular grids for local experiments.
- Limited-area grids extracted from generated global parent grids.
- ICON-style NetCDF export when the optional `netCDF4` dependency is installed.
- In-memory geometry, topology, connectivity, metric, and refinement arrays for
  plotting, diagnostics, and downstream conversion.

## Quick Example

```python
from grid_generator import generate_grid

grid = generate_grid("R2B3")
print(grid.dims)
print(grid.metadata["mean_edge_length"])
```

## Project Links

- [Examples](examples.md)
- [API overview](api.md)
- [Design notes and limitations](design.md)
- [Changelog](https://github.com/ofuhrer/icon-grid-generator/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/ofuhrer/icon-grid-generator/blob/main/CITATION.cff)
