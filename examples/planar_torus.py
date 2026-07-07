"""Generate a planar triangular torus grid in memory."""

from grid_generator import TorusGridSpec, generate_grid
from grid_generator.visualization import write_svg

grid = generate_grid(TorusGridSpec(nx=16, ny=8, edge_length=1_000.0))
print(grid.name)
print(grid.dims)
print(grid.metadata["domain_length"])
write_svg(grid, "planar_torus.svg")
