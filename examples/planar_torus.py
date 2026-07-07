"""Generate a planar triangular torus grid in memory."""

from grid_generator import TorusGridSpec, generate_grid

grid = generate_grid(TorusGridSpec(nx=16, ny=8, edge_length=1_000.0))
print(grid.name)
print(grid.dims)
print(grid.metadata["domain_length"])
