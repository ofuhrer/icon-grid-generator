"""Generate a global ICON-style grid and write it as NetCDF."""

from grid_generator import generate_grid


grid = generate_grid("R2B4")
print(grid.name)
print(grid.dims)
grid.to_netcdf("icon_grid_R02B04.nc")
