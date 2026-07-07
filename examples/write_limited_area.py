"""Generate a limited-area ICON-style grid and write it as NetCDF."""

from grid_generator import LimitedAreaGridSpec, Region, generate_grid

spec = LimitedAreaGridSpec(
    parent="R2B2",
    region=Region.lonlat_box(lon_min=-20.0, lon_max=20.0, lat_min=35.0, lat_max=60.0),
    boundary_depth=2,
)

grid = generate_grid(spec, max_cells=None)
print(grid.name)
print(grid.dims)
grid.to_netcdf("icon_grid_limited_area.nc")
