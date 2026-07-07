"""Geometry optimization and diffusion transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._validation import finite_float_option


@dataclass(frozen=True)
class _GlobalGridOptions:
    """Internal options for compatible global spherical grid generation."""

    beta_spring: float = 0.9
    maxit: int = 2000
    fixed_boundary: bool = True
    north_pole_lon: float = 0.0
    north_pole_lat: float = 90.0
    rotation_angle_degrees: float = 0.0
    indexing_algorithm: str = "new"
    centre: int = 78
    subcentre: int = 255
    number_of_grid_used: int = 0

    def __post_init__(self) -> None:
        beta_spring = finite_float_option("beta_spring", self.beta_spring)
        if beta_spring <= 0.0:
            raise ValueError("beta_spring must be positive")
        _validate_iterations("maxit", self.maxit)
        if not isinstance(self.fixed_boundary, bool):
            raise TypeError("fixed_boundary must be a boolean")
        for name in ("north_pole_lon", "north_pole_lat", "rotation_angle_degrees"):
            object.__setattr__(self, name, finite_float_option(name, getattr(self, name)))
        if self.indexing_algorithm not in {"new", "old"}:
            raise ValueError("indexing_algorithm must be 'new' or 'old'")
        for name in ("centre", "subcentre", "number_of_grid_used"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "beta_spring", beta_spring)


@dataclass(frozen=True)
class OptimizationOptions:
    """Options for deterministic Laplacian/spring grid smoothing."""

    iterations: int = 10
    relaxation: float = 0.25
    fixed_boundary: bool = True
    target_edge_length: float | None = None

    def __post_init__(self) -> None:
        _validate_iterations("iterations", self.iterations)
        relaxation = finite_float_option("relaxation", self.relaxation)
        if not 0.0 <= relaxation <= 1.0:
            raise ValueError("relaxation must be in [0, 1]")
        if not isinstance(self.fixed_boundary, bool):
            raise TypeError("fixed_boundary must be a boolean")
        if self.target_edge_length is not None:
            target = finite_float_option("target_edge_length", self.target_edge_length)
            if target <= 0.0:
                raise ValueError("target_edge_length must be positive")
            object.__setattr__(self, "target_edge_length", target)
        object.__setattr__(self, "relaxation", relaxation)


@dataclass(frozen=True)
class _GlobalOptimizationOptions:
    """Internal options for spring-relaxed global spherical grids."""

    method: str = "none"
    iterations: int = 250

    def __post_init__(self) -> None:
        if not isinstance(self.method, str):
            raise TypeError("global optimization method must be a string")
        if self.method not in {"none", "spring"}:
            raise ValueError("global optimization method must be 'none' or 'spring'")
        _validate_iterations("global optimization iterations", self.iterations)


@dataclass(frozen=True)
class DiffusionOptions:
    """Options for explicit geometry diffusion over vertex adjacency."""

    iterations: int = 10
    diffusion_constant: float = 0.1
    dt: float = 1.0
    neighbor_weight: float = 1.0
    fixed_boundary: bool = True

    def __post_init__(self) -> None:
        _validate_iterations("iterations", self.iterations)
        diffusion_constant = finite_float_option("diffusion_constant", self.diffusion_constant)
        dt = finite_float_option("dt", self.dt)
        neighbor_weight = finite_float_option("neighbor_weight", self.neighbor_weight)
        if diffusion_constant < 0.0:
            raise ValueError("diffusion_constant must be non-negative")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if neighbor_weight <= 0.0:
            raise ValueError("neighbor_weight must be positive")
        if not isinstance(self.fixed_boundary, bool):
            raise TypeError("fixed_boundary must be a boolean")
        object.__setattr__(self, "diffusion_constant", diffusion_constant)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "neighbor_weight", neighbor_weight)


def resolve_global_optimization_options(value: Any) -> _GlobalOptimizationOptions:
    """Normalize global optimization option shorthands."""
    if value is None:
        return _GlobalOptimizationOptions()
    if isinstance(value, _GlobalOptimizationOptions):
        return value
    if isinstance(value, str):
        return _GlobalOptimizationOptions(method=value)
    if isinstance(value, dict):
        return _GlobalOptimizationOptions(**value)
    raise TypeError(
        "options must be None, a method string, a mapping, "
        "or an internal global optimization option instance"
    )


def optimize_global_grid(grid: Any, options: Any = None) -> Any:
    """Return a spring-relaxed global spherical grid with unchanged topology."""
    opts = _GlobalOptimizationOptions(method="spring") if options is None else resolve_global_optimization_options(options)
    if opts.method == "none" or opts.iterations == 0:
        return grid
    if grid.metadata.get("grid_geometry") != 1:
        raise ValueError("global optimization requires a spherical global grid")
    vertices = _spring_relaxed_vertices(grid, opts)
    return _rebuild_grid(grid, vertices)


def optimize_grid(grid: Any, options: OptimizationOptions | None = None) -> Any:
    """Return a geometry-optimized copy of `grid` with unchanged topology."""
    opts = OptimizationOptions() if options is None else options
    if not isinstance(opts, OptimizationOptions):
        raise TypeError("options must be an OptimizationOptions instance or None")
    vertices = np.asarray(grid.vertices, dtype=np.float64).copy()
    adjacency = _vertex_adjacency(grid)
    movable = _movable_vertices(grid, opts.fixed_boundary)
    for _ in range(opts.iterations):
        updated = vertices.copy()
        for vertex, neighbors in enumerate(adjacency):
            if not movable[vertex] or not neighbors:
                continue
            target = _spring_target(vertices, vertex, neighbors, opts.target_edge_length)
            updated[vertex] = vertices[vertex] + opts.relaxation * (target - vertices[vertex])
        vertices = _project_vertices(grid, updated)
    return _rebuild_grid(grid, vertices)


def diffuse_grid(grid: Any, options: DiffusionOptions | None = None) -> Any:
    """Return a geometry-diffused copy of `grid` with unchanged topology."""
    opts = DiffusionOptions() if options is None else options
    if not isinstance(opts, DiffusionOptions):
        raise TypeError("options must be a DiffusionOptions instance or None")
    vertices = np.asarray(grid.vertices, dtype=np.float64).copy()
    adjacency = _vertex_adjacency(grid)
    movable = _movable_vertices(grid, opts.fixed_boundary)
    step = opts.diffusion_constant * opts.dt
    for _ in range(opts.iterations):
        updated = vertices.copy()
        for vertex, neighbors in enumerate(adjacency):
            if not movable[vertex] or not neighbors:
                continue
            average = vertices[neighbors].mean(axis=0)
            updated[vertex] = vertices[vertex] + step * opts.neighbor_weight * (
                average - vertices[vertex]
            )
        vertices = _project_vertices(grid, updated)
    return _rebuild_grid(grid, vertices)


def _validate_iterations(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _spring_relaxed_vertices(grid: Any, opts: _GlobalOptimizationOptions) -> np.ndarray:
    vertices = np.asarray(grid.vertices, dtype=np.float64).copy()
    _normalize_force_rows_inplace(vertices)
    edges = np.asarray(grid.edges, dtype=np.int64)
    global_grid = getattr(grid.options, "global_grid", _GlobalGridOptions())
    beta_spring = global_grid.beta_spring
    maxit = opts.iterations
    if maxit == 0:
        return _project_vertices(grid, vertices)

    edge_start = edges[:, 0]
    edge_end = edges[:, 1]
    dots = np.sum(vertices[edge_start] * vertices[edge_end], axis=1)
    mean_edge_length = float(np.mean(np.arccos(np.clip(dots, -1.0, 1.0))))
    len0 = beta_spring * mean_edge_length * 1.164
    velocity = np.zeros_like(vertices)
    movable = _movable_vertices(grid, global_grid.fixed_boundary)
    all_movable = bool(np.all(movable))
    fixed_vertices = None if all_movable else vertices[~movable].copy()
    max_ekin = 0.0
    max_test = 0.0
    edge_start_vertices = np.empty((edges.shape[0], 3), dtype=np.float64)
    edge_force = np.empty((edges.shape[0], 3), dtype=np.float64)
    edge_dot = np.empty(edges.shape[0], dtype=np.float64)
    edge_scale = np.empty(edges.shape[0], dtype=np.float64)
    spring = np.empty_like(vertices)
    vertex_dot = np.empty(vertices.shape[0], dtype=np.float64)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for iteration in range(1, maxit + 1):
        if iteration <= 50:
            dt = 1.6e-2
        elif iteration <= 150:
            dt = 1.6e-2 * (1.0 + 0.04 * (iteration - 50))
        else:
            dt = 8.0e-2

        np.take(vertices, edge_start, axis=0, out=edge_start_vertices)
        np.take(vertices, edge_end, axis=0, out=edge_force)
        edge_force -= edge_start_vertices
        np.einsum("ij,ij->i", edge_start_vertices, edge_force, out=edge_dot)
        edge_dot += 1.0
        np.clip(edge_dot, -1.0, 1.0, out=edge_dot)
        np.subtract(1.0, edge_dot, out=edge_scale)
        np.maximum(edge_scale, np.finfo(np.float64).eps, out=edge_scale)
        np.sqrt(edge_scale, out=edge_scale)
        np.arccos(edge_dot, out=edge_dot)
        edge_dot -= len0
        edge_scale = np.divide(edge_dot, edge_scale, out=edge_scale)
        edge_force *= edge_scale[:, np.newaxis]

        spring.fill(0.0)
        np.add.at(spring, edge_start, edge_force)
        np.add.at(spring, edge_end, -edge_force)
        spring *= inv_sqrt2
        if not all_movable:
            spring[~movable] = 0.0

        vertices += dt * velocity
        _normalize_force_rows_inplace(vertices)
        if not all_movable:
            vertices[~movable] = fixed_vertices

        velocity *= 1.0 - dt
        velocity += dt * spring
        np.einsum("ij,ij->i", velocity, vertices, out=vertex_dot)
        velocity -= vertex_dot[:, np.newaxis] * vertices
        if not all_movable:
            velocity[~movable] = 0.0

        ekin = 0.5 * float(np.sum(velocity * velocity))
        test = float(np.sum(spring * spring))
        max_ekin = max(max_ekin, ekin)
        max_test = max(max_test, test)
        if iteration > 5 and test == max_test and max_test > 0.0:
            break
        if iteration > 5 and max_ekin > 0.0 and ekin < 0.001 * max_ekin:
            break
    return _project_vertices(grid, vertices)


def _normalize_force_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    normalized = values.copy()
    active = norms > 0.0
    normalized[active] /= norms[active, np.newaxis]
    normalized[~active] = 0.0
    return normalized


def _normalize_force_rows_inplace(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    active = norms > 0.0
    values[active] /= norms[active, np.newaxis]
    values[~active] = 0.0
    return values


def _vertex_adjacency(grid: Any) -> list[list[int]]:
    adjacency: list[set[int]] = [set() for _ in range(grid.dims["vertex"])]
    for v0, v1 in grid.edges:
        adjacency[int(v0)].add(int(v1))
        adjacency[int(v1)].add(int(v0))
    return [sorted(neighbors) for neighbors in adjacency]


def _movable_vertices(grid: Any, fixed_boundary: bool) -> np.ndarray:
    movable = np.ones(grid.dims["vertex"], dtype=bool)
    if not fixed_boundary:
        return movable
    boundary_edges = grid.edge_cells[:, 1] < 0
    if np.any(boundary_edges):
        movable[np.unique(grid.edges[boundary_edges])] = False
    return movable


def _spring_target(
    vertices: np.ndarray,
    vertex: int,
    neighbors: list[int],
    target_edge_length: float | None,
) -> np.ndarray:
    if target_edge_length is None:
        return vertices[neighbors].mean(axis=0)
    directions = vertices[neighbors] - vertices[vertex]
    lengths = np.linalg.norm(directions, axis=1)
    active = lengths > 0.0
    if not np.any(active):
        return vertices[vertex]
    desired = vertices[vertex] + directions[active] / lengths[active, np.newaxis] * target_edge_length
    return desired.mean(axis=0)


def _project_vertices(grid: Any, vertices: np.ndarray) -> np.ndarray:
    projected = vertices.copy()
    if _uses_planar_projection(grid):
        projected[:, 2] = grid.vertices[:, 2]
        if grid.metadata.get("periodic"):
            projected[:, 0] %= grid.spec.domain_length
            projected[:, 1] %= grid.spec.domain_height
        return projected

    radius = grid.options.radius
    norms = np.linalg.norm(projected, axis=1)
    if np.any(norms == 0.0):
        raise RuntimeError("optimization produced a zero-length vertex")
    return projected / norms[:, np.newaxis] * radius


def _rebuild_grid(grid: Any, vertices: np.ndarray) -> Any:
    if _uses_planar_projection(grid):
        from ._planar import rebuild_planar_grid

        return rebuild_planar_grid(grid, vertices)
    return _rebuild_spherical_grid(grid, vertices)


def _uses_planar_projection(grid: Any) -> bool:
    return (
        grid.metadata.get("grid_geometry") == 2
        or grid.metadata.get("source_grid_geometry") == 2
    )


def _rebuild_spherical_grid(grid: Any, vertices: np.ndarray) -> Any:
    from . import grid_generator as gg

    cell_center_xyz = gg._cell_centers(vertices, grid.cells, grid.options.radius)
    vertex_lon, vertex_lat = gg._lon_lat(vertices)
    lon, lat = gg._lon_lat(cell_center_xyz)
    edge_center_xyz = gg._edge_centers(vertices, grid.edges, grid.options.radius)
    edge_lon, edge_lat = gg._lon_lat(edge_center_xyz)
    geometry = _spherical_metrics(grid, vertices, cell_center_xyz, edge_center_xyz)
    metadata = dict(grid.metadata)
    metadata.update(gg._metadata(grid.spec, grid.options, geometry))
    return gg.IconGrid(
        spec=grid.spec,
        options=grid.options,
        vertices=vertices,
        cells=grid.cells.copy(),
        lon=lon,
        lat=lat,
        vertex_lon=vertex_lon,
        vertex_lat=vertex_lat,
        cell_center_xyz=cell_center_xyz,
        cell_vertex_lon=vertex_lon[grid.cells],
        cell_vertex_lat=vertex_lat[grid.cells],
        edges=grid.edges.copy(),
        cell_edges=grid.cell_edges.copy(),
        edge_cells=grid.edge_cells.copy(),
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity={name: value.copy() for name, value in grid.icon_connectivity.items()},
        connectivity={name: value.copy() for name, value in grid.connectivity.items()},
        neighbor_tables={name: value.copy() for name, value in grid.neighbor_tables.items()},
        geometry=geometry,
        refinement={name: value.copy() for name, value in grid.refinement.items()},
        metadata=metadata,
    )


def _spherical_metrics(
    grid: Any,
    vertices: np.ndarray,
    cell_center_xyz: np.ndarray,
    edge_center_xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg

    if np.all(grid.edge_cells >= 0):
        return gg._geometry_fields(
            vertices,
            grid.cells,
            cell_center_xyz,
            grid.edges,
            grid.edge_cells,
            edge_center_xyz,
            grid.icon_connectivity,
            grid.options.sphere_radius,
        )
    cell_areas = gg._cell_areas(vertices, grid.cells, grid.options.sphere_radius)
    edge_lengths = gg._edge_lengths(vertices, grid.edges, grid.options.sphere_radius)
    edge_cell_distance = _open_edge_cell_distances(
        cell_center_xyz,
        grid.edge_cells,
        edge_center_xyz,
        grid.options.sphere_radius,
    )
    dual_edge_lengths = edge_cell_distance.sum(axis=1)
    boundary = grid.edge_cells[:, 1] < 0
    dual_edge_lengths[boundary] = 2.0 * edge_cell_distance[boundary, 0]
    edge_system_orientation = np.ones(grid.edges.shape[0], dtype=np.int32)
    normals = gg._edge_normal_fields(vertices, grid.edges, edge_center_xyz, edge_system_orientation)
    return {
        "cell_area": cell_areas,
        "dual_area": gg._dual_areas(vertices.shape[0], grid.cells, cell_areas),
        "edge_length": edge_lengths,
        "dual_edge_length": dual_edge_lengths,
        "edge_cell_distance": edge_cell_distance,
        "edge_vert_distance": np.column_stack((edge_lengths * 0.5, edge_lengths * 0.5)),
        "orientation_of_normal": grid.icon_connectivity["orientation_of_normal"],
        "edge_system_orientation": edge_system_orientation,
        "edge_orientation": grid.icon_connectivity["edge_orientation"],
        "edgequad_area": 0.5 * edge_lengths * dual_edge_lengths,
        **normals,
    }


def _open_edge_cell_distances(
    cell_center_xyz: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    from . import grid_generator as gg

    edge_centers = gg._normalize_rows(edge_center_xyz)
    cell_centers = gg._normalize_rows(cell_center_xyz)
    distances = np.zeros((edge_cells.shape[0], 2), dtype=np.float64)
    for edge_index, adjacent in enumerate(edge_cells):
        for side in range(2):
            cell = int(adjacent[side])
            if cell < 0:
                distances[edge_index, side] = distances[edge_index, 0]
                continue
            dot = float(np.dot(cell_centers[cell], edge_centers[edge_index]))
            distances[edge_index, side] = np.arccos(np.clip(dot, -1.0, 1.0)) * sphere_radius
    return distances
