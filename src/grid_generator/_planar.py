"""Additional planar triangular grid variants and metrics."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from ._limited_area import (
    _open_icon_connectivity,
    _open_neighbor_tables,
    _open_public_connectivity,
)
from ._types import GeometryData, MetricsData, RefinementData, TopologyData
from ._torus import _scale_to_degrees


class PlanarTriangularGeometry:
    """Build triangular planar grids from spec-defined lattice coordinates."""

    def build(self, spec: Any, options: Any) -> GeometryData:
        if getattr(spec, "periodic", False):
            return _periodic_geometry(spec, options)
        return _open_geometry(spec, options)


class PlanarTriangularTopologyBuilder:
    """Build topology for periodic or open planar triangular grids."""

    def build(self, spec: Any, geometry: GeometryData) -> TopologyData:
        if getattr(spec, "periodic", False):
            return _periodic_topology(spec, geometry)
        return _open_topology(spec, geometry)


class PlanarTriangularMetricsBuilder:
    """Compute planar metrics from actual vertex coordinates."""

    def build(self, spec: Any, geometry: GeometryData, topology: TopologyData) -> MetricsData:
        return _planar_metrics(spec, geometry, topology)


class PlanarRefinementBuilder:
    """Return default refinement fields for standalone planar grids."""

    def build(self, geometry: GeometryData, topology: TopologyData) -> RefinementData:
        from . import grid_generator as gg

        n_cells = geometry.cells.shape[0]
        n_edges = topology.edges.shape[0]
        n_vertices = geometry.vertices.shape[0]
        return RefinementData(
            fields={
                "refin_c_ctrl": np.zeros(n_cells, dtype=np.int32),
                "refin_e_ctrl": np.zeros(n_edges, dtype=np.int32),
                "refin_v_ctrl": np.zeros(n_vertices, dtype=np.int32),
                "start_idx_c": gg._start_index_fixed("cell_grf", n_cells),
                "end_idx_c": gg._end_index_fixed("cell_grf", n_cells),
                "start_idx_e": gg._start_index_fixed("edge_grf", n_edges),
                "end_idx_e": gg._end_index_fixed("edge_grf", n_edges),
                "start_idx_v": gg._start_index_fixed("vert_grf", n_vertices),
                "end_idx_v": gg._end_index_fixed("vert_grf", n_vertices),
                "parent_cell_index": np.zeros(n_cells, dtype=np.int32),
                "parent_cell_type": np.zeros(n_cells, dtype=np.int32),
                "edge_parent_type": np.zeros(n_edges, dtype=np.int32),
                "parent_edge_index": np.zeros(n_edges, dtype=np.int32),
                "parent_vertex_index": np.zeros(n_vertices, dtype=np.int32),
            }
        )


def rebuild_planar_grid(grid: Any, vertices: np.ndarray) -> Any:
    """Return a grid with unchanged topology and recomputed planar geometry."""
    from . import grid_generator as gg

    vertices = np.asarray(vertices, dtype=np.float64)
    geometry = _geometry_from_existing(grid, vertices)
    topology = _topology_from_existing(grid, geometry)
    metrics = _planar_metrics(grid.spec, geometry, topology)
    metadata = dict(grid.metadata)
    metadata.update(gg._metadata(grid.spec, grid.options, metrics.fields))
    return gg.IconGrid(
        spec=grid.spec,
        options=grid.options,
        vertices=geometry.vertices,
        cells=geometry.cells,
        lon=geometry.lon,
        lat=geometry.lat,
        vertex_lon=geometry.vertex_lon,
        vertex_lat=geometry.vertex_lat,
        cell_center_xyz=geometry.cell_center_xyz,
        cell_vertex_lon=geometry.cell_vertex_lon,
        cell_vertex_lat=geometry.cell_vertex_lat,
        edges=topology.edges,
        cell_edges=topology.cell_edges,
        edge_cells=topology.edge_cells,
        edge_center_xyz=topology.edge_center_xyz,
        edge_lon=topology.edge_lon,
        edge_lat=topology.edge_lat,
        icon_connectivity=topology.icon_connectivity,
        connectivity=topology.connectivity,
        neighbor_tables=topology.neighbor_tables,
        geometry=metrics.fields,
        refinement={name: value.copy() for name, value in grid.refinement.items()},
        metadata=metadata,
    )


def _periodic_geometry(spec: Any, options: Any) -> GeometryData:
    nx = spec.nx
    ny = spec.ny
    vertices = np.zeros((nx * ny, 3), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            vertices[_periodic_vertex_id(i, j, nx, ny)] = (*_periodic_xy(spec, i, j), options.radius)

    cells: list[tuple[int, int, int]] = []
    centers: list[np.ndarray] = []
    for j in range(ny):
        for i in range(nx):
            up = (
                _periodic_vertex_id(i, j, nx, ny),
                _periodic_vertex_id(i + 1, j, nx, ny),
                _periodic_vertex_id(i, j + 1, nx, ny),
            )
            down = (
                _periodic_vertex_id(i, j, nx, ny),
                _periodic_vertex_id(i + 1, j - 1, nx, ny),
                _periodic_vertex_id(i + 1, j, nx, ny),
            )
            for cell in (up, down):
                cells.append(cell)
                centers.append(_periodic_triangle_center(vertices, cell, spec))

    return _geometry_data(
        spec,
        vertices,
        np.asarray(cells, dtype=np.int32),
        np.asarray(centers, dtype=np.float64),
        periodic=True,
    )


def _open_geometry(spec: Any, options: Any) -> GeometryData:
    if _has_periodic_x(spec):
        return _channel_geometry(spec, options)

    vertices = np.zeros(((spec.nx + 1) * (spec.ny + 1), 3), dtype=np.float64)
    for j in range(spec.ny + 1):
        for i in range(spec.nx + 1):
            vertices[_open_vertex_id(i, j, spec.nx)] = (*_open_xy(spec, i, j), options.radius)

    cells: list[tuple[int, int, int]] = []
    for j in range(spec.ny):
        for i in range(spec.nx):
            v00 = _open_vertex_id(i, j, spec.nx)
            v10 = _open_vertex_id(i + 1, j, spec.nx)
            v01 = _open_vertex_id(i, j + 1, spec.nx)
            v11 = _open_vertex_id(i + 1, j + 1, spec.nx)
            cells.extend(((v00, v10, v01), (v10, v11, v01)))

    cell_array = np.asarray(cells, dtype=np.int32)
    centers = vertices[cell_array].mean(axis=1)
    return _geometry_data(spec, vertices, cell_array, centers, periodic=False)


def _channel_geometry(spec: Any, options: Any) -> GeometryData:
    vertices = np.zeros((spec.nx * (spec.ny + 1), 3), dtype=np.float64)
    for j in range(spec.ny + 1):
        for i in range(spec.nx):
            vertices[_channel_vertex_id(i, j, spec.nx)] = (*_open_xy(spec, i, j), options.radius)

    cells: list[tuple[int, int, int]] = []
    centers: list[np.ndarray] = []
    for j in range(spec.ny):
        for i in range(spec.nx):
            v00 = _channel_vertex_id(i, j, spec.nx)
            v10 = _channel_vertex_id(i + 1, j, spec.nx)
            v01 = _channel_vertex_id(i, j + 1, spec.nx)
            v11 = _channel_vertex_id(i + 1, j + 1, spec.nx)
            for cell in ((v00, v10, v01), (v10, v11, v01)):
                cells.append(cell)
                centers.append(_channel_triangle_center(vertices, cell, spec))

    return _geometry_data(
        spec,
        vertices,
        np.asarray(cells, dtype=np.int32),
        np.asarray(centers, dtype=np.float64),
        periodic=False,
    )


def _geometry_data(
    spec: Any,
    vertices: np.ndarray,
    cells: np.ndarray,
    centers: np.ndarray,
    *,
    periodic: bool,
) -> GeometryData:
    if periodic:
        lon = _scale_to_degrees(centers[:, 0], spec.domain_length, -180.0, 180.0)
        lat = _scale_to_degrees(centers[:, 1], spec.domain_height, -90.0, 90.0)
        vertex_lon = _scale_to_degrees(vertices[:, 0], spec.domain_length, -180.0, 180.0)
        vertex_lat = _scale_to_degrees(vertices[:, 1], spec.domain_height, -90.0, 90.0)
    else:
        x_min, x_max = float(vertices[:, 0].min()), float(vertices[:, 0].max())
        y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
        lon = _linear_scale(centers[:, 0], x_min, x_max, -180.0, 180.0)
        lat = _linear_scale(centers[:, 1], y_min, y_max, -90.0, 90.0)
        vertex_lon = _linear_scale(vertices[:, 0], x_min, x_max, -180.0, 180.0)
        vertex_lat = _linear_scale(vertices[:, 1], y_min, y_max, -90.0, 90.0)
    return GeometryData(
        vertices=vertices,
        cells=cells,
        lon=lon,
        lat=lat,
        vertex_lon=vertex_lon,
        vertex_lat=vertex_lat,
        cell_center_xyz=centers,
        cell_vertex_lon=vertex_lon[cells],
        cell_vertex_lat=vertex_lat[cells],
        source_cell_index=np.arange(cells.shape[0], dtype=np.int32),
        source_vertex_index=np.arange(vertices.shape[0], dtype=np.int32),
    )


def _periodic_topology(spec: Any, geometry: GeometryData) -> TopologyData:
    from . import grid_generator as gg

    edges, cell_edges, edge_cells = gg._build_edges(geometry.cells)
    edge_center_xyz = _periodic_edge_centers(geometry.vertices, edges, spec)
    edge_lon = _scale_to_degrees(edge_center_xyz[:, 0], spec.domain_length, -180.0, 180.0)
    edge_lat = _scale_to_degrees(edge_center_xyz[:, 1], spec.domain_height, -90.0, 90.0)
    icon = gg._icon_connectivity(
        geometry.vertices,
        geometry.cells,
        geometry.cell_center_xyz,
        edges,
        cell_edges,
        edge_cells,
    )
    return TopologyData(
        edges=edges,
        cell_edges=cell_edges,
        edge_cells=edge_cells,
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon,
        connectivity=gg._public_connectivity(geometry.cells, edges, edge_cells, icon),
        neighbor_tables=gg._neighbor_tables(geometry.cells, edges, edge_cells, icon),
        source_edge_index=np.arange(edges.shape[0], dtype=np.int32),
    )


def _open_topology(spec: Any, geometry: GeometryData) -> TopologyData:
    edges, cell_edges, edge_cells = _build_edges_with_boundary(geometry.cells)
    edge_center_xyz = (
        _channel_edge_centers(geometry.vertices, edges, spec)
        if _has_periodic_x(spec)
        else geometry.vertices[edges].mean(axis=1)
    )
    edge_lon = _edge_lon_from_vertices(geometry, edges)
    edge_lat = _edge_lat_from_vertices(geometry, edges)
    icon = _open_icon_connectivity(
        geometry.vertices,
        geometry.cells,
        geometry.cell_center_xyz,
        edges,
        cell_edges,
        edge_cells,
    )
    return TopologyData(
        edges=edges,
        cell_edges=cell_edges,
        edge_cells=edge_cells,
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon,
        connectivity=_open_public_connectivity(geometry.cells, edges, edge_cells, icon),
        neighbor_tables=_open_neighbor_tables(geometry.cells, edges, edge_cells, icon),
        source_edge_index=np.arange(edges.shape[0], dtype=np.int32),
    )


def _geometry_from_existing(grid: Any, vertices: np.ndarray) -> GeometryData:
    centers = _cell_centers_from_existing(grid, vertices)
    return _geometry_data(
        grid.spec,
        vertices,
        grid.cells.copy(),
        centers,
        periodic=bool(grid.metadata.get("periodic")),
    )


def _topology_from_existing(grid: Any, geometry: GeometryData) -> TopologyData:
    if bool(grid.metadata.get("periodic")):
        edge_center_xyz = _periodic_edge_centers(geometry.vertices, grid.edges, grid.spec)
        edge_lon = _scale_to_degrees(edge_center_xyz[:, 0], grid.spec.domain_length, -180.0, 180.0)
        edge_lat = _scale_to_degrees(edge_center_xyz[:, 1], grid.spec.domain_height, -90.0, 90.0)
    else:
        edge_center_xyz = (
            _channel_edge_centers(geometry.vertices, grid.edges, grid.spec)
            if _has_periodic_x(grid.spec)
            else geometry.vertices[grid.edges].mean(axis=1)
        )
        edge_lon = _edge_lon_from_vertices(geometry, grid.edges)
        edge_lat = _edge_lat_from_vertices(geometry, grid.edges)
    return TopologyData(
        edges=grid.edges.copy(),
        cell_edges=grid.cell_edges.copy(),
        edge_cells=grid.edge_cells.copy(),
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity={name: value.copy() for name, value in grid.icon_connectivity.items()},
        connectivity={name: value.copy() for name, value in grid.connectivity.items()},
        neighbor_tables={name: value.copy() for name, value in grid.neighbor_tables.items()},
        source_edge_index=np.arange(grid.edges.shape[0], dtype=np.int32),
    )


def _planar_metrics(spec: Any, geometry: GeometryData, topology: TopologyData) -> MetricsData:
    vectors = _edge_vectors(spec, geometry.vertices, topology.edges)
    edge_lengths = np.linalg.norm(vectors, axis=1)
    tangent = vectors / edge_lengths[:, np.newaxis]
    primal_normal = np.column_stack((-tangent[:, 1], tangent[:, 0], np.zeros(tangent.shape[0])))
    dual_normal = tangent
    triangles = _cell_triangles(spec, geometry.vertices, geometry.cells)
    cell_area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if _uses_regular_planar_metrics(spec):
        dual_edge_length = _regular_dual_edge_lengths(spec, vectors)
        edge_cell_distance = np.column_stack((0.5 * dual_edge_length, 0.5 * dual_edge_length))
        dual_area = np.full(
            geometry.vertices.shape[0],
            2.0 * float(np.mean(cell_area)),
            dtype=np.float64,
        )
    else:
        edge_cell_distance = _edge_cell_distances(spec, geometry, topology)
        dual_edge_length = edge_cell_distance.sum(axis=1)
        boundary = topology.edge_cells[:, 1] < 0
        dual_edge_length[boundary] = 2.0 * edge_cell_distance[boundary, 0]
        dual_area = _dual_areas(geometry.vertices.shape[0], geometry.cells, cell_area)
    fields = {
        "cell_area": cell_area,
        "dual_area": dual_area,
        "edge_length": edge_lengths,
        "dual_edge_length": dual_edge_length,
        "edge_cell_distance": edge_cell_distance,
        "edge_vert_distance": np.column_stack((edge_lengths * 0.5, edge_lengths * 0.5)),
        "orientation_of_normal": topology.icon_connectivity["orientation_of_normal"],
        "edge_system_orientation": np.ones(topology.edges.shape[0], dtype=np.int32),
        "edge_orientation": topology.icon_connectivity["edge_orientation"],
        "edgequad_area": np.zeros(topology.edges.shape[0], dtype=np.float64),
        "edge_primal_normal_cartesian": primal_normal,
        "edge_dual_normal_cartesian": dual_normal,
        "zonal_normal_primal_edge": primal_normal[:, 0],
        "meridional_normal_primal_edge": primal_normal[:, 1],
        "zonal_normal_dual_edge": dual_normal[:, 0],
        "meridional_normal_dual_edge": dual_normal[:, 1],
    }
    return MetricsData(fields=fields)


def _build_edges_with_boundary(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_ids: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    edge_cells: list[list[int]] = []
    cell_edges = np.empty((cells.shape[0], 3), dtype=np.int32)
    for cell_index, (v0, v1, v2) in enumerate(cells):
        for local_index, pair in enumerate(((v0, v1), (v1, v2), (v2, v0))):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            edge_id = edge_ids.get(key)
            if edge_id is None:
                edge_id = len(edges)
                edge_ids[key] = edge_id
                edges.append(key)
                edge_cells.append([cell_index])
            else:
                edge_cells[edge_id].append(cell_index)
            cell_edges[cell_index, local_index] = edge_id
    edge_cell_array = np.full((len(edges), 2), -1, dtype=np.int32)
    for edge_index, adjacent in enumerate(edge_cells):
        edge_cell_array[edge_index, : len(adjacent)] = adjacent
    return np.asarray(edges, dtype=np.int32), cell_edges, edge_cell_array


def _periodic_vertex_id(i: int, j: int, nx: int, ny: int) -> int:
    return (j % ny) * nx + (i % nx)


def _channel_vertex_id(i: int, j: int, nx: int) -> int:
    return j * nx + (i % nx)


def _open_vertex_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i


def _periodic_xy(spec: Any, i: int, j: int) -> tuple[float, float]:
    height = sqrt(3.0) * 0.5 * spec.edge_length * spec.stretch_y
    return (i + 0.5 * j) * spec.edge_length * spec.stretch_x, j * height


def _open_xy(spec: Any, i: int, j: int) -> tuple[float, float]:
    if hasattr(spec, "dx"):
        return _ragged_xy(spec, i, j)
    height = sqrt(3.0) * 0.5 * spec.edge_length
    shear = getattr(spec, "shear", 0.0)
    x = (i + 0.5 * j) * spec.edge_length + shear * j * height
    y = j * height
    return x, y


def _ragged_xy(spec: Any, i: int, j: int) -> tuple[float, float]:
    x = i * spec.dx
    if 0 < i < spec.nx:
        x += spec.raggedness * spec.dx * np.sin((i * 1.7) + (j * 0.9))
    y = j * spec.dy
    if 0 < j < spec.ny:
        y += spec.raggedness * spec.dy * np.cos((i * 0.8) - (j * 1.3))
    return float(x), float(y)


def _periodic_triangle_center(vertices: np.ndarray, cell: tuple[int, int, int], spec: Any) -> np.ndarray:
    base = vertices[cell[0]].copy()
    points = [base]
    for vertex in cell[1:]:
        points.append(base + _periodic_lattice_delta(vertices[vertex] - base, spec))
    center = np.mean(points, axis=0)
    center[0] %= spec.domain_length
    center[1] %= spec.domain_height
    return center


def _channel_triangle_center(vertices: np.ndarray, cell: tuple[int, int, int], spec: Any) -> np.ndarray:
    base = vertices[cell[0]].copy()
    points = [base]
    for vertex in cell[1:]:
        point = vertices[vertex].copy()
        point[0] = base[0] + _horizontal_periodic_delta(point[0] - base[0], spec)
        points.append(point)
    center = np.mean(points, axis=0)
    center[0] %= spec.nx * spec.edge_length
    return center


def _periodic_edge_centers(vertices: np.ndarray, edges: np.ndarray, spec: Any) -> np.ndarray:
    vectors = _edge_vectors(spec, vertices, edges)
    centers = vertices[edges[:, 0]] + 0.5 * vectors
    centers[:, 0] %= spec.domain_length
    centers[:, 1] %= spec.domain_height
    centers[:, 2] = vertices[:, 2].mean()
    return centers


def _channel_edge_centers(vertices: np.ndarray, edges: np.ndarray, spec: Any) -> np.ndarray:
    vectors = _edge_vectors(spec, vertices, edges)
    centers = vertices[edges[:, 0]] + 0.5 * vectors
    centers[:, 0] %= spec.nx * spec.edge_length
    centers[:, 2] = vertices[:, 2].mean()
    return centers


def _cell_centers_from_existing(grid: Any, vertices: np.ndarray) -> np.ndarray:
    if bool(grid.metadata.get("periodic")):
        return np.asarray(
            [_periodic_triangle_center(vertices, tuple(cell), grid.spec) for cell in grid.cells],
            dtype=np.float64,
        )
    if _has_periodic_x(grid.spec):
        return np.asarray(
            [_channel_triangle_center(vertices, tuple(cell), grid.spec) for cell in grid.cells],
            dtype=np.float64,
        )
    return vertices[grid.cells].mean(axis=1)


def _edge_vectors(spec: Any, vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    if getattr(spec, "periodic", False):
        vectors = _periodic_lattice_delta(vectors, spec)
    elif _has_periodic_x(spec):
        vectors[:, 0] = _horizontal_periodic_delta(vectors[:, 0], spec)
    return vectors


def _cell_triangles(spec: Any, vertices: np.ndarray, cells: np.ndarray) -> np.ndarray:
    triangles = vertices[cells].copy()
    if not getattr(spec, "periodic", False) and not _has_periodic_x(spec):
        return triangles
    for cell_index, cell in enumerate(cells):
        base = vertices[cell[0]]
        triangles[cell_index, 0] = base
        for local_index in (1, 2):
            delta = vertices[cell[local_index]] - base
            if getattr(spec, "periodic", False):
                delta = _periodic_lattice_delta(delta, spec)
            else:
                delta[0] = _horizontal_periodic_delta(delta[0], spec)
            triangles[cell_index, local_index] = base + delta
    return triangles


def _periodic_lattice_delta(delta: np.ndarray, spec: Any) -> np.ndarray:
    result = np.asarray(delta, dtype=np.float64).copy()
    flat = result.reshape((-1, result.shape[-1]))
    for row in flat:
        best = row.copy()
        best_norm = np.linalg.norm(best[:2])
        y_center = int(np.rint(row[1] / spec.domain_height))
        for y_wrap in range(y_center - 1, y_center + 2):
            for y_shift in (0.0, _periodic_y_shift(spec)):
                for row_shift in (-1, 0, 1):
                    shifted = row.copy()
                    shifted[1] -= y_wrap * spec.domain_height
                    shifted[0] -= y_wrap * y_shift
                    shifted[0] -= row_shift * _periodic_y_shift(spec)
                    x_center = int(np.rint(shifted[0] / spec.domain_length))
                    for x_wrap in range(x_center - 1, x_center + 2):
                        candidate = shifted.copy()
                        candidate[0] -= x_wrap * spec.domain_length
                        candidate_norm = np.linalg.norm(candidate[:2])
                        if candidate_norm < best_norm:
                            best = candidate
                            best_norm = candidate_norm
        row[:] = best
    return result


def _periodic_y_shift(spec: Any) -> float:
    return 0.5 * spec.ny * spec.edge_length * getattr(spec, "stretch_x", 1.0)


def _horizontal_periodic_delta(delta: np.ndarray | float, spec: Any) -> np.ndarray | float:
    period = spec.nx * spec.edge_length
    return np.asarray(delta) - np.round(np.asarray(delta) / period) * period


def _has_periodic_x(spec: Any) -> bool:
    return bool(getattr(spec, "periodic_x", False))


def _uses_regular_planar_metrics(spec: Any) -> bool:
    return (
        getattr(spec, "periodic", False)
        or _has_periodic_x(spec)
        or type(spec).__name__ == "ParallelogramGridSpec"
    ) and not hasattr(spec, "dx")


def _regular_dual_edge_lengths(spec: Any, edge_vectors: np.ndarray) -> np.ndarray:
    base_length = spec.edge_length * getattr(spec, "stretch_x", 1.0)
    y_step = spec.edge_length * sqrt(3.0) * 0.5 * getattr(spec, "stretch_y", 1.0)
    if getattr(spec, "periodic", False) and not np.isclose(getattr(spec, "stretch_y", 1.0), 1.0):
        phi = np.arctan2(2.0 * y_step, base_length)
        horizontal = base_length / sqrt(3.0)
        diagonal = 0.5 * base_length / np.sin(phi)
        return np.where(np.isclose(edge_vectors[:, 1], 0.0), horizontal, diagonal).astype(np.float64)
    return np.full(edge_vectors.shape[0], base_length / sqrt(3.0), dtype=np.float64)


def _edge_cell_distances(spec: Any, geometry: GeometryData, topology: TopologyData) -> np.ndarray:
    distances = np.zeros((topology.edges.shape[0], 2), dtype=np.float64)
    for edge_index, adjacent in enumerate(topology.edge_cells):
        for side in range(2):
            cell = int(adjacent[side])
            if cell < 0:
                distances[edge_index, side] = distances[edge_index, 0]
                continue
            delta = geometry.cell_center_xyz[cell] - topology.edge_center_xyz[edge_index]
            if getattr(spec, "periodic", False):
                delta = _periodic_lattice_delta(delta, spec)
            elif _has_periodic_x(spec):
                delta[0] = _horizontal_periodic_delta(delta[0], spec)
            distances[edge_index, side] = float(np.linalg.norm(delta))
    return distances


def _dual_areas(n_vertices: int, cells: np.ndarray, cell_areas: np.ndarray) -> np.ndarray:
    dual = np.zeros(n_vertices, dtype=np.float64)
    for cell_index, cell in enumerate(cells):
        dual[cell] += cell_areas[cell_index] / 3.0
    return dual


def _linear_scale(
    values: np.ndarray,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> np.ndarray:
    if np.isclose(source_min, source_max):
        return np.full_like(values, 0.5 * (target_min + target_max), dtype=np.float64)
    return target_min + (values - source_min) / (source_max - source_min) * (target_max - target_min)


def _edge_lon_from_vertices(geometry: GeometryData, edges: np.ndarray) -> np.ndarray:
    return geometry.vertex_lon[edges].mean(axis=1)


def _edge_lat_from_vertices(geometry: GeometryData, edges: np.ndarray) -> np.ndarray:
    return geometry.vertex_lat[edges].mean(axis=1)
