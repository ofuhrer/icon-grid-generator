"""Limited-area grids extracted from generated global grids."""

from __future__ import annotations

from collections import deque
from math import cos, radians
from typing import Any

import numpy as np

from ._types import GeometryData, MetricsData, RefinementData, TopologyData


class LimitedAreaExtractor:
    """Extract a compact limited-area grid from a generated global parent."""

    def build(self, spec: Any, options: Any) -> tuple[GeometryData, TopologyData, MetricsData, RefinementData]:
        from . import grid_generator as gg

        parent = gg.generate_grid(spec.parent_grid_name, options=options)
        selected = _selected_cells(parent, spec)
        selected = _expand_cells(parent, selected, spec.boundary_depth)
        if not selected:
            raise ValueError("limited-area selection does not contain any cells")
        ordered_parent_cells = _order_cells_by_boundary(parent, selected)
        geometry = _compact_geometry(parent, ordered_parent_cells)
        topology = _open_topology(parent, geometry, ordered_parent_cells, options)
        metrics = _limited_metrics(parent, geometry, topology, options.sphere_radius)
        refinement = _limited_refinement(parent, geometry, topology, ordered_parent_cells)
        return geometry, topology, metrics, refinement


def cut_existing_grid(parent: Any, spec: Any) -> tuple[GeometryData, TopologyData, MetricsData, RefinementData]:
    """Cut an existing grid using region predicates."""
    selected = _selected_cells_from_regions(parent, spec)
    if spec.mode == "remove":
        selected = set(range(parent.dims["cell"])) - selected
    selected = _expand_cells(parent, selected, spec.boundary_depth)
    if not selected:
        raise ValueError("cut grid selection does not contain any cells")
    ordered_parent_cells = _order_cells_by_boundary(parent, selected)
    geometry = _compact_geometry(parent, ordered_parent_cells)
    topology = _open_topology(parent, geometry, ordered_parent_cells, parent.options)
    metrics = _limited_metrics(parent, geometry, topology, parent.options.sphere_radius)
    refinement = _limited_refinement(parent, geometry, topology, ordered_parent_cells)
    refinement.fields["smooth_c_ctrl"] = np.full(
        geometry.cells.shape[0],
        spec.smoothing_depth,
        dtype=np.int32,
    )
    return geometry, topology, metrics, refinement


def _selected_cells(parent: Any, spec: Any) -> set[int]:
    return set(np.nonzero(_region_mask(parent.lon, parent.lat, spec.region))[0].astype(int))


def _selected_cells_from_regions(parent: Any, spec: Any) -> set[int]:
    mask = np.zeros(parent.dims["cell"], dtype=bool)
    for region in spec.regions:
        mask |= _region_mask(parent.lon, parent.lat, region)
    return set(np.nonzero(mask)[0].astype(int))


def _region_mask(lon: np.ndarray, lat: np.ndarray, region: Any) -> np.ndarray:
    name = region.__class__.__name__.removeprefix("_")
    if name == "LonLatBoxRegion":
        if region.lon_min <= region.lon_max:
            lon_mask = (lon >= region.lon_min) & (lon <= region.lon_max)
        else:
            lon_mask = (lon >= region.lon_min) | (lon <= region.lon_max)
        return lon_mask & (lat >= region.lat_min) & (lat <= region.lat_max)
    if name == "CircleRegion":
        return _angular_distance_degrees(lon, lat, region.lon, region.lat) <= region.radius_degrees
    if name == "OrientedRectangleRegion":
        dx = _wrapped_lon_delta(lon - region.center_lon) * cos(radians(region.center_lat))
        dy = lat - region.center_lat
        angle = radians(region.angle_degrees)
        x_rot = dx * np.cos(angle) + dy * np.sin(angle)
        y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
        return (
            (np.abs(x_rot) <= 0.5 * region.width_degrees)
            & (np.abs(y_rot) <= 0.5 * region.height_degrees)
        )
    if name == "PolygonRegion":
        return _polygon_mask(lon, lat, region.points)
    raise TypeError(f"unsupported cut region {name}")


def _angular_distance_degrees(
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> np.ndarray:
    lon1 = np.radians(lon)
    lat1 = np.radians(lat)
    lon2 = radians(center_lon)
    lat2 = radians(center_lat)
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = np.sin(0.5 * dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    return np.degrees(2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)))


def _polygon_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    points: tuple[tuple[float, float], ...],
) -> np.ndarray:
    reference = points[0][0]
    x = _wrapped_lon_delta(lon - reference)
    y = lat
    polygon = np.asarray(
        [(_wrapped_lon_delta(point_lon - reference), point_lat) for point_lon, point_lat in points],
        dtype=np.float64,
    )
    inside = np.zeros(lon.shape, dtype=bool)
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1.0e-30) + xi
        )
        inside ^= crosses
        j = i
    return inside


def _wrapped_lon_delta(delta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(delta) + 180.0) % 360.0 - 180.0


def _expand_cells(parent: Any, selected: set[int], depth: int) -> set[int]:
    n_cells = parent.dims["cell"]
    expanded = {cell for cell in selected if 0 <= cell < n_cells}
    frontier = set(expanded)
    for _ in range(depth):
        next_frontier: set[int] = set()
        for cell in frontier:
            for neighbor in parent.icon_connectivity["c2c"][cell]:
                neighbor = int(neighbor)
                if 0 <= neighbor < n_cells:
                    next_frontier.add(neighbor)
        next_frontier -= expanded
        expanded.update(next_frontier)
        frontier = next_frontier
    return expanded


def _order_cells_by_boundary(parent: Any, selected: set[int]) -> np.ndarray:
    boundary = {
        cell
        for cell in selected
        if any(int(neighbor) not in selected for neighbor in parent.icon_connectivity["c2c"][cell])
    }
    levels = {cell: 0 for cell in boundary}
    queue = deque(boundary)
    while queue:
        cell = queue.popleft()
        for neighbor in parent.icon_connectivity["c2c"][cell]:
            neighbor = int(neighbor)
            if neighbor in selected and neighbor not in levels:
                levels[neighbor] = levels[cell] + 1
                queue.append(neighbor)
    return np.asarray(
        sorted(selected, key=lambda cell: (levels.get(cell, 10**9), cell)),
        dtype=np.int32,
    )


def _compact_geometry(parent: Any, parent_cells: np.ndarray) -> GeometryData:
    source_vertices = np.asarray(
        sorted({int(vertex) for cell in parent.cells[parent_cells] for vertex in cell}),
        dtype=np.int32,
    )
    vertex_map = {int(source): local for local, source in enumerate(source_vertices)}
    cells = np.asarray(
        [[vertex_map[int(vertex)] for vertex in parent.cells[cell]] for cell in parent_cells],
        dtype=np.int32,
    )
    return GeometryData(
        vertices=parent.vertices[source_vertices],
        cells=cells,
        lon=parent.lon[parent_cells],
        lat=parent.lat[parent_cells],
        vertex_lon=parent.vertex_lon[source_vertices],
        vertex_lat=parent.vertex_lat[source_vertices],
        cell_center_xyz=parent.cell_center_xyz[parent_cells],
        cell_vertex_lon=parent.cell_vertex_lon[parent_cells],
        cell_vertex_lat=parent.cell_vertex_lat[parent_cells],
        source_cell_index=parent_cells,
        source_vertex_index=source_vertices,
    )


def _open_topology(
    parent: Any,
    geometry: GeometryData,
    parent_cells: np.ndarray,
    options: Any,
) -> TopologyData:
    from . import grid_generator as gg

    edge_ids: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    edge_cells: list[list[int]] = []
    source_edge_ids: list[int] = []
    parent_edge_lookup = {
        tuple(sorted((int(v0), int(v1)))): edge_index
        for edge_index, (v0, v1) in enumerate(parent.edges)
    }
    source_vertex_to_local = {
        int(source): local for local, source in enumerate(geometry.source_vertex_index)
    }
    cell_edges = np.empty((geometry.cells.shape[0], 3), dtype=np.int32)
    for local_cell, cell in enumerate(geometry.cells):
        for local_edge, pair in enumerate(((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0]))):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            edge_id = edge_ids.get(key)
            if edge_id is None:
                edge_id = len(edges)
                edge_ids[key] = edge_id
                edges.append(key)
                edge_cells.append([local_cell])
                source_pair = tuple(
                    sorted(
                        int(geometry.source_vertex_index[vertex])
                        for vertex in key
                    )
                )
                source_edge_ids.append(parent_edge_lookup[source_pair])
            else:
                edge_cells[edge_id].append(local_cell)
            cell_edges[local_cell, local_edge] = edge_id

    edge_cell_array = np.full((len(edges), 2), -1, dtype=np.int32)
    for edge_id, adjacent in enumerate(edge_cells):
        edge_cell_array[edge_id, : len(adjacent)] = adjacent

    edges_array = np.asarray(edges, dtype=np.int32)
    edge_center_xyz = parent.edge_center_xyz[source_edge_ids]
    edge_lon = parent.edge_lon[source_edge_ids]
    edge_lat = parent.edge_lat[source_edge_ids]
    icon_connectivity = _open_icon_connectivity(
        geometry.vertices,
        geometry.cells,
        geometry.cell_center_xyz,
        edges_array,
        cell_edges,
        edge_cell_array,
    )
    del gg, options, parent_cells, source_vertex_to_local
    return TopologyData(
        edges=edges_array,
        cell_edges=cell_edges,
        edge_cells=edge_cell_array,
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon_connectivity,
        connectivity=_open_public_connectivity(geometry.cells, edges_array, edge_cell_array, icon_connectivity),
        neighbor_tables=_open_neighbor_tables(geometry.cells, edges_array, edge_cell_array, icon_connectivity),
        source_edge_index=np.asarray(source_edge_ids, dtype=np.int32),
    )


def _open_icon_connectivity(
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg

    n_vertices = vertices.shape[0]
    c2e = np.asarray(cell_edges, dtype=np.int32)
    c2c = np.full_like(c2e, -1)
    orientation = np.empty_like(c2e)
    for cell_index in range(cells.shape[0]):
        for local_index, edge_index in enumerate(c2e[cell_index]):
            adjacent = edge_cells[edge_index]
            if adjacent[1] < 0:
                c2c[cell_index, local_index] = -1
            else:
                c2c[cell_index, local_index] = (
                    adjacent[1] if adjacent[0] == cell_index else adjacent[0]
                )
            orientation[cell_index, local_index] = 1 if adjacent[0] == cell_index else -1

    incident_cells: list[list[int]] = [[] for _ in range(n_vertices)]
    incident_edges: list[list[int]] = [[] for _ in range(n_vertices)]
    incident_vertices: list[list[int]] = [[] for _ in range(n_vertices)]
    for cell_index, cell in enumerate(cells):
        for vertex in cell:
            incident_cells[int(vertex)].append(cell_index + 1)
    for edge_index, (v0, v1) in enumerate(edges):
        incident_edges[int(v0)].append(edge_index + 1)
        incident_edges[int(v1)].append(edge_index + 1)
        incident_vertices[int(v0)].append(int(v1) + 1)
        incident_vertices[int(v1)].append(int(v0) + 1)

    v2c = np.zeros((n_vertices, 6), dtype=np.int32)
    v2e = np.zeros((n_vertices, 6), dtype=np.int32)
    v2v = np.zeros((n_vertices, 6), dtype=np.int32)
    edge_orientation = np.zeros((n_vertices, 6), dtype=np.int32)
    edge_lookup = {edge_id + 1: tuple(edge) for edge_id, edge in enumerate(edges)}
    edge_centers = gg._edge_centers(vertices, edges, 1.0)
    unit_centers = gg._normalize_rows(cell_center_xyz)
    for vertex in range(n_vertices):
        ordered_vertices = gg._sort_around_vertex(vertices, vertex, incident_vertices[vertex])
        ordered_edges = gg._sort_around_vertex(vertices, vertex, incident_edges[vertex], points=edge_centers)
        ordered_cells = gg._sort_around_vertex(vertices, vertex, incident_cells[vertex], points=unit_centers)
        v2v[vertex, : len(ordered_vertices)] = ordered_vertices
        v2e[vertex, : len(ordered_edges)] = ordered_edges
        v2c[vertex, : len(ordered_cells)] = ordered_cells
        for pos, edge_id in enumerate(ordered_edges):
            edge = edge_lookup[edge_id]
            edge_orientation[vertex, pos] = 1 if edge[0] == vertex else -1
    return {
        "c2e": c2e,
        "c2c": c2c,
        "v2c": v2c,
        "v2e": v2e,
        "v2v": v2v,
        "orientation_of_normal": orientation,
        "edge_orientation": edge_orientation,
    }


def _open_public_connectivity(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg

    return {
        "edge_of_cell": icon_connectivity["c2e"],
        "vertex_of_cell": cells,
        "neighbor_cell_index": icon_connectivity["c2c"],
        "adjacent_cell_of_edge": edge_cells,
        "edge_vertices": edges,
        "cells_of_vertex": gg._zero_based_with_skip(icon_connectivity["v2c"]),
        "edges_of_vertex": gg._zero_based_with_skip(icon_connectivity["v2e"]),
        "vertices_of_vertex": gg._zero_based_with_skip(icon_connectivity["v2v"]),
    }


def _open_neighbor_tables(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    from . import grid_generator as gg

    return {
        "c2e2c": icon_connectivity["c2c"],
        "c2e": icon_connectivity["c2e"],
        "e2c": np.asarray(edge_cells, dtype=np.int32),
        "v2e": gg._zero_based_with_skip(icon_connectivity["v2e"]),
        "v2c": gg._zero_based_with_skip(icon_connectivity["v2c"]),
        "c2v": np.asarray(cells, dtype=np.int32),
        "v2e2v": gg._zero_based_with_skip(icon_connectivity["v2v"]),
        "e2v": np.asarray(edges, dtype=np.int32),
    }


def _limited_metrics(parent: Any, geometry: GeometryData, topology: TopologyData, sphere_radius: float) -> MetricsData:
    from . import grid_generator as gg

    source_edges = topology.source_edge_index
    edge_lengths = parent.geometry["edge_length"][source_edges]
    dual_edge_lengths = parent.geometry["dual_edge_length"][source_edges].copy()
    edge_cell_distance = np.empty((topology.edges.shape[0], 2), dtype=np.float64)
    for edge_index, adjacent in enumerate(topology.edge_cells):
        for side in range(2):
            if adjacent[side] >= 0:
                center = geometry.cell_center_xyz[adjacent[side]][np.newaxis, :]
                edge_center = topology.edge_center_xyz[edge_index][np.newaxis, :]
                edge_cell_distance[edge_index, side] = gg._edge_cell_distances(
                    center,
                    np.array([[0, 0]], dtype=np.int32),
                    edge_center,
                    sphere_radius,
                )[0, 0]
            else:
                edge_cell_distance[edge_index, side] = edge_cell_distance[edge_index, 0]
                dual_edge_lengths[edge_index] = 2.0 * edge_cell_distance[edge_index, 0]
    edge_system_orientation = np.ones(topology.edges.shape[0], dtype=np.int32)
    normals = gg._edge_normal_fields(
        geometry.vertices,
        topology.edges,
        topology.edge_center_xyz,
        edge_system_orientation,
    )
    cell_areas = parent.geometry["cell_area"][geometry.source_cell_index]
    return MetricsData(
        fields={
            "cell_area": cell_areas,
            "dual_area": gg._dual_areas(geometry.vertices.shape[0], geometry.cells, cell_areas),
            "edge_length": edge_lengths,
            "dual_edge_length": dual_edge_lengths,
            "edge_cell_distance": edge_cell_distance,
            "edge_vert_distance": np.column_stack((edge_lengths * 0.5, edge_lengths * 0.5)),
            "orientation_of_normal": topology.icon_connectivity["orientation_of_normal"],
            "edge_system_orientation": edge_system_orientation,
            "edge_orientation": topology.icon_connectivity["edge_orientation"],
            "edgequad_area": 0.5 * edge_lengths * dual_edge_lengths,
            **normals,
        }
    )


def _limited_refinement(
    parent: Any,
    geometry: GeometryData,
    topology: TopologyData,
    parent_cells: np.ndarray,
) -> RefinementData:
    from . import grid_generator as gg

    n_cells = geometry.cells.shape[0]
    n_edges = topology.edges.shape[0]
    n_vertices = geometry.vertices.shape[0]
    boundary_distance = _boundary_distance(topology)
    refin_c_ctrl = np.asarray(boundary_distance + 1, dtype=np.int32)
    edge_ctrl = np.zeros(n_edges, dtype=np.int32)
    for edge_index, adjacent in enumerate(topology.edge_cells):
        active = adjacent[adjacent >= 0]
        edge_ctrl[edge_index] = int(np.max(refin_c_ctrl[active]))
    vertex_ctrl = np.zeros(n_vertices, dtype=np.int32)
    for cell_index, cell in enumerate(geometry.cells):
        vertex_ctrl[cell] = np.maximum(vertex_ctrl[cell], refin_c_ctrl[cell_index])
    return RefinementData(
        fields={
            "refin_c_ctrl": refin_c_ctrl,
            "refin_e_ctrl": edge_ctrl,
            "refin_v_ctrl": vertex_ctrl,
            "start_idx_c": gg._start_index_fixed("cell_grf", n_cells),
            "end_idx_c": gg._end_index_fixed("cell_grf", n_cells),
            "start_idx_e": gg._start_index_fixed("edge_grf", n_edges),
            "end_idx_e": gg._end_index_fixed("edge_grf", n_edges),
            "start_idx_v": gg._start_index_fixed("vert_grf", n_vertices),
            "end_idx_v": gg._end_index_fixed("vert_grf", n_vertices),
            "parent_cell_index": parent_cells.astype(np.int32) + 1,
            "parent_cell_type": np.zeros(n_cells, dtype=np.int32),
            "edge_parent_type": np.zeros(n_edges, dtype=np.int32),
            "parent_edge_index": topology.source_edge_index.astype(np.int32) + 1,
            "parent_vertex_index": geometry.source_vertex_index.astype(np.int32) + 1,
        }
    )


def _boundary_distance(topology: TopologyData) -> np.ndarray:
    boundary_cells = {
        int(cell)
        for adjacent in topology.edge_cells
        if adjacent[1] < 0
        for cell in adjacent
        if cell >= 0
    }
    distances = np.full(topology.cell_edges.shape[0], -1, dtype=np.int32)
    queue = deque(boundary_cells)
    for cell in boundary_cells:
        distances[cell] = 0
    while queue:
        cell = queue.popleft()
        for neighbor in topology.icon_connectivity["c2c"][cell]:
            neighbor = int(neighbor)
            if neighbor >= 0 and distances[neighbor] < 0:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances
