"""Planar doubly periodic triangular torus grids."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from ._types import GeometryData, MetricsData, RefinementData, TopologyData


class PlanarTorusGeometry:
    """Build a triangular, doubly periodic planar grid."""

    def build(self, spec: Any, options: Any) -> GeometryData:
        nx = spec.nx
        ny = spec.ny
        edge_length = spec.edge_length
        height = sqrt(3.0) * 0.5 * edge_length
        x_period = nx * edge_length
        y_period = ny * height

        vertices = np.zeros((nx * ny, 3), dtype=np.float64)
        for j in range(ny):
            for i in range(nx):
                vertices[_vertex_id(i, j, nx, ny)] = (
                    (i + 0.5 * j) * edge_length,
                    j * height,
                    options.radius,
                )

        cells: list[tuple[int, int, int]] = []
        centers: list[np.ndarray] = []
        for j in range(ny):
            for i in range(nx):
                up = (
                    _vertex_id(i, j, nx, ny),
                    _vertex_id(i + 1, j, nx, ny),
                    _vertex_id(i, j + 1, nx, ny),
                )
                down = (
                    _vertex_id(i, j, nx, ny),
                    _vertex_id(i + 1, j - 1, nx, ny),
                    _vertex_id(i + 1, j, nx, ny),
                )
                for cell in (up, down):
                    cells.append(cell)
                    centers.append(_periodic_triangle_center(vertices, cell, x_period, y_period))

        cell_array = np.asarray(cells, dtype=np.int32)
        center_array = np.asarray(centers, dtype=np.float64)
        lon = _scale_to_degrees(center_array[:, 0], x_period, -180.0, 180.0)
        lat = _scale_to_degrees(center_array[:, 1], y_period, -90.0, 90.0)
        vertex_lon = _scale_to_degrees(vertices[:, 0], x_period, -180.0, 180.0)
        vertex_lat = _scale_to_degrees(vertices[:, 1], y_period, -90.0, 90.0)
        return GeometryData(
            vertices=vertices,
            cells=cell_array,
            lon=lon,
            lat=lat,
            vertex_lon=vertex_lon,
            vertex_lat=vertex_lat,
            cell_center_xyz=center_array,
            cell_vertex_lon=vertex_lon[cell_array],
            cell_vertex_lat=vertex_lat[cell_array],
            source_cell_index=np.arange(cell_array.shape[0], dtype=np.int32),
            source_vertex_index=np.arange(vertices.shape[0], dtype=np.int32),
        )


class PeriodicTopologyBuilder:
    """Build closed torus topology using modulo vertex connectivity."""

    def build(self, spec: Any, options: Any, geometry: GeometryData) -> TopologyData:
        from . import grid_generator as gg

        edges, cell_edges, edge_cells = gg._build_edges(geometry.cells)
        if edges.shape[0] != spec.expected_edges:
            raise RuntimeError(
                f"generated {edges.shape[0]} edges, expected {spec.expected_edges}"
            )
        edge_center_xyz = _edge_centers(
            geometry.vertices,
            edges,
            spec.domain_length,
            spec.domain_height,
            options.radius,
        )
        edge_lon = _scale_to_degrees(edge_center_xyz[:, 0], spec.domain_length, -180.0, 180.0)
        edge_lat = _scale_to_degrees(edge_center_xyz[:, 1], spec.domain_height, -90.0, 90.0)
        icon_connectivity = gg._icon_connectivity(
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
            icon_connectivity=icon_connectivity,
            connectivity=gg._public_connectivity(
                geometry.cells,
                edges,
                edge_cells,
                icon_connectivity,
            ),
            neighbor_tables=gg._neighbor_tables(
                geometry.cells,
                edges,
                edge_cells,
                icon_connectivity,
            ),
            source_edge_index=np.arange(edges.shape[0], dtype=np.int32),
        )


class PlanarTorusMetricsBuilder:
    """Compute planar torus metrics."""

    def build(self, spec: Any, geometry: GeometryData, topology: TopologyData) -> MetricsData:
        edge_length = spec.edge_length
        cell_area = sqrt(3.0) * 0.25 * edge_length**2
        dual_edge_length = edge_length / sqrt(3.0)
        edge_vectors = _edge_vectors(
            geometry.vertices,
            topology.edges,
            spec.domain_length,
            spec.domain_height,
        )
        tangent = edge_vectors / np.linalg.norm(edge_vectors, axis=1)[:, np.newaxis]
        primal_normal = np.column_stack((-tangent[:, 1], tangent[:, 0], np.zeros(tangent.shape[0])))
        dual_normal = tangent
        n_cells = geometry.cells.shape[0]
        n_edges = topology.edges.shape[0]
        n_vertices = geometry.vertices.shape[0]
        fields = {
            "cell_area": np.full(n_cells, cell_area, dtype=np.float64),
            "dual_area": np.full(n_vertices, 2.0 * cell_area, dtype=np.float64),
            "edge_length": np.full(n_edges, edge_length, dtype=np.float64),
            "dual_edge_length": np.full(n_edges, dual_edge_length, dtype=np.float64),
            "edge_cell_distance": np.full((n_edges, 2), 0.5 * dual_edge_length, dtype=np.float64),
            "edge_vert_distance": np.full((n_edges, 2), 0.5 * edge_length, dtype=np.float64),
            "orientation_of_normal": topology.icon_connectivity["orientation_of_normal"],
            "edge_system_orientation": np.ones(n_edges, dtype=np.int32),
            "edge_orientation": topology.icon_connectivity["edge_orientation"],
            "edgequad_area": np.full(n_edges, 0.5 * edge_length * dual_edge_length),
            "edge_primal_normal_cartesian": primal_normal,
            "edge_dual_normal_cartesian": dual_normal,
            "zonal_normal_primal_edge": primal_normal[:, 0],
            "meridional_normal_primal_edge": primal_normal[:, 1],
            "zonal_normal_dual_edge": dual_normal[:, 0],
            "meridional_normal_dual_edge": dual_normal[:, 1],
        }
        return MetricsData(fields=fields)


class TorusRefinementBuilder:
    """Return default refinement fields for a standalone torus grid."""

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


def _vertex_id(i: int, j: int, nx: int, ny: int) -> int:
    return (j % ny) * nx + (i % nx)


def _periodic_delta(delta: np.ndarray, period: float) -> np.ndarray:
    return delta - np.round(delta / period) * period


def _periodic_triangle_center(
    vertices: np.ndarray,
    cell: tuple[int, int, int],
    x_period: float,
    y_period: float,
) -> np.ndarray:
    base = vertices[cell[0]].copy()
    points = [base]
    for vertex in cell[1:]:
        point = vertices[vertex].copy()
        point[0] = base[0] + _periodic_delta(point[0] - base[0], x_period)
        point[1] = base[1] + _periodic_delta(point[1] - base[1], y_period)
        points.append(point)
    center = np.mean(points, axis=0)
    center[0] %= x_period
    center[1] %= y_period
    return center


def _edge_vectors(
    vertices: np.ndarray,
    edges: np.ndarray,
    x_period: float,
    y_period: float,
) -> np.ndarray:
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    vectors[:, 0] = _periodic_delta(vectors[:, 0], x_period)
    vectors[:, 1] = _periodic_delta(vectors[:, 1], y_period)
    return vectors


def _edge_centers(
    vertices: np.ndarray,
    edges: np.ndarray,
    x_period: float,
    y_period: float,
    z_value: float,
) -> np.ndarray:
    vectors = _edge_vectors(vertices, edges, x_period, y_period)
    centers = vertices[edges[:, 0]] + 0.5 * vectors
    centers[:, 0] %= x_period
    centers[:, 1] %= y_period
    centers[:, 2] = z_value
    return centers


def _scale_to_degrees(values: np.ndarray, period: float, start: float, end: float) -> np.ndarray:
    return start + (np.asarray(values, dtype=np.float64) % period) / period * (end - start)
