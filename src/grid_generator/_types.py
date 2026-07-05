"""Internal data containers for the grid generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BisectionProvenance:
    """Immediate-parent topology and ICON provenance for one bisection step."""

    cells: np.ndarray
    edges: np.ndarray
    cell_edges: np.ndarray
    parent_vertex_index: np.ndarray
    parent_cell_index: np.ndarray
    parent_cell_type: np.ndarray


@dataclass(frozen=True)
class GeometryData:
    """Primal grid points and cell centers."""

    vertices: np.ndarray
    cells: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    vertex_lon: np.ndarray
    vertex_lat: np.ndarray
    cell_center_xyz: np.ndarray
    cell_vertex_lon: np.ndarray
    cell_vertex_lat: np.ndarray
    source_cell_index: np.ndarray | None = None
    source_vertex_index: np.ndarray | None = None
    bisection_provenance: BisectionProvenance | None = None


@dataclass(frozen=True)
class TopologyData:
    """Edges, adjacency tables, and ICON connectivity for a closed triangular grid."""

    edges: np.ndarray
    cell_edges: np.ndarray
    edge_cells: np.ndarray
    edge_center_xyz: np.ndarray
    edge_lon: np.ndarray
    edge_lat: np.ndarray
    icon_connectivity: dict[str, np.ndarray]
    connectivity: dict[str, np.ndarray]
    neighbor_tables: dict[str, np.ndarray]
    source_edge_index: np.ndarray | None = None


@dataclass(frozen=True)
class MetricsData:
    """Metric and orientation fields derived from geometry and topology."""

    fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class RefinementData:
    """Refinement-control and parent-provenance fields."""

    fields: dict[str, np.ndarray]
