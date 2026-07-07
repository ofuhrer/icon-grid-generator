"""Global spherical grid generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._geometry import SphericalIcosahedralGeometry
from ._metrics import SphericalMetricsBuilder
from ._optimization import _GlobalOptimizationOptions, optimize_global_grid
from ._ordering import IconOrderingBuilder
from ._refinement import GlobalRefinementBuilder
from ._topology import GlobalTopologyBuilder
from ._types import BisectionProvenance, GeometryData, TopologyData

GLOBAL_RELAXATION_LONG_ITER_CELL_THRESHOLD = 100_000

@dataclass
class _GlobalGenerationContext:
    """Shared internal state for one global generation request."""

    grids: dict[tuple[int, int], Any] = field(default_factory=dict)
    parent_data: dict[tuple[int, int], "_GlobalParentData"] = field(default_factory=dict)
    parent_vertex_indices: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def key(self, spec: Any) -> tuple[int, int]:
        return spec.root, spec.bisections


@dataclass(frozen=True)
class _GlobalParentData:
    """Geometry and topology needed for child ordering and provenance."""

    spec: Any
    vertices: np.ndarray
    cells: np.ndarray
    edges: np.ndarray
    cell_edges: np.ndarray
    edge_center_xyz: np.ndarray


def _generate_grid(
    spec: Any,
    options: Any,
    context: _GlobalGenerationContext | None = None,
) -> Any:
    if context is None:
        context = _GlobalGenerationContext()
    cache_key = context.key(spec)
    cached = context.grids.get(cache_key)
    if cached is not None:
        return cached

    if options.global_optimization.method == "spring" or spec.bisections > 0:
        grid = _generate_staged_global_grid(spec, options, context)
    else:
        grid = _generate_raw_global_grid(spec, options, context)
    context.grids[cache_key] = grid
    return grid


def _generate_raw_global_grid(
    spec: Any,
    options: Any,
    context: _GlobalGenerationContext,
) -> Any:
    geometry = SphericalIcosahedralGeometry().build(spec, options)
    geometry = IconOrderingBuilder(context).order_spherical_bisection(spec, options, geometry)
    return _assemble_global_grid(spec, options, geometry, context)


def _generate_staged_global_grid(
    spec: Any,
    options: Any,
    context: _GlobalGenerationContext,
) -> Any:
    if spec.bisections == 0:
        return _generate_raw_global_grid(spec, options, context)

    parent_spec = _gg().GlobalGridSpec(root=spec.root, bisections=spec.bisections - 1)
    parent = _generate_grid(parent_spec, options, context)
    vertices, cells, provenance = _gg()._refine_triangles_bisection_with_provenance(
        parent.vertices,
        parent.cells,
        options.accelerator,
    )
    vertices = vertices * options.radius
    geometry = _geometry_from_vertices(spec, options, vertices, cells, provenance)
    grid = _assemble_global_grid(spec, options, geometry, context)
    stage_iterations = _stage_global_optimization_iterations(options, parent)
    if stage_iterations == 0:
        _evict_staged_parent_cache(context, parent_spec, spec)
        return grid
    relaxed = optimize_global_grid(
        grid,
        _GlobalOptimizationOptions(method="spring", iterations=stage_iterations),
    )
    _evict_staged_parent_cache(context, parent_spec, spec)
    return relaxed


def _evict_staged_parent_cache(
    context: _GlobalGenerationContext,
    parent_spec: Any,
    child_spec: Any,
) -> None:
    parent_key = context.key(parent_spec)
    child_key = context.key(child_spec)
    context.grids.pop(parent_key, None)
    context.parent_data.pop(parent_key, None)
    context.parent_vertex_indices.pop(parent_key, None)
    context.parent_vertex_indices.pop(child_key, None)


def _geometry_from_vertices(
    spec: Any,
    options: Any,
    vertices: np.ndarray,
    cells: np.ndarray,
    provenance: BisectionProvenance | None,
) -> GeometryData:
    _gg()._check_expected_counts(spec, vertices, cells)
    vertex_lon, vertex_lat = _gg()._lon_lat(vertices)
    cell_center_xyz = _gg()._cell_centers(vertices, cells, options.radius)
    lon, lat = _gg()._lon_lat(cell_center_xyz)
    return GeometryData(
        vertices=vertices,
        cells=cells,
        lon=lon,
        lat=lat,
        vertex_lon=vertex_lon,
        vertex_lat=vertex_lat,
        cell_center_xyz=cell_center_xyz,
        cell_vertex_lon=vertex_lon[cells],
        cell_vertex_lat=vertex_lat[cells],
        bisection_provenance=provenance,
    )


def _stage_global_optimization_iterations(
    options: Any,
    parent: Any,
) -> int:
    iterations = options.global_optimization.iterations
    if parent.dims["cell"] < GLOBAL_RELAXATION_LONG_ITER_CELL_THRESHOLD:
        return iterations * 10
    return iterations


def _assemble_global_grid(
    spec: Any,
    options: Any,
    geometry: GeometryData,
    context: _GlobalGenerationContext,
) -> Any:
    topology = GlobalTopologyBuilder().build(spec, options, geometry)
    topology = _adjust_global_edge_orientation(
        spec,
        options,
        geometry,
        topology,
        context,
    )
    metrics = SphericalMetricsBuilder().build(options, geometry, topology)
    refinement = GlobalRefinementBuilder(context).build(spec, options, geometry, topology)
    metadata = _gg()._metadata(spec, options, metrics.fields)

    grid = _gg().IconGrid(
        spec=spec,
        options=options,
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
        refinement=refinement.fields,
        metadata=metadata,
    )
    return grid


def _adjust_global_edge_orientation(
    spec: Any,
    options: Any,
    geometry: GeometryData,
    topology: TopologyData,
    context: _GlobalGenerationContext,
) -> TopologyData:
    if spec.bisections == 0:
        return topology
    parent_spec = _gg().GlobalGridSpec(root=spec.root, bisections=spec.bisections - 1)
    parent = _generate_grid(parent_spec, options, context)
    provenance = geometry.bisection_provenance
    if provenance is None:
        parent_vertex_index = _gg()._parent_vertex_indices(geometry.vertices, parent)
        parent_for_mapping: Any | _GlobalParentData | BisectionProvenance = parent
        parent_normals = parent.geometry["edge_primal_normal_cartesian"]
    else:
        parent_vertex_index = provenance.parent_vertex_index
        parent_for_mapping = provenance
        parent_edge_map = _matching_edge_indices_by_vertices(provenance.edges, parent.edges)
        parent_normals = parent.geometry["edge_primal_normal_cartesian"][parent_edge_map]
    if (
        isinstance(parent_for_mapping, BisectionProvenance)
        and parent_for_mapping.child_parent_edge_index is not None
    ):
        parent_edge_index = parent_for_mapping.child_parent_edge_index
    else:
        parent_edge_index, _ = _gg()._parent_edge_fields(
            topology.edges,
            parent_vertex_index,
            parent_for_mapping,
            options.accelerator,
        )

    edge_system_orientation = _gg()._edge_system_orientation(
        geometry.vertices,
        geometry.cell_center_xyz,
        topology.edges,
        topology.edge_cells,
        topology.edge_center_xyz,
    )
    child_normals = _gg()._edge_normal_fields(
        geometry.vertices,
        topology.edges,
        topology.edge_center_xyz,
        edge_system_orientation,
    )["edge_primal_normal_cartesian"]
    alignment = np.sum(
        child_normals * parent_normals[parent_edge_index.astype(np.int64) - 1],
        axis=1,
    )
    flip = alignment < 0.0
    if not np.any(flip):
        return topology

    edges = topology.edges.copy()
    edge_cells = topology.edge_cells.copy()
    edges[flip] = edges[flip][:, ::-1]
    edge_cells[flip] = edge_cells[flip][:, ::-1]

    edge_center_xyz = _gg()._edge_centers(geometry.vertices, edges, options.radius)
    edge_lon, edge_lat = _gg()._lon_lat(edge_center_xyz)
    icon_connectivity = _gg()._icon_connectivity(
        geometry.vertices,
        geometry.cells,
        geometry.cell_center_xyz,
        edges,
        topology.cell_edges,
        edge_cells,
    )
    return TopologyData(
        edges=edges,
        cell_edges=topology.cell_edges,
        edge_cells=edge_cells,
        edge_center_xyz=edge_center_xyz,
        edge_lon=edge_lon,
        edge_lat=edge_lat,
        icon_connectivity=icon_connectivity,
        connectivity=_gg()._public_connectivity(
            geometry.cells,
            edges,
            edge_cells,
            icon_connectivity,
        ),
        neighbor_tables=_gg()._neighbor_tables(
            geometry.cells,
            edges,
            edge_cells,
            icon_connectivity,
        ),
        source_edge_index=topology.source_edge_index,
    )


def _nearest_unit_point_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    unit_source = _gg()._normalize_rows(source)
    unit_target = _gg()._normalize_rows(target)
    block_size = 1024
    indices = np.empty(unit_source.shape[0], dtype=np.int64)
    for start in range(0, unit_source.shape[0], block_size):
        stop = min(start + block_size, unit_source.shape[0])
        distances = np.sum(
            (unit_source[start:stop, np.newaxis, :] - unit_target[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        indices[start:stop] = np.argmin(distances, axis=1)
    return indices


def _matching_unit_point_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup = {
        _gg()._point_key(point): point_index
        for point_index, point in enumerate(_gg()._normalize_rows(target))
    }
    indices = np.empty(source.shape[0], dtype=np.int64)
    for point_index, point in enumerate(_gg()._normalize_rows(source)):
        match = lookup.get(_gg()._point_key(point))
        if match is None:
            raise RuntimeError(f"point {point_index} has no matching target point")
        indices[point_index] = match
    return indices


def _matching_edge_indices_by_vertices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_keys = np.sort(source.astype(np.int64, copy=False), axis=1)
    target_keys = np.sort(target.astype(np.int64, copy=False), axis=1)
    order = np.lexsort(tuple(target_keys[:, column] for column in range(target_keys.shape[1] - 1, -1, -1)))
    sorted_keys = np.ascontiguousarray(target_keys[order])
    sorted_view = _gg()._row_view(sorted_keys)
    source_view = _gg()._row_view(source_keys)
    positions = np.searchsorted(sorted_view, source_view)
    valid = positions < sorted_view.shape[0]
    found = np.zeros(source_view.shape[0], dtype=bool)
    found[valid] = sorted_view[positions[valid]] == source_view[valid]
    missing = np.flatnonzero(~found)
    if missing.size:
        raise RuntimeError(f"edge {int(missing[0])} has no matching target edge")
    return order[positions].astype(np.int64, copy=False)


def _parent_grid(
    spec: Any,
    options: Any,
    context: _GlobalGenerationContext,
) -> Any | _GlobalParentData:
    if spec.bisections == 0:
        raise ValueError("grid has no bisection parent")
    parent_spec = _gg().GlobalGridSpec(root=spec.root, bisections=spec.bisections - 1)
    parent_key = context.key(parent_spec)
    full_parent = context.grids.get(parent_key)
    if full_parent is not None:
        return full_parent
    parent_data = context.parent_data.get(parent_key)
    if parent_data is not None:
        return parent_data

    parent_geometry = SphericalIcosahedralGeometry().build(parent_spec, options)
    parent_geometry = IconOrderingBuilder(context).order_spherical_bisection(
        parent_spec,
        options,
        parent_geometry,
    )
    parent_topology = GlobalTopologyBuilder().build(parent_spec, options, parent_geometry)
    parent_data = _GlobalParentData(
        spec=parent_spec,
        vertices=parent_geometry.vertices,
        cells=parent_geometry.cells,
        edges=parent_topology.edges,
        cell_edges=parent_topology.cell_edges,
        edge_center_xyz=parent_topology.edge_center_xyz,
    )
    context.parent_data[parent_key] = parent_data
    return parent_data


def _parent_vertex_indices_cached(
    spec: Any,
    options: Any,
    vertices: np.ndarray,
    context: _GlobalGenerationContext,
) -> tuple[np.ndarray, Any | _GlobalParentData]:
    parent = _parent_grid(spec, options, context)
    cache_key = context.key(spec)
    parent_vertex_index = context.parent_vertex_indices.get(cache_key)
    if parent_vertex_index is None:
        parent_vertex_index = _gg()._parent_vertex_indices(vertices, parent)
        context.parent_vertex_indices[cache_key] = parent_vertex_index
    return parent_vertex_index, parent


def _gg() -> Any:
    from . import grid_generator as gg

    return gg
