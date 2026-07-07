"""Pure Python ICON-style geodesic grid generation.

The generator accepts ICON R<n>B<k> grid names and canonicalizes them to the
zero-padded form commonly used in ICON grid file names. It creates triangular
spherical grids with the topology, metric, orientation, normal-vector, and
refinement-provenance fields needed to write a compact ICON grid NetCDF file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping
import re
import json
import uuid

import numpy as np

from ._io import IconNetcdfWriter
from ._limited_area import LimitedAreaExtractor
from ._optimization import (
    _GlobalGridOptions,
    _GlobalOptimizationOptions,
)
from ._planar import (
    PlanarRefinementBuilder,
    PlanarTriangularGeometry,
    PlanarTriangularMetricsBuilder,
    PlanarTriangularTopologyBuilder,
)
from ._torus import (
    PeriodicTopologyBuilder,
    PlanarTorusGeometry,
    PlanarTorusMetricsBuilder,
    TorusRefinementBuilder,
)
from ._types import BisectionProvenance, GeometryData
from ._global import (
    _GlobalGenerationContext as _GlobalGenerationContext,
    _GlobalParentData as _GlobalParentData,
    _adjust_global_edge_orientation as _adjust_global_edge_orientation,
    _generate_grid,
    _generate_raw_global_grid as _generate_raw_global_grid,
    _geometry_from_vertices as _geometry_from_vertices,
    _matching_edge_indices_by_vertices as _matching_edge_indices_by_vertices,
    _matching_unit_point_indices as _matching_unit_point_indices,
    _nearest_unit_point_indices as _nearest_unit_point_indices,
    _parent_vertex_indices_cached as _parent_vertex_indices_cached,
)
from ._validation import finite_float_option, validate_grid_options
from . import _accelerated

GRID_NAME_RE = re.compile(r"^R0*(\d+)B0*(\d+)$", re.IGNORECASE)
EARTH_RADIUS_M = 6_371_229.0
POINT_MATCH_DECIMALS = 11
XYZ_LABELS = np.array(["x", "y", "z"])
CELL_VERTEX_LABELS = np.array([0, 1, 2], dtype=np.int32)
EDGE_VERTEX_LABELS = np.array([0, 1], dtype=np.int32)
EDGE_CELL_LABELS = np.array([0, 1], dtype=np.int32)
FIXED_DIMS = {
    "nc": 2,
    "nv": 3,
    "ne": 6,
    "no": 4,
    "max_chdom": 1,
    "cell_grf": 14,
    "edge_grf": 24,
    "vert_grf": 13,
}
ACTIVE_REFINEMENT_START = {
    "cell_grf": 9,
    "edge_grf": 14,
    "vert_grf": 8,
}
CHILD_CELL_TYPE_CENTER = 200
CHILD_CELL_TYPE_AT_VERTEX_0 = 201
CHILD_CELL_TYPE_AT_VERTEX_1 = 202
CHILD_CELL_TYPE_AT_VERTEX_2 = 203
EDGE_CHILD_TYPE_FROM_VERTEX_0 = 101
EDGE_CHILD_TYPE_FROM_VERTEX_1 = 102
EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_0 = 201
EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_1 = 202
EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_2 = 203
@dataclass(frozen=True)
class GlobalGridSpec:
    """Normalized ICON R<n>B<k> grid specification."""

    root: int
    bisections: int
    frequency: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.root, int) or isinstance(self.root, bool):
            raise TypeError("global grid root must be an integer")
        if self.root < 1:
            raise ValueError("global grid root must be at least 1")
        if not isinstance(self.bisections, int) or isinstance(self.bisections, bool):
            raise TypeError("global grid bisections must be an integer")
        if self.bisections < 0:
            raise ValueError("global grid bisections must be non-negative")

        expected_frequency = self.root * 2**self.bisections
        if self.frequency not in (0, expected_frequency):
            raise ValueError("global grid frequency must equal root * 2**bisections")
        object.__setattr__(self, "frequency", expected_frequency)
        canonical_name = f"R{self.root:02d}B{self.bisections:02d}"
        if not isinstance(self.name, str):
            raise TypeError("global grid name must be a string")
        if not self.name.strip():
            object.__setattr__(self, "name", canonical_name)
            return

        match = GRID_NAME_RE.fullmatch(self.name.strip())
        if match is None:
            raise ValueError("global grid name must have the form R<n>B<k>")
        name_root = int(match.group(1))
        name_bisections = int(match.group(2))
        if name_root != self.root or name_bisections != self.bisections:
            raise ValueError("global grid name must match root and bisections")
        object.__setattr__(self, "name", canonical_name)

    @property
    def expected_cells(self) -> int:
        return 20 * self.frequency**2

    @property
    def expected_edges(self) -> int:
        return 30 * self.frequency**2

    @property
    def expected_vertices(self) -> int:
        return 10 * self.frequency**2 + 2


@dataclass(frozen=True)
class TorusGridSpec:
    """Planar triangular torus grid specification."""

    nx: int
    ny: int
    edge_length: float
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.nx, int) or isinstance(self.nx, bool) or self.nx < 3:
            raise ValueError("torus nx must be an integer greater than or equal to 3")
        if not isinstance(self.ny, int) or isinstance(self.ny, bool) or self.ny < 3:
            raise ValueError("torus ny must be an integer greater than or equal to 3")
        edge_length = _finite_float_option("edge_length", self.edge_length)
        if edge_length <= 0.0:
            raise ValueError("edge_length must be positive")
        if not self.name:
            object.__setattr__(self, "name", f"TORUS{self.nx}x{self.ny}")

    @property
    def expected_cells(self) -> int:
        return 2 * self.nx * self.ny

    @property
    def expected_edges(self) -> int:
        return 3 * self.nx * self.ny

    @property
    def expected_vertices(self) -> int:
        return self.nx * self.ny

    @property
    def domain_length(self) -> float:
        return self.nx * self.edge_length

    @property
    def domain_height(self) -> float:
        return self.ny * np.sqrt(3.0) * 0.5 * self.edge_length


@dataclass(frozen=True)
class StretchedTorusGridSpec:
    """Planar triangular torus grid with anisotropic coordinate stretching."""

    nx: int
    ny: int
    edge_length: float
    stretch_x: float = 1.0
    stretch_y: float = 1.0
    name: str = ""

    periodic: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_planar_counts("stretched torus", self.nx, self.ny, minimum=3)
        edge_length = _finite_float_option("edge_length", self.edge_length)
        stretch_x = _finite_float_option("stretch_x", self.stretch_x)
        stretch_y = _finite_float_option("stretch_y", self.stretch_y)
        if edge_length <= 0.0:
            raise ValueError("edge_length must be positive")
        if stretch_x <= 0.0 or stretch_y <= 0.0:
            raise ValueError("stretch factors must be positive")
        object.__setattr__(self, "edge_length", edge_length)
        object.__setattr__(self, "stretch_x", stretch_x)
        object.__setattr__(self, "stretch_y", stretch_y)
        if not self.name:
            object.__setattr__(self, "name", f"STRETCHED_TORUS{self.nx}x{self.ny}")

    @property
    def expected_cells(self) -> int:
        return 2 * self.nx * self.ny

    @property
    def expected_edges(self) -> int:
        return 3 * self.nx * self.ny

    @property
    def expected_vertices(self) -> int:
        return self.nx * self.ny

    @property
    def domain_length(self) -> float:
        return self.nx * self.edge_length * self.stretch_x

    @property
    def domain_height(self) -> float:
        return self.ny * np.sqrt(3.0) * 0.5 * self.edge_length * self.stretch_y


@dataclass(frozen=True)
class ChannelGridSpec:
    """Open planar triangular channel grid."""

    nx: int
    ny: int
    edge_length: float
    name: str = ""

    periodic: bool = field(default=False, init=False, repr=False)
    periodic_x: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.nx, int) or isinstance(self.nx, bool) or self.nx < 3:
            raise ValueError("channel nx must be an integer greater than or equal to 3")
        if not isinstance(self.ny, int) or isinstance(self.ny, bool) or self.ny < 2:
            raise ValueError("channel ny must be an integer greater than or equal to 2")
        edge_length = _finite_float_option("edge_length", self.edge_length)
        if edge_length <= 0.0:
            raise ValueError("edge_length must be positive")
        object.__setattr__(self, "edge_length", edge_length)
        if not self.name:
            object.__setattr__(self, "name", f"CHANNEL{self.nx}x{self.ny}")

    @property
    def expected_cells(self) -> int:
        return 2 * self.nx * self.ny

    @property
    def expected_edges(self) -> int:
        return self.nx * (3 * self.ny + 1)

    @property
    def expected_vertices(self) -> int:
        return self.nx * (self.ny + 1)


@dataclass(frozen=True)
class ParallelogramGridSpec:
    """Open planar triangular parallelogram grid."""

    nx: int
    ny: int
    edge_length: float
    shear: float = 0.0
    name: str = ""

    periodic: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_planar_counts("parallelogram", self.nx, self.ny)
        edge_length = _finite_float_option("edge_length", self.edge_length)
        shear = _finite_float_option("shear", self.shear)
        if edge_length <= 0.0:
            raise ValueError("edge_length must be positive")
        object.__setattr__(self, "edge_length", edge_length)
        object.__setattr__(self, "shear", shear)
        if not self.name:
            object.__setattr__(self, "name", f"PARALLELOGRAM{self.nx}x{self.ny}")

    @property
    def expected_cells(self) -> int:
        return 2 * self.nx * self.ny

    @property
    def expected_edges(self) -> int:
        return 3 * self.nx * self.ny + self.nx + self.ny

    @property
    def expected_vertices(self) -> int:
        return (self.nx + 1) * (self.ny + 1)


@dataclass(frozen=True)
class RaggedOrthogonalGridSpec:
    """Open triangular grid on a deterministic ragged orthogonal lattice."""

    nx: int
    ny: int
    dx: float
    dy: float
    raggedness: float = 0.15
    name: str = ""

    periodic: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_planar_counts("ragged orthogonal", self.nx, self.ny)
        dx = _finite_float_option("dx", self.dx)
        dy = _finite_float_option("dy", self.dy)
        raggedness = _finite_float_option("raggedness", self.raggedness)
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError("dx and dy must be positive")
        if not 0.0 <= raggedness < 0.45:
            raise ValueError("raggedness must be in [0, 0.45)")
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "raggedness", raggedness)
        if not self.name:
            object.__setattr__(self, "name", f"RAGGED_ORTHOGONAL{self.nx}x{self.ny}")

    @property
    def expected_cells(self) -> int:
        return 2 * self.nx * self.ny

    @property
    def expected_edges(self) -> int:
        return 3 * self.nx * self.ny + self.nx + self.ny

    @property
    def expected_vertices(self) -> int:
        return (self.nx + 1) * (self.ny + 1)


@dataclass(frozen=True)
class LimitedAreaGridSpec:
    """Limited-area grid extracted from a generated global parent grid."""

    parent: str | GlobalGridSpec
    region: Any
    boundary_depth: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        parent = parse_grid_spec(self.parent) if isinstance(self.parent, str) else self.parent
        if not isinstance(parent, GlobalGridSpec):
            raise TypeError("limited-area parent must be a global grid spec or grid name")
        region = _normalize_region(self.region)
        if not isinstance(self.boundary_depth, int) or isinstance(self.boundary_depth, bool):
            raise TypeError("boundary_depth must be a non-negative integer")
        if self.boundary_depth < 0:
            raise ValueError("boundary_depth must be non-negative")
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "parent_grid_name", parent.name)
        if not self.name:
            object.__setattr__(self, "name", f"LAM_{parent.name}")

    @property
    def expected_cells(self) -> int:
        return 0

    @property
    def expected_edges(self) -> int:
        return 0

    @property
    def expected_vertices(self) -> int:
        return 0


@dataclass(frozen=True)
class _LonLatBoxRegion:
    """Select cells whose centers fall in a longitude/latitude box."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        lon_min = _finite_float_option("lon_min", self.lon_min)
        lon_max = _finite_float_option("lon_max", self.lon_max)
        lat_min = _finite_float_option("lat_min", self.lat_min)
        lat_max = _finite_float_option("lat_max", self.lat_max)
        if not -180.0 <= lon_min <= 180.0 or not -180.0 <= lon_max <= 180.0:
            raise ValueError("longitude bounds must be within [-180, 180]")
        if not -90.0 <= lat_min <= 90.0 or not -90.0 <= lat_max <= 90.0:
            raise ValueError("latitude bounds must be within [-90, 90]")
        if lat_min > lat_max:
            raise ValueError("lat_min must be less than or equal to lat_max")
        object.__setattr__(self, "lon_min", lon_min)
        object.__setattr__(self, "lon_max", lon_max)
        object.__setattr__(self, "lat_min", lat_min)
        object.__setattr__(self, "lat_max", lat_max)


@dataclass(frozen=True)
class _CircleRegion:
    """Select cells within an angular radius of a lon/lat center."""

    lon: float
    lat: float
    radius_degrees: float

    def __post_init__(self) -> None:
        lon = _finite_float_option("lon", self.lon)
        lat = _finite_float_option("lat", self.lat)
        radius = _finite_float_option("radius_degrees", self.radius_degrees)
        if not -180.0 <= lon <= 180.0:
            raise ValueError("lon must be within [-180, 180]")
        if not -90.0 <= lat <= 90.0:
            raise ValueError("lat must be within [-90, 90]")
        if radius <= 0.0:
            raise ValueError("radius_degrees must be positive")
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "radius_degrees", radius)


@dataclass(frozen=True)
class _OrientedRectangleRegion:
    """Select cells inside a rotated local lon/lat rectangle."""

    center_lon: float
    center_lat: float
    width_degrees: float
    height_degrees: float
    angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        center_lon = _finite_float_option("center_lon", self.center_lon)
        center_lat = _finite_float_option("center_lat", self.center_lat)
        width = _finite_float_option("width_degrees", self.width_degrees)
        height = _finite_float_option("height_degrees", self.height_degrees)
        angle = _finite_float_option("angle_degrees", self.angle_degrees)
        if not -180.0 <= center_lon <= 180.0:
            raise ValueError("center_lon must be within [-180, 180]")
        if not -90.0 <= center_lat <= 90.0:
            raise ValueError("center_lat must be within [-90, 90]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("rectangle width and height must be positive")
        object.__setattr__(self, "center_lon", center_lon)
        object.__setattr__(self, "center_lat", center_lat)
        object.__setattr__(self, "width_degrees", width)
        object.__setattr__(self, "height_degrees", height)
        object.__setattr__(self, "angle_degrees", angle)


@dataclass(frozen=True)
class _PolygonRegion:
    """Select cells inside a lon/lat polygon."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        points = tuple(tuple(point) for point in self.points)
        if len(points) < 3:
            raise ValueError("polygon requires at least three points")
        normalized: list[tuple[float, float]] = []
        for lon, lat in points:
            lon = _finite_float_option("polygon longitude", lon)
            lat = _finite_float_option("polygon latitude", lat)
            if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
                raise ValueError("polygon points must be valid lon/lat pairs")
            normalized.append((lon, lat))
        object.__setattr__(self, "points", tuple(normalized))


class Region:
    """Factory namespace for public grid-cut and limited-area region specs."""

    @staticmethod
    def lonlat_box(
        *,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> _LonLatBoxRegion:
        return _LonLatBoxRegion(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
        )

    @staticmethod
    def circle(*, lon: float, lat: float, radius_degrees: float) -> _CircleRegion:
        return _CircleRegion(lon=lon, lat=lat, radius_degrees=radius_degrees)

    @staticmethod
    def rectangle(
        *,
        center_lon: float,
        center_lat: float,
        width_degrees: float,
        height_degrees: float,
        angle_degrees: float = 0.0,
    ) -> _OrientedRectangleRegion:
        return _OrientedRectangleRegion(
            center_lon=center_lon,
            center_lat=center_lat,
            width_degrees=width_degrees,
            height_degrees=height_degrees,
            angle_degrees=angle_degrees,
        )

    @staticmethod
    def polygon(points: tuple[tuple[float, float], ...]) -> _PolygonRegion:
        return _PolygonRegion(points=points)


_RegionSpec = _LonLatBoxRegion | _CircleRegion | _OrientedRectangleRegion | _PolygonRegion


def _normalize_region(region: Any) -> _RegionSpec:
    supported_region_types = (
        _LonLatBoxRegion,
        _CircleRegion,
        _OrientedRectangleRegion,
        _PolygonRegion,
    )
    if not isinstance(region, supported_region_types):
        raise TypeError("region must be created with Region")
    return region


def _normalize_regions(regions: Any) -> tuple[_RegionSpec, ...]:
    if isinstance(regions, (_LonLatBoxRegion, _CircleRegion, _OrientedRectangleRegion, _PolygonRegion)):
        normalized = (regions,)
    else:
        normalized = tuple(regions)
    if not normalized:
        raise ValueError("cut grid spec requires at least one region")
    for region in normalized:
        _normalize_region(region)
    return normalized


def _region_class_name(region: Any) -> str:
    return region.__class__.__name__.removeprefix("_")


@dataclass(frozen=True)
class CutGridSpec:
    """Selection options for extracting a cut grid from an existing grid."""

    regions: Any
    mode: str = "keep"
    boundary_depth: int = 0
    smoothing_depth: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        regions = _normalize_regions(self.regions)
        if self.mode not in {"keep", "remove"}:
            raise ValueError("cut mode must be 'keep' or 'remove'")
        for name, value in {
            "boundary_depth": self.boundary_depth,
            "smoothing_depth": self.smoothing_depth,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "regions", regions)
        if not self.name:
            object.__setattr__(self, "name", "CUT_GRID")


_PLANAR_GRID_SPEC_TYPES = (
    StretchedTorusGridSpec,
    ChannelGridSpec,
    ParallelogramGridSpec,
    RaggedOrthogonalGridSpec,
)
_SUPPORTED_GRID_SPEC_TYPES = (
    GlobalGridSpec,
    TorusGridSpec,
    LimitedAreaGridSpec,
    *_PLANAR_GRID_SPEC_TYPES,
)


@dataclass(frozen=True)
class IconGridOptions:
    """Options for pure Python ICON grid generation."""

    max_cells: int | None = 2_000_000
    accelerator: str = "auto"
    radius: float = 1.0
    sphere_radius: float = EARTH_RADIUS_M
    optimize_global: bool = True
    spring_beta: float = 0.9
    spring_iterations: int = 2000
    fixed_boundary: bool = True
    north_pole_lon: float = 0.0
    north_pole_lat: float = 90.0
    rotation_angle_degrees: float = 0.0
    indexing: str = "new"
    centre: int = 78
    subcentre: int = 255
    number_of_grid_used: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.optimize_global, bool):
            raise TypeError("optimize_global must be a boolean")
        global_grid = _GlobalGridOptions(
            beta_spring=self.spring_beta,
            maxit=self.spring_iterations,
            fixed_boundary=self.fixed_boundary,
            north_pole_lon=self.north_pole_lon,
            north_pole_lat=self.north_pole_lat,
            rotation_angle_degrees=self.rotation_angle_degrees,
            indexing_algorithm=self.indexing,
            centre=self.centre,
            subcentre=self.subcentre,
            number_of_grid_used=self.number_of_grid_used,
        )
        global_optimization = _GlobalOptimizationOptions(
            method="spring" if self.optimize_global else "none",
            iterations=global_grid.maxit if self.optimize_global else 0,
        )
        object.__setattr__(self, "global_grid", global_grid)
        object.__setattr__(self, "global_optimization", global_optimization)


@dataclass(frozen=True)
class IconGrid:
    """ICON grid geometry, topology, metrics, and NetCDF export support."""

    spec: Any
    options: IconGridOptions
    vertices: np.ndarray
    cells: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    vertex_lon: np.ndarray
    vertex_lat: np.ndarray
    cell_center_xyz: np.ndarray
    cell_vertex_lon: np.ndarray
    cell_vertex_lat: np.ndarray
    edges: np.ndarray
    cell_edges: np.ndarray
    edge_cells: np.ndarray
    edge_center_xyz: np.ndarray
    edge_lon: np.ndarray
    edge_lat: np.ndarray
    icon_connectivity: dict[str, np.ndarray] = field(default_factory=dict)
    connectivity: dict[str, np.ndarray] = field(default_factory=dict)
    neighbor_tables: dict[str, np.ndarray] = field(default_factory=dict)
    geometry: dict[str, np.ndarray] = field(default_factory=dict)
    refinement: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dims(self) -> dict[str, int]:
        dims = {
            "cell": int(self.cells.shape[0]),
            "vertex": int(self.vertices.shape[0]),
            "edge": int(self.edges.shape[0]),
        }
        return dims

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary with arrays commonly used by plotting helpers."""
        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.name,
            "spec": self.spec,
            "dims": self.dims,
            "vertices": self.vertices,
            "cells": self.cells,
            "lon": self.lon,
            "lat": self.lat,
            "vertex_lon": self.vertex_lon,
            "vertex_lat": self.vertex_lat,
            "cell_center_xyz": self.cell_center_xyz,
            "cell_vertex_lon": self.cell_vertex_lon,
            "cell_vertex_lat": self.cell_vertex_lat,
        }
        data["edges"] = self.edges
        data["cell_edges"] = self.cell_edges
        data["edge_cells"] = self.edge_cells
        data["edge_center_xyz"] = self.edge_center_xyz
        data["edge_lon"] = self.edge_lon
        data["edge_lat"] = self.edge_lat
        if self.connectivity:
            data["connectivity"] = self.connectivity
        if self.neighbor_tables:
            data["neighbor_tables"] = self.neighbor_tables
        if self.geometry:
            data["geometry"] = self.geometry
        if self.refinement:
            data["refinement"] = self.refinement
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def to_xarray(self) -> Any:
        """Return an xarray Dataset, importing xarray only when requested."""
        import xarray as xr

        data_vars: dict[str, Any] = {
            "vertices": (("vertex", "xyz"), self.vertices),
            "cells": (("cell", "cell_vertex"), self.cells),
            "lon": (("cell",), self.lon),
            "lat": (("cell",), self.lat),
            "vertex_lon": (("vertex",), self.vertex_lon),
            "vertex_lat": (("vertex",), self.vertex_lat),
            "cell_center_xyz": (("cell", "xyz"), self.cell_center_xyz),
            "cell_vertex_lon": (("cell", "cell_vertex"), self.cell_vertex_lon),
            "cell_vertex_lat": (("cell", "cell_vertex"), self.cell_vertex_lat),
        }
        coords: dict[str, Any] = {
            "xyz": XYZ_LABELS,
            "cell_vertex": CELL_VERTEX_LABELS,
        }
        data_vars["edges"] = (("edge", "edge_vertex"), self.edges)
        data_vars["cell_edges"] = (("cell", "cell_vertex"), self.cell_edges)
        data_vars["edge_cells"] = (("edge", "edge_cell"), self.edge_cells)
        data_vars["edge_center_xyz"] = (("edge", "xyz"), self.edge_center_xyz)
        data_vars["edge_lon"] = (("edge",), self.edge_lon)
        data_vars["edge_lat"] = (("edge",), self.edge_lat)
        coords["edge_vertex"] = EDGE_VERTEX_LABELS
        coords["edge_cell"] = EDGE_CELL_LABELS

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "name": self.name,
                "root": getattr(self.spec, "root", 0),
                "bisections": getattr(self.spec, "bisections", 0),
                "frequency": getattr(self.spec, "frequency", 0),
                "radius": self.options.radius,
            },
        )

    def to_netcdf(self, path: str | Any, *, sphere_radius: float | None = None) -> Any:
        """Write an ICON-style NetCDF grid file."""
        return IconNetcdfWriter().write(self, path, sphere_radius=sphere_radius)


def generate_grid(
    spec: str | GlobalGridSpec | TorusGridSpec | LimitedAreaGridSpec | Any,
    options: IconGridOptions | Mapping[str, Any] | None = None,
) -> IconGrid:
    """Create a pure Python ICON geodesic, torus, or limited-area grid."""
    grid_spec = parse_grid_spec(spec) if isinstance(spec, str) else spec
    if not isinstance(grid_spec, _SUPPORTED_GRID_SPEC_TYPES):
        raise TypeError("spec must be an ICON R<n>B<k> string or a supported grid spec")
    resolved_options = _resolve_options(options)
    explicit_optimize_global = isinstance(options, Mapping) and "optimize_global" in options
    if (
        not isinstance(grid_spec, GlobalGridSpec)
        and resolved_options.global_optimization.method == "spring"
        and not explicit_optimize_global
    ):
        resolved_options = IconGridOptions(
            max_cells=resolved_options.max_cells,
            accelerator=resolved_options.accelerator,
            radius=resolved_options.radius,
            sphere_radius=resolved_options.sphere_radius,
            optimize_global=False,
            spring_beta=resolved_options.spring_beta,
            spring_iterations=resolved_options.spring_iterations,
            fixed_boundary=resolved_options.fixed_boundary,
            north_pole_lon=resolved_options.north_pole_lon,
            north_pole_lat=resolved_options.north_pole_lat,
            rotation_angle_degrees=resolved_options.rotation_angle_degrees,
            indexing=resolved_options.indexing,
            centre=resolved_options.centre,
            subcentre=resolved_options.subcentre,
            number_of_grid_used=resolved_options.number_of_grid_used,
        )
    _validate_options(grid_spec, resolved_options)

    if isinstance(grid_spec, TorusGridSpec):
        return _generate_torus_grid(grid_spec, resolved_options)
    if isinstance(grid_spec, StretchedTorusGridSpec) and np.isclose(
        [grid_spec.stretch_x, grid_spec.stretch_y],
        [1.0, 1.0],
    ).all():
        return _generate_torus_grid(grid_spec, resolved_options)
    if isinstance(grid_spec, _PLANAR_GRID_SPEC_TYPES):
        return _generate_planar_grid(grid_spec, resolved_options)
    if isinstance(grid_spec, LimitedAreaGridSpec):
        return _generate_limited_area_grid(grid_spec, resolved_options)
    return _generate_grid(grid_spec, resolved_options, _GlobalGenerationContext())


def _validate_options(
    spec: GlobalGridSpec | TorusGridSpec | LimitedAreaGridSpec,
    options: IconGridOptions,
) -> None:
    validate_grid_options(spec, options)


def _finite_float_option(name: str, value: Any) -> float:
    return finite_float_option(name, value)


def _validate_planar_counts(name: str, nx: Any, ny: Any, *, minimum: int = 1) -> None:
    if not isinstance(nx, int) or isinstance(nx, bool) or nx < minimum:
        raise ValueError(f"{name} nx must be an integer greater than or equal to {minimum}")
    if not isinstance(ny, int) or isinstance(ny, bool) or ny < minimum:
        raise ValueError(f"{name} ny must be an integer greater than or equal to {minimum}")





def parse_grid_spec(
    grid_name: str | GlobalGridSpec | TorusGridSpec | LimitedAreaGridSpec | Any,
) -> GlobalGridSpec | TorusGridSpec | LimitedAreaGridSpec | Any:
    """Parse and normalize an ICON R<n>B<k> grid name."""
    if isinstance(grid_name, _SUPPORTED_GRID_SPEC_TYPES):
        return grid_name
    if not isinstance(grid_name, str):
        raise TypeError("grid_name must be a string such as 'R2B3' or 'R02B03'")

    match = GRID_NAME_RE.fullmatch(grid_name.strip())
    if match is None:
        raise ValueError("grid_name must have the form R<n>B<k>, for example R2B3 or R02B03")

    root = int(match.group(1))
    bisections = int(match.group(2))
    if root < 1:
        raise ValueError("grid root must be at least 1")
    if bisections < 0:
        raise ValueError("grid bisections must be non-negative")

    return GlobalGridSpec(
        root=root,
        bisections=bisections,
    )


def _resolve_options(options: IconGridOptions | Mapping[str, Any] | None) -> IconGridOptions:
    if options is None:
        return IconGridOptions()
    if isinstance(options, IconGridOptions):
        return options
    if not isinstance(options, Mapping):
        raise TypeError("options must be None, an IconGridOptions instance, or a mapping")

    allowed = set(IconGridOptions.__dataclass_fields__)
    unknown = set(options) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown grid option(s): {names}")
    return IconGridOptions(**dict(options))




























def _generate_torus_grid(spec: TorusGridSpec, options: IconGridOptions) -> IconGrid:
    geometry = PlanarTorusGeometry().build(spec, options)
    topology = PeriodicTopologyBuilder().build(spec, options, geometry)
    metrics = PlanarTorusMetricsBuilder().build(spec, geometry, topology)
    refinement = TorusRefinementBuilder().build(geometry, topology)
    metadata = _metadata(spec, options, metrics.fields)
    return IconGrid(
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


def _generate_planar_grid(spec: Any, options: IconGridOptions) -> IconGrid:
    geometry = PlanarTriangularGeometry().build(spec, options)
    topology = PlanarTriangularTopologyBuilder().build(spec, geometry)
    if topology.edges.shape[0] != spec.expected_edges:
        raise RuntimeError(
            f"generated {topology.edges.shape[0]} edges, expected {spec.expected_edges}"
        )
    metrics = PlanarTriangularMetricsBuilder().build(spec, geometry, topology)
    refinement = PlanarRefinementBuilder().build(geometry, topology)
    metadata = _metadata(spec, options, metrics.fields)
    return IconGrid(
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


def _generate_limited_area_grid(spec: LimitedAreaGridSpec, options: IconGridOptions) -> IconGrid:
    geometry, topology, metrics, refinement = LimitedAreaExtractor().build(spec, options)
    metadata = _metadata(spec, options, metrics.fields)
    return IconGrid(
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


def cut_grid(grid: Any, spec: CutGridSpec) -> IconGrid:
    """Extract an open cut grid from an existing in-memory grid."""
    from ._limited_area import cut_existing_grid

    geometry, topology, metrics, refinement = cut_existing_grid(grid, spec)
    metadata = _metadata(spec, grid.options, metrics.fields)
    metadata.update(
        {
            "source_grid_name": grid.name,
            "boundary_depth_index": spec.boundary_depth,
            "smoothing_depth": spec.smoothing_depth,
            "cut_mode": spec.mode,
        }
    )
    return IconGrid(
        spec=spec,
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
        refinement=refinement.fields,
        metadata=metadata,
    )


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    z = 1.0 / np.sqrt(5.0)
    x_major = 2.0 / np.sqrt(5.0)
    x_minor = (np.sqrt(5.0) - 1.0) / (2.0 * np.sqrt(5.0))
    x_mid = (1.0 + np.sqrt(5.0)) / (2.0 * np.sqrt(5.0))
    y_mid = np.sqrt((5.0 - np.sqrt(5.0)) / 10.0)
    y_major = np.sqrt((5.0 + np.sqrt(5.0)) / 10.0)
    vertices = np.asarray(
        [
            (0.0, 0.0, 1.0),
            (-x_major, 0.0, z),
            (x_major, 0.0, -z),
            (0.0, 0.0, -1.0),
            (x_mid, -y_mid, z),
            (x_mid, y_mid, z),
            (-x_mid, -y_mid, -z),
            (-x_mid, y_mid, -z),
            (-x_minor, -y_major, z),
            (-x_minor, y_major, z),
            (x_minor, -y_major, -z),
            (x_minor, y_major, -z),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (1, 2, 9),
            (1, 9, 5),
            (1, 5, 6),
            (1, 6, 10),
            (1, 10, 2),
            (2, 7, 9),
            (9, 7, 11),
            (9, 11, 5),
            (5, 11, 3),
            (5, 3, 6),
            (6, 3, 12),
            (6, 12, 10),
            (10, 12, 8),
            (10, 8, 2),
            (2, 8, 7),
            (4, 7, 8),
            (4, 8, 12),
            (4, 12, 3),
            (4, 3, 11),
            (4, 11, 7),
        ],
        dtype=np.int32,
    ) - 1
    return vertices, faces


def _sadourny_root_grid(root: int) -> tuple[np.ndarray, np.ndarray, BisectionProvenance | None]:
    if root < 1:
        raise ValueError("root must be at least 1")
    if root == 1:
        vertices, cells = _icosahedron()
        return vertices, cells, None

    base_vertices, base_faces = _icosahedron()
    vertices: list[np.ndarray] = [point.copy() for point in base_vertices]
    directed_edge_vertices: dict[tuple[int, int], list[int]] = {}

    def subdivide(first: int, second: int) -> list[int]:
        key = (first, second)
        existing = directed_edge_vertices.get(key)
        if existing is not None:
            return existing

        reverse_key = (second, first)
        reverse = directed_edge_vertices.get(reverse_key)
        if reverse is not None:
            values = list(reversed(reverse))
            directed_edge_vertices[key] = values
            return values

        values = [first]
        for cut in range(1, root):
            point = (root - cut) * base_vertices[first] + cut * base_vertices[second]
            values.append(len(vertices))
            vertices.append(_normalize(point))
        values.append(second)
        directed_edge_vertices[key] = values
        directed_edge_vertices[reverse_key] = list(reversed(values))
        return values

    vertex_neighbors: dict[int, dict[int, int]] = {}
    edges: list[tuple[int, int]] = []
    cell_edges: list[tuple[int, int, int]] = []

    def edge_index(first: int, second: int) -> int:
        neighbors = vertex_neighbors.setdefault(first, {})
        existing = neighbors.get(second)
        if existing is not None:
            return existing

        index = len(edges)
        edges.append((first, second))
        neighbors[second] = index
        vertex_neighbors.setdefault(second, {})[first] = index
        return index

    for face in base_faces:
        a, b, c = map(int, face)
        v0 = [a] + [-1] * root
        for row in range(1, root + 1):
            if row == 1:
                v1 = [subdivide(a, b)[row], subdivide(a, c)[row]] + [-1] * (root - 1)
            elif row == root:
                v1 = subdivide(b, c).copy()
            else:
                left = subdivide(a, b)[row]
                right = subdivide(a, c)[row]
                row_vertices = [left]
                for cut in range(1, row):
                    point = (row - cut) * vertices[left] + cut * vertices[right]
                    row_vertices.append(len(vertices))
                    vertices.append(_normalize(point))
                row_vertices.append(right)
                v1 = row_vertices + [-1] * (root - row)

            new_edge_indices: list[int] = []
            for index in range(row):
                new_edge_indices.extend(
                    [
                        edge_index(v0[index], v1[index]),
                        edge_index(v1[index], v1[index + 1]),
                        edge_index(v1[index + 1], v0[index]),
                    ]
                )
            for index in range(row - 1):
                new_edge_indices.extend(
                    [
                        edge_index(v0[index], v0[index + 1]),
                        edge_index(v0[index + 1], v1[index + 1]),
                        edge_index(v1[index + 1], v0[index]),
                    ]
                )
            cell_edges.extend(
                tuple(new_edge_indices[index : index + 3])
                for index in range(0, len(new_edge_indices), 3)
            )
            v0 = v1

    vertex_array = _normalize_rows(np.asarray(vertices, dtype=np.float64))
    edge_array = np.asarray(edges, dtype=np.int32)
    cell_edge_array = np.asarray(cell_edges, dtype=np.int32)
    cell_array = _cells_from_edge_cells(vertex_array, edge_array, cell_edge_array)
    edge_cell_array = _edge_cells_from_cell_edges(cell_edge_array, edge_array.shape[0])
    edge_array = _edge_vertices_from_cell_edges(cell_array, cell_edge_array, edge_cell_array)
    provenance = BisectionProvenance(
        cells=np.empty((0, 3), dtype=np.int32),
        edges=np.empty((0, 2), dtype=np.int32),
        cell_edges=np.empty((0, 3), dtype=np.int32),
        parent_vertex_index=np.zeros(vertex_array.shape[0], dtype=np.int32),
        parent_cell_index=np.zeros(cell_array.shape[0], dtype=np.int32),
        parent_cell_type=np.zeros(cell_array.shape[0], dtype=np.int32),
        child_edges=edge_array,
        child_cell_edges=cell_edge_array,
        child_edge_cells=edge_cell_array,
    )
    return vertex_array, cell_array, provenance


def _cells_from_edge_cells(
    vertices: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
) -> np.ndarray:
    cells = np.empty_like(cell_edges)
    for cell_index, edge_indices in enumerate(cell_edges):
        cell_vertices = np.empty(3, dtype=np.int32)
        for local_index, edge_index in enumerate(edge_indices):
            previous = edge_indices[local_index - 1]
            previous_edge = edges[previous]
            first, second = edges[edge_index]
            if first in previous_edge:
                cell_vertices[local_index] = first
            elif second in previous_edge:
                cell_vertices[local_index] = second
            else:
                raise RuntimeError("cell edges do not share a vertex")
        if _orient_cell(tuple(map(int, cell_vertices)), vertices) != tuple(cell_vertices):
            cell_vertices[[0, 1]] = cell_vertices[[1, 0]]
            cell_edges[cell_index, [1, 2]] = cell_edges[cell_index, [2, 1]]
        cells[cell_index] = cell_vertices
    return cells


def _normalize(point: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(point)
    if norm == 0:
        raise RuntimeError("cannot normalize a zero-length grid point")
    return point / norm


def _normalize_rows(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms == 0.0):
        raise RuntimeError("cannot normalize zero-length grid point rows")
    return points / norms[:, np.newaxis]


def _rotate_points(
    points: np.ndarray,
    axis: tuple[float, float, float],
    angle_degrees: float,
) -> np.ndarray:
    """Rotate Cartesian sphere points around `axis` by `angle_degrees`."""
    if angle_degrees == 0.0:
        return points.copy()
    axis_vector = np.asarray(axis, dtype=np.float64)
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    angle = np.radians(angle_degrees)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross = np.cross(axis_vector, points)
    projection = np.sum(points * axis_vector, axis=1)[:, np.newaxis] * axis_vector
    rotated = points * cos_angle + cross * sin_angle + projection * (1.0 - cos_angle)
    return _normalize_rows(rotated)


def _apply_global_grid_rotation(points: np.ndarray, options: _GlobalGridOptions) -> np.ndarray:
    """Apply compatible pole placement and north-pole rotation."""

    if (
        options.north_pole_lon == 0.0
        and options.north_pole_lat == 90.0
        and options.rotation_angle_degrees == 0.0
    ):
        return points.copy()
    matrix = _global_grid_rotation_matrix(options)
    return _normalize_rows(points @ matrix.T)


def _global_grid_rotation_matrix(options: _GlobalGridOptions) -> np.ndarray:
    default = _global_grid_seed_vertices(_GlobalGridOptions())
    target = _global_grid_seed_vertices(options)
    u, _, vt = np.linalg.svd(default.T @ target)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0.0:
        vt[-1] *= -1.0
        matrix = vt.T @ u.T
    return matrix


def _global_grid_seed_vertices(options: _GlobalGridOptions) -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = _normalize_rows(
        np.asarray(
            [
                (0.0, 1.0, phi),
                (0.0, -1.0, phi),
                (0.0, 1.0, -phi),
                (0.0, -1.0, -phi),
                (1.0, phi, 0.0),
                (-1.0, phi, 0.0),
                (1.0, -phi, 0.0),
                (-1.0, -phi, 0.0),
                (phi, 0.0, 1.0),
                (-phi, 0.0, 1.0),
                (phi, 0.0, -1.0),
                (-phi, 0.0, -1.0),
            ],
            dtype=np.float64,
        )
    )
    first = vertices[0]
    first_lon = np.degrees(np.arctan2(first[1], first[0]))
    first_lat = np.degrees(np.arcsin(np.clip(first[2], -1.0, 1.0)))
    rotated = _unrotate_latlon(vertices, first_lon, first_lat)
    return _raw_global_grid_rotation(rotated, options)


def _raw_global_grid_rotation(points: np.ndarray, options: _GlobalGridOptions) -> np.ndarray:
    rotated = _unrotate_latlon(
        points,
        options.north_pole_lon,
        options.north_pole_lat,
    )
    if options.rotation_angle_degrees != 0.0:
        target = np.array(
            [
                np.cos(np.radians(options.north_pole_lat))
                * np.cos(np.radians(options.north_pole_lon)),
                np.cos(np.radians(options.north_pole_lat))
                * np.sin(np.radians(options.north_pole_lon)),
                np.sin(np.radians(options.north_pole_lat)),
            ],
            dtype=np.float64,
        )
        rotated = _rotate_points(rotated, tuple(target), options.rotation_angle_degrees)
    return rotated


def _unrotate_latlon(
    points: np.ndarray,
    pole_lon_degrees: float,
    pole_lat_degrees: float,
) -> np.ndarray:
    unit_points = _normalize_rows(points)
    lon = np.arctan2(unit_points[:, 1], unit_points[:, 0])
    lat = np.arcsin(np.clip(unit_points[:, 2], -1.0, 1.0))
    pole_lon = np.radians(pole_lon_degrees)
    pole_lat = np.radians(pole_lat_degrees)
    lon_delta = lon - pole_lon
    lon_numerator = -np.sin(lon_delta) * np.cos(lat)
    lon_denominator = (
        -np.sin(pole_lat) * np.cos(lat) * np.cos(lon_delta)
        + np.cos(pole_lat) * np.sin(lat)
    )
    rotated_lon = np.where(
        np.abs(lon_denominator) > 1.0e-15,
        np.arctan2(lon_numerator, lon_denominator),
        0.0,
    )
    rotated_lat = np.arcsin(
        np.clip(
            np.sin(lat) * np.sin(pole_lat)
            + np.cos(lat) * np.cos(pole_lat) * np.cos(lon_delta),
            -1.0,
            1.0,
        )
    )
    return np.column_stack(
        (
            np.cos(rotated_lat) * np.cos(rotated_lon),
            np.cos(rotated_lat) * np.sin(rotated_lon),
            np.sin(rotated_lat),
        )
    )

def _orient_cell(cell: tuple[int, int, int], vertices: Any) -> tuple[int, int, int]:
    a, b, c = (vertices[index] for index in cell)
    normal = np.cross(b - a, c - a)
    if np.dot(normal, a + b + c) < 0:
        return (cell[0], cell[2], cell[1])
    return cell


def _refine_triangles_bisection(
    vertices: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each triangle into four children using typed array operations."""
    new_vertices, new_cells, _ = _refine_triangles_bisection_with_provenance(
        vertices,
        cells,
    )
    return new_vertices, new_cells


def _refine_triangles_bisection_with_provenance(
    vertices: np.ndarray,
    cells: np.ndarray,
    accelerator: str = "auto",
) -> tuple[np.ndarray, np.ndarray, BisectionProvenance]:
    """Split triangles into ICON-ordered bisection children and provenance."""
    edge_vertices, cell_edges, _ = _build_edges(cells)
    old_vertex_count = vertices.shape[0]
    old_edge_count = edge_vertices.shape[0]
    old_cell_count = cells.shape[0]
    edge_midpoint_index = (
        old_vertex_count + np.arange(old_edge_count, dtype=np.int32)
    )
    midpoint_vertices = 0.5 * (
        vertices[edge_vertices[:, 0]] + vertices[edge_vertices[:, 1]]
    )
    new_vertices = np.vstack((vertices, midpoint_vertices))

    new_cell_count = old_cell_count * 4
    new_edge_count = old_edge_count * 2 + old_cell_count * 3
    split_edge_index = (
        2 * np.arange(old_edge_count, dtype=np.int32)[:, np.newaxis]
        + np.array([0, 1], dtype=np.int32)
    )
    inner_edge_index = (
        2 * old_edge_count
        + 3 * np.arange(old_cell_count, dtype=np.int32)[:, np.newaxis]
        + np.array([0, 1, 2], dtype=np.int32)
    )

    use_compiled_fill = _accelerated.should_use_numba_ordering(accelerator, new_cell_count)
    if use_compiled_fill:
        (
            new_cells,
            raw_cell_edges,
            new_edges,
            child_parent_edge_index,
            child_edge_parent_type,
            failure_cell,
            failure_kind,
        ) = _accelerated.fill_bisection_children_numba(
            cells,
            edge_vertices,
            cell_edges,
            edge_midpoint_index,
            split_edge_index,
            inner_edge_index,
            EDGE_CHILD_TYPE_FROM_VERTEX_0,
            EDGE_CHILD_TYPE_FROM_VERTEX_1,
            EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_0,
            EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_1,
            EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_2,
        )
        if failure_kind == 1:
            raise RuntimeError("parent cell edges do not share exactly one vertex")
        if failure_kind == 2:
            raise RuntimeError("parent vertex is not present exactly once in parent cell")
        if failure_kind == 3:
            raise RuntimeError("vertex is not an endpoint of parent edge")
        if failure_cell >= 0:
            raise RuntimeError(f"cell {failure_cell} could not be refined")
    else:
        new_cells = np.empty((new_cell_count, 3), dtype=np.int32)
        raw_cell_edges = np.empty((new_cell_count, 3), dtype=np.int32)
        new_edges = np.empty((new_edge_count, 2), dtype=np.int32)

        for edge_index, (first, second) in enumerate(edge_vertices):
            midpoint = edge_midpoint_index[edge_index]
            new_edges[split_edge_index[edge_index, 0]] = (first, midpoint)
            new_edges[split_edge_index[edge_index, 1]] = (midpoint, second)

        edge_pairs_by_vertex = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        child_slot_by_vertex = np.array([2, 3, 1], dtype=np.int32)
        for cell_index, cell in enumerate(cells):
            parent_edges = cell_edges[cell_index]
            midpoints = edge_midpoint_index[parent_edges]
            center_cell = 4 * cell_index
            new_cells[center_cell] = midpoints
            raw_cell_edges[center_cell] = inner_edge_index[cell_index]

            for first_edge_pos, second_edge_pos, opposite_edge_pos in edge_pairs_by_vertex:
                first_edge = parent_edges[first_edge_pos]
                second_edge = parent_edges[second_edge_pos]
                common_vertex = _common_edge_vertex(
                    edge_vertices[first_edge],
                    edge_vertices[second_edge],
                )
                vertex_pos = _local_vertex_position(cell, common_vertex)
                child_cell = center_cell + int(child_slot_by_vertex[vertex_pos])
                first_split_slot = _edge_endpoint_slot(edge_vertices[first_edge], common_vertex)
                second_split_slot = _edge_endpoint_slot(edge_vertices[second_edge], common_vertex)
                first_midpoint = edge_midpoint_index[first_edge]
                second_midpoint = edge_midpoint_index[second_edge]

                new_cells[child_cell] = (first_midpoint, common_vertex, second_midpoint)
                raw_cell_edges[child_cell] = (
                    split_edge_index[first_edge, first_split_slot],
                    split_edge_index[second_edge, second_split_slot],
                    inner_edge_index[cell_index, vertex_pos],
                )
                new_edges[inner_edge_index[cell_index, vertex_pos]] = (
                    first_midpoint,
                    second_midpoint,
                )
        child_parent_edge_index = np.empty(new_edge_count, dtype=np.int32)
        child_edge_parent_type = np.empty(new_edge_count, dtype=np.int32)
        parent_edge_ids = np.arange(1, old_edge_count + 1, dtype=np.int32)
        child_parent_edge_index[split_edge_index[:, 0]] = parent_edge_ids
        child_parent_edge_index[split_edge_index[:, 1]] = parent_edge_ids
        child_edge_parent_type[split_edge_index[:, 0]] = EDGE_CHILD_TYPE_FROM_VERTEX_0
        child_edge_parent_type[split_edge_index[:, 1]] = EDGE_CHILD_TYPE_FROM_VERTEX_1
        parent_cell_edges = cell_edges.astype(np.int32, copy=False) + 1
        child_parent_edge_index[inner_edge_index[:, 0]] = parent_cell_edges[:, 1]
        child_parent_edge_index[inner_edge_index[:, 1]] = parent_cell_edges[:, 2]
        child_parent_edge_index[inner_edge_index[:, 2]] = parent_cell_edges[:, 0]
        child_edge_parent_type[inner_edge_index[:, 0]] = EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_0
        child_edge_parent_type[inner_edge_index[:, 1]] = EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_1
        child_edge_parent_type[inner_edge_index[:, 2]] = EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_2

    raw_edge_cells = _edge_cells_from_cell_edges(raw_cell_edges, new_edge_count)
    new_edges = _edge_vertices_from_cell_edges(new_cells, raw_cell_edges, raw_edge_cells)
    new_cells, new_cell_edges = _order_cells_by_edges(
        new_vertices,
        new_cells,
        new_edges,
        raw_cell_edges,
        raw_edge_cells,
        accelerator,
    )

    if child_parent_edge_index.shape[0] != new_edge_count:
        raise RuntimeError("generated child parent edge provenance has unexpected size")

    if child_edge_parent_type.shape[0] != new_edge_count:
        raise RuntimeError("generated child edge parent type provenance has unexpected size")

    parent_vertex_index = np.empty(new_vertices.shape[0], dtype=np.int32)
    parent_vertex_index[:old_vertex_count] = np.arange(
        1,
        old_vertex_count + 1,
        dtype=np.int32,
    )
    parent_vertex_index[old_vertex_count:] = -np.arange(
        1,
        old_edge_count + 1,
        dtype=np.int32,
    )
    parent_cell_index = np.repeat(
        np.arange(1, old_cell_count + 1, dtype=np.int32),
        4,
    )
    parent_cell_type = np.tile(
        np.array(
            [
                CHILD_CELL_TYPE_CENTER,
                CHILD_CELL_TYPE_AT_VERTEX_2,
                CHILD_CELL_TYPE_AT_VERTEX_0,
                CHILD_CELL_TYPE_AT_VERTEX_1,
            ],
            dtype=np.int32,
        ),
        cells.shape[0],
    )
    provenance = BisectionProvenance(
        cells=cells,
        edges=edge_vertices,
        cell_edges=cell_edges,
        parent_vertex_index=parent_vertex_index,
        parent_cell_index=parent_cell_index,
        parent_cell_type=parent_cell_type,
        child_edges=new_edges,
        child_cell_edges=new_cell_edges,
        child_edge_cells=raw_edge_cells,
        child_parent_edge_index=child_parent_edge_index,
        child_edge_parent_type=child_edge_parent_type,
    )
    return (
        _normalize_rows(new_vertices.astype(np.float64, copy=False)),
        new_cells,
        provenance,
    )


def _common_edge_vertex(first_edge: np.ndarray, second_edge: np.ndarray) -> int:
    first_0 = int(first_edge[0])
    first_1 = int(first_edge[1])
    second_0 = int(second_edge[0])
    second_1 = int(second_edge[1])
    if first_0 == second_0 or first_0 == second_1:
        return first_0
    if first_1 == second_0 or first_1 == second_1:
        return first_1
    raise RuntimeError("parent cell edges do not share exactly one vertex")


def _local_vertex_position(cell: np.ndarray, vertex: int) -> int:
    if int(cell[0]) == vertex:
        return 0
    if int(cell[1]) == vertex:
        return 1
    if int(cell[2]) == vertex:
        return 2
    raise RuntimeError("parent vertex is not present exactly once in parent cell")


def _edge_endpoint_slot(edge: np.ndarray, vertex: int) -> int:
    if int(edge[0]) == vertex:
        return 0
    if int(edge[1]) == vertex:
        return 1
    raise RuntimeError("vertex is not an endpoint of parent edge")


def _edge_cells_from_cell_edges(cell_edges: np.ndarray, edge_count: int) -> np.ndarray:
    flat_edges = cell_edges.ravel()
    if flat_edges.size == 0:
        open_edges = np.arange(edge_count, dtype=np.int32)
        if open_edges.size:
            raise RuntimeError(f"edge {int(open_edges[0])} has 1 adjacent cells, expected 2")
        return np.empty((0, 2), dtype=np.int32)

    cell_index = np.repeat(np.arange(cell_edges.shape[0], dtype=np.int32), cell_edges.shape[1])
    sort_order = np.argsort(flat_edges, kind="stable")
    sorted_edges = flat_edges[sort_order]
    sorted_cells = cell_index[sort_order]
    counts = np.bincount(sorted_edges, minlength=edge_count)

    overfull_edges = np.flatnonzero(counts > 2)
    if overfull_edges.size:
        raise RuntimeError(f"edge {int(overfull_edges[0])} has more than two adjacent cells")

    open_edges = np.flatnonzero(counts < 2)
    if open_edges.size:
        raise RuntimeError(f"edge {int(open_edges[0])} has 1 adjacent cells, expected 2")

    starts = np.empty(edge_count, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.cumsum(counts[:-1])
    edge_cells = np.empty((edge_count, 2), dtype=np.int32)
    edge_cells[:, 0] = sorted_cells[starts]
    edge_cells[:, 1] = sorted_cells[starts + 1]
    return edge_cells


def _edge_vertices_from_cell_edges(
    cells: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> np.ndarray:
    edge_count = edge_cells.shape[0]
    edge_index = np.arange(edge_count, dtype=np.int32)
    cell_index = edge_cells[:, 1].copy()
    open_edge = cell_index < 0
    cell_index[open_edge] = edge_cells[open_edge, 0]
    if np.any(cell_index < 0):
        raise RuntimeError("edge is not present exactly once in adjacent cell")

    local_positions = np.empty(edge_count, dtype=np.int8)
    found = np.zeros(edge_count, dtype=bool)
    selected_cell_edges = cell_edges[cell_index]
    for position in range(3):
        matches = selected_cell_edges[:, position] == edge_index
        if np.any(found & matches):
            raise RuntimeError("edge is not present exactly once in adjacent cell")
        local_positions[matches] = position
        found |= matches
    if not np.all(found):
        raise RuntimeError("edge is not present exactly once in adjacent cell")

    next_positions = (local_positions + 1) % 3
    edges = np.empty((edge_count, 2), dtype=np.int32)
    edges[:, 0] = cells[cell_index, local_positions]
    edges[:, 1] = cells[cell_index, next_positions]
    edges[open_edge] = edges[open_edge, ::-1]
    return edges


def _order_cells_by_edges(
    vertices: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
    accelerator: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    cell_centers = _cell_centers(vertices, cells, 1.0)
    edge_centers = _edge_centers(vertices, edges, 1.0)
    edge_system_orientation = _edge_system_orientation(
        vertices,
        cell_centers,
        edges,
        edge_cells,
        edge_centers,
    )
    if _accelerated.should_use_numba_ordering(accelerator, cells.shape[0]):
        ordered_cells, ordered_cell_edges, failure_cell, failure_kind = (
            _accelerated.order_cells_by_edges_numba(
                edges,
                cell_edges,
                edge_cells,
                edge_system_orientation.astype(np.int32, copy=False),
            )
        )
        if failure_kind == 1:
            raise RuntimeError("could not find next cell edge")
        if failure_kind == 2:
            raise RuntimeError("cell edges do not form a closed triangle")
        if failure_cell >= 0:
            raise RuntimeError(f"cell {failure_cell} could not be ordered")
        return ordered_cells, ordered_cell_edges

    ordered_cells = np.empty_like(cells)
    ordered_cell_edges = np.empty_like(cell_edges)
    for cell_index, edges_for_cell in enumerate(cell_edges):
        first_edge = int(edges_for_cell[0])
        start_vertex, next_vertex = map(int, edges[first_edge])
        cell_orientation = 1 if edge_cells[first_edge, 0] == cell_index else -1
        if cell_orientation * int(edge_system_orientation[first_edge]) > 0:
            start_vertex, next_vertex = next_vertex, start_vertex

        ordered_cell_edges[cell_index, 0] = first_edge
        ordered_cells[cell_index, 0] = start_vertex
        current_edge = first_edge
        current_vertex = next_vertex
        previous_edge = -1
        for output_index in range(1, 3):
            edge_index = -1
            following_vertex = -1
            for candidate in map(int, edges_for_cell):
                if candidate == current_edge or candidate == previous_edge:
                    continue
                first, second = map(int, edges[candidate])
                if first == current_vertex:
                    edge_index = candidate
                    following_vertex = second
                    break
                if second == current_vertex:
                    edge_index = candidate
                    following_vertex = first
                    break
            if edge_index < 0:
                raise RuntimeError("could not find next cell edge")
            ordered_cell_edges[cell_index, output_index] = edge_index
            ordered_cells[cell_index, output_index] = current_vertex
            previous_edge = current_edge
            current_edge = edge_index
            current_vertex = following_vertex
        if current_vertex != start_vertex or current_edge < 0:
            raise RuntimeError("cell edges do not form a closed triangle")
    return ordered_cells, ordered_cell_edges


def _check_expected_counts(spec: GlobalGridSpec, vertices: np.ndarray, cells: np.ndarray) -> None:
    if cells.shape[0] != spec.expected_cells:
        raise RuntimeError(f"generated {cells.shape[0]} cells, expected {spec.expected_cells}")
    if vertices.shape[0] != spec.expected_vertices:
        raise RuntimeError(
            f"generated {vertices.shape[0]} vertices, expected {spec.expected_vertices}"
        )


def _lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(points, axis=1)
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2] / radius, -1.0, 1.0)))
    return lon, lat


def _cell_centers(vertices: np.ndarray, cells: np.ndarray, radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    triangles = unit_vertices[cells]
    centers = np.cross(
        triangles[:, 0] - triangles[:, 1],
        triangles[:, 0] - triangles[:, 2],
    )
    centers = _normalize_rows(centers)
    reference = _normalize_rows(triangles.sum(axis=1))
    centers = np.where(np.sum(centers * reference, axis=1)[:, np.newaxis] < 0.0, -centers, centers)
    return centers * radius


def _build_edges(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_count = cells.shape[0]
    edge_lookup: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    edge_cells: list[list[int]] = []
    cell_edges = np.empty((cell_count, 3), dtype=np.int32)
    scan_pairs = ((1, 0, 0), (2, 1, 1), (0, 2, 2))

    for cell_index, cell in enumerate(cells):
        for start, end, local_index in scan_pairs:
            first = int(cell[start])
            second = int(cell[end])
            key = (first, second) if first < second else (second, first)
            edge_index = edge_lookup.get(key)
            if edge_index is None:
                edge_index = len(edges)
                edge_lookup[key] = edge_index
                edges.append((first, second))
                edge_cells.append([cell_index, -1])
            elif edge_cells[edge_index][1] == -1:
                edge_cells[edge_index][1] = cell_index
            else:
                raise RuntimeError(f"edge {edge_index} has more than two adjacent cells")
            cell_edges[cell_index, local_index] = edge_index

    edge_cells_array = np.asarray(edge_cells, dtype=np.int32)
    open_edges = np.flatnonzero(edge_cells_array[:, 1] < 0)
    if open_edges.size:
        bad_edge = int(open_edges[0])
        raise RuntimeError(f"edge {bad_edge} has 1 adjacent cells, expected 2")
    return (
        np.asarray(edges, dtype=np.int32),
        cell_edges,
        edge_cells_array,
    )




























def _zeros_fixed(name: str) -> np.ndarray:
    return np.zeros((1, FIXED_DIMS[name]), dtype=np.int32)


def _start_index_fixed(name: str, size: int) -> np.ndarray:
    values = np.full((1, FIXED_DIMS[name]), size + 1, dtype=np.int32)
    values[:, ACTIVE_REFINEMENT_START[name] :] = 1
    return values


def _end_index_fixed(name: str, size: int) -> np.ndarray:
    values = np.full((1, FIXED_DIMS[name]), size, dtype=np.int32)
    values[:, ACTIVE_REFINEMENT_START[name] :] = 0
    return values


def _fixed_incidence(
    owners: np.ndarray,
    values: np.ndarray,
    row_count: int,
    width: int,
) -> np.ndarray:
    counts = np.bincount(owners, minlength=row_count)
    oversized = np.flatnonzero(counts > width)
    if oversized.size:
        raise RuntimeError(
            f"vertex {int(oversized[0])} has {int(counts[oversized[0]])} incident "
            f"items, expected at most {width}"
        )

    order = np.argsort(owners, kind="stable")
    sorted_owners = owners[order]
    start_by_owner = np.r_[0, np.cumsum(counts[:-1])]
    positions = np.arange(owners.size, dtype=np.int32) - start_by_owner[sorted_owners]
    incidence = np.zeros((row_count, width), dtype=np.int32)
    incidence[sorted_owners, positions] = values[order]
    return incidence


def _sort_fixed_around_vertices(
    vertices: np.ndarray,
    ids: np.ndarray,
    *,
    points: np.ndarray | None = None,
) -> np.ndarray:
    if points is None:
        points = _normalize_rows(vertices)
    origins = _normalize_rows(vertices)
    references = np.tile(np.array([0.0, 0.0, 1.0]), (vertices.shape[0], 1))
    pole_mask = np.abs(origins[:, 2]) > 0.9
    references[pole_mask] = np.array([1.0, 0.0, 0.0])

    axis_1 = references - np.sum(references * origins, axis=1)[:, np.newaxis] * origins
    axis_1 = _normalize_rows(axis_1)
    axis_2 = np.cross(origins, axis_1)

    valid = ids > 0
    safe_ids = np.where(valid, ids, 1)
    point_values = points[safe_ids - 1]
    tangent = point_values - np.sum(
        point_values * origins[:, np.newaxis, :],
        axis=2,
    )[:, :, np.newaxis] * origins[:, np.newaxis, :]
    angles = np.arctan2(
        np.sum(tangent * axis_2[:, np.newaxis, :], axis=2),
        np.sum(tangent * axis_1[:, np.newaxis, :], axis=2),
    )
    angles = np.where(valid, angles, np.inf)
    angle_order = np.argsort(angles, axis=1, kind="stable")
    ordered = np.take_along_axis(ids, angle_order, axis=1)

    counts = valid.sum(axis=1)
    min_position = np.argmin(np.where(ordered > 0, ordered, np.iinfo(np.int32).max), axis=1)
    rotation = (min_position[:, np.newaxis] + np.arange(ids.shape[1])) % np.maximum(
        counts[:, np.newaxis],
        1,
    )
    rotated = np.take_along_axis(ordered, rotation, axis=1)
    return np.where(np.arange(ids.shape[1]) < counts[:, np.newaxis], rotated, 0).astype(
        np.int32,
        copy=False,
    )


def _icon_connectivity(
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> dict[str, np.ndarray]:
    n_vertices = vertices.shape[0]
    c2e = np.asarray(cell_edges, dtype=np.int32)
    adjacent = edge_cells[c2e]
    cell_ids = np.arange(cells.shape[0], dtype=np.int32)[:, np.newaxis]
    first_adjacent = adjacent[:, :, 0] == cell_ids
    c2c = np.where(first_adjacent, adjacent[:, :, 1], adjacent[:, :, 0]).astype(
        np.int32,
        copy=False,
    )
    orientation = np.where(first_adjacent, 1, -1).astype(np.int32, copy=False)

    cell_owners = cells.reshape(-1)
    cell_values = np.repeat(
        np.arange(1, cells.shape[0] + 1, dtype=np.int32),
        3,
    )
    edge_values = np.arange(1, edges.shape[0] + 1, dtype=np.int32)
    edge_owners = np.concatenate((edges[:, 0], edges[:, 1]))
    incident_edges = np.concatenate((edge_values, edge_values))
    incident_vertices = np.concatenate((edges[:, 1] + 1, edges[:, 0] + 1)).astype(
        np.int32,
        copy=False,
    )

    edge_centers = _edge_centers(vertices, edges, 1.0)
    unit_centers = _normalize_rows(cell_center_xyz)

    v2v = _sort_fixed_around_vertices(
        vertices,
        _fixed_incidence(edge_owners, incident_vertices, n_vertices, 6),
    )
    v2e = _sort_fixed_around_vertices(
        vertices,
        _fixed_incidence(edge_owners, incident_edges, n_vertices, 6),
        points=edge_centers,
    )
    v2c = _sort_fixed_around_vertices(
        vertices,
        _fixed_incidence(cell_owners, cell_values, n_vertices, 6),
        points=unit_centers,
    )
    edge_start_vertices = edges[np.maximum(v2e - 1, 0), 0]
    vertex_ids = np.arange(n_vertices, dtype=np.int32)[:, np.newaxis]
    edge_orientation = np.where(edge_start_vertices == vertex_ids, 1, -1).astype(
        np.int32,
        copy=False,
    )
    edge_orientation = np.where(v2e > 0, edge_orientation, 0)

    return {
        "c2e": c2e,
        "c2c": c2c,
        "v2c": v2c,
        "v2e": v2e,
        "v2v": v2v,
        "orientation_of_normal": orientation,
        "edge_orientation": edge_orientation,
    }


def _public_connectivity(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "edge_of_cell": icon_connectivity["c2e"],
        "vertex_of_cell": cells,
        "neighbor_cell_index": icon_connectivity["c2c"],
        "adjacent_cell_of_edge": edge_cells,
        "edge_vertices": edges,
        "cells_of_vertex": _zero_based_with_skip(icon_connectivity["v2c"]),
        "edges_of_vertex": _zero_based_with_skip(icon_connectivity["v2e"]),
        "vertices_of_vertex": _zero_based_with_skip(icon_connectivity["v2v"]),
    }


def _neighbor_tables(
    cells: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "c2e2c": icon_connectivity["c2c"],
        "c2e": icon_connectivity["c2e"],
        "e2c": np.asarray(edge_cells, dtype=np.int32),
        "v2e": _zero_based_with_skip(icon_connectivity["v2e"]),
        "v2c": _zero_based_with_skip(icon_connectivity["v2c"]),
        "c2v": np.asarray(cells, dtype=np.int32),
        "v2e2v": _zero_based_with_skip(icon_connectivity["v2v"]),
        "e2v": np.asarray(edges, dtype=np.int32),
    }


def _geometry_fields(
    vertices: np.ndarray,
    cells: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    icon_connectivity: dict[str, np.ndarray],
    sphere_radius: float,
) -> dict[str, np.ndarray]:
    cell_areas = _cell_areas(vertices, cells, sphere_radius)
    edge_lengths = _edge_lengths(vertices, edges, sphere_radius)
    dual_edge_lengths = _dual_edge_lengths(cell_center_xyz, edge_cells, sphere_radius)
    edge_cell_distance = _edge_cell_distances(
        cell_center_xyz,
        edge_cells,
        edge_center_xyz,
        sphere_radius,
    )
    edge_system_orientation = _edge_system_orientation(
        vertices,
        cell_center_xyz,
        edges,
        edge_cells,
        edge_center_xyz,
    )
    normals = _edge_normal_fields(
        vertices,
        edges,
        edge_center_xyz,
        edge_system_orientation,
    )
    return {
        "cell_area": cell_areas,
        "dual_area": _dual_areas(
            vertices.shape[0],
            cells,
            cell_areas,
            cell_center_xyz,
            icon_connectivity["v2c"],
            sphere_radius,
        ),
        "edge_length": edge_lengths,
        "dual_edge_length": dual_edge_lengths,
        "edge_cell_distance": edge_cell_distance,
        "edge_vert_distance": np.column_stack((edge_lengths * 0.5, edge_lengths * 0.5)),
        "orientation_of_normal": icon_connectivity["orientation_of_normal"],
        "edge_system_orientation": edge_system_orientation,
        "edge_orientation": icon_connectivity["edge_orientation"],
        "edgequad_area": 0.5 * edge_lengths * dual_edge_lengths,
        **normals,
    }


def _edge_system_orientation(
    vertices: np.ndarray,
    cell_center_xyz: np.ndarray,
    edges: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    unit_cells = _normalize_rows(cell_center_xyz)
    unit_edges = _normalize_rows(edge_center_xyz)
    vertex_direction = unit_vertices[edges[:, 1]] - unit_vertices[edges[:, 0]]
    cell_direction = unit_cells[edge_cells[:, 1]] - unit_cells[edge_cells[:, 0]]
    outward_component = np.sum(
        np.cross(vertex_direction, cell_direction) * unit_edges,
        axis=1,
    )
    if np.any(np.isclose(outward_component, 0.0)):
        raise RuntimeError("edge system orientation is degenerate for at least one edge")
    return np.where(outward_component > 0.0, 1, -1).astype(np.int32)


def _edge_normal_fields(
    vertices: np.ndarray,
    edges: np.ndarray,
    edge_center_xyz: np.ndarray,
    edge_system_orientation: np.ndarray,
) -> dict[str, np.ndarray]:
    unit_vertices = _normalize_rows(vertices)
    unit_edges = _normalize_rows(edge_center_xyz)
    tangent = _normalize_rows(
        edge_system_orientation[:, np.newaxis]
        * (unit_vertices[edges[:, 1]] - unit_vertices[edges[:, 0]])
    )
    normal = _normalize_rows(np.cross(unit_edges, tangent))
    primal_u, primal_v = _zonal_meridional_components(unit_edges, normal)
    dual_u, dual_v = _zonal_meridional_components(unit_edges, tangent)
    return {
        "edge_primal_normal_cartesian": normal,
        "edge_dual_normal_cartesian": tangent,
        "zonal_normal_primal_edge": primal_u,
        "meridional_normal_primal_edge": primal_v,
        "zonal_normal_dual_edge": dual_u,
        "meridional_normal_dual_edge": dual_v,
    }


def _zonal_meridional_components(
    points: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unit_points = _normalize_rows(points)
    lon = np.arctan2(unit_points[:, 1], unit_points[:, 0])
    lat = np.arcsin(np.clip(unit_points[:, 2], -1.0, 1.0))
    east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)))
    north = np.column_stack(
        (-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat))
    )
    return np.sum(vectors * east, axis=1), np.sum(vectors * north, axis=1)


def _refinement_fields(
    spec: GlobalGridSpec,
    options: IconGridOptions,
    geometry: GeometryData,
    edges: np.ndarray,
    context: _GlobalGenerationContext | None = None,
) -> dict[str, np.ndarray]:
    """Return ICON refinement-control and parent-provenance fields.

    For bisection-refined grids, fine vertices either coincide with a parent
    vertex or with the midpoint of a parent edge. ICON encodes those two cases
    in one field: positive values are one-based parent vertex IDs, and negative
    values are one-based parent edge IDs with a minus sign.
    """
    vertices = geometry.vertices
    cells = geometry.cells
    refinement = {
        "refin_c_ctrl": np.full(cells.shape[0], -4, dtype=np.int32),
        "refin_e_ctrl": np.full(edges.shape[0], -8, dtype=np.int32),
        "refin_v_ctrl": np.zeros(vertices.shape[0], dtype=np.int32),
        "start_idx_c": _start_index_fixed("cell_grf", cells.shape[0]),
        "end_idx_c": _end_index_fixed("cell_grf", cells.shape[0]),
        "start_idx_e": _start_index_fixed("edge_grf", edges.shape[0]),
        "end_idx_e": _end_index_fixed("edge_grf", edges.shape[0]),
        "start_idx_v": _start_index_fixed("vert_grf", vertices.shape[0]),
        "end_idx_v": _end_index_fixed("vert_grf", vertices.shape[0]),
        "parent_cell_index": np.zeros(cells.shape[0], dtype=np.int32),
        "parent_cell_type": np.zeros(cells.shape[0], dtype=np.int32),
        "edge_parent_type": np.zeros(edges.shape[0], dtype=np.int32),
        "parent_edge_index": np.zeros(edges.shape[0], dtype=np.int32),
        "parent_vertex_index": np.zeros(vertices.shape[0], dtype=np.int32),
    }
    if spec.bisections == 0:
        return refinement

    parent = geometry.bisection_provenance
    if parent is None:
        if context is None:
            context = _GlobalGenerationContext()
        parent_vertex_index, parent = _parent_vertex_indices_cached(
            spec,
            options,
            vertices,
            context,
        )
        parent_cell_index, parent_cell_type = _parent_cell_fields(
            cells,
            parent_vertex_index,
            parent,
            options.accelerator,
        )
    else:
        parent_vertex_index = parent.parent_vertex_index
        parent_cell_index = parent.parent_cell_index
        parent_cell_type = parent.parent_cell_type

    refinement["parent_vertex_index"] = parent_vertex_index
    refinement["parent_cell_index"] = parent_cell_index
    refinement["parent_cell_type"] = parent_cell_type
    if (
        isinstance(parent, BisectionProvenance)
        and parent.child_parent_edge_index is not None
        and parent.child_edge_parent_type is not None
    ):
        refinement["parent_edge_index"] = parent.child_parent_edge_index
        refinement["edge_parent_type"] = parent.child_edge_parent_type
    else:
        refinement["parent_edge_index"], refinement["edge_parent_type"] = (
            _parent_edge_fields(edges, parent_vertex_index, parent, options.accelerator)
        )
    return refinement


def _parent_vertex_indices(
    vertices: np.ndarray,
    parent: IconGrid | _GlobalParentData,
) -> np.ndarray:
    lookup: dict[tuple[float, float, float], int] = {}
    for vertex_index, point in enumerate(_normalize_rows(parent.vertices)):
        lookup[_point_key(point)] = vertex_index + 1
    for edge_index, point in enumerate(_normalize_rows(parent.edge_center_xyz)):
        lookup[_point_key(point)] = -(edge_index + 1)

    parent_index = np.empty(vertices.shape[0], dtype=np.int32)
    for vertex_index, point in enumerate(_normalize_rows(vertices)):
        value = lookup.get(_point_key(point))
        if value is None:
            raise RuntimeError(f"vertex {vertex_index} has no parent vertex or edge")
        parent_index[vertex_index] = value
    return parent_index


def _point_key(point: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.round(point.astype(np.float64), decimals=POINT_MATCH_DECIMALS))


def _lookup_parent_signatures(
    signature_keys: np.ndarray,
    parent_index_values: np.ndarray,
    type_values: np.ndarray,
    query_keys: np.ndarray,
    accelerator: str,
    item_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(tuple(signature_keys[:, column] for column in range(signature_keys.shape[1] - 1, -1, -1)))
    sorted_keys = np.ascontiguousarray(signature_keys[order])
    sorted_parent_index = np.ascontiguousarray(parent_index_values[order])
    sorted_type = np.ascontiguousarray(type_values[order])

    if _accelerated.should_use_numba(accelerator, query_keys.shape[0]):
        if sorted_keys.shape[1] == 2:
            parent_index, parent_type = _accelerated.lookup_width2_numba(
                sorted_keys,
                sorted_parent_index,
                sorted_type,
                np.ascontiguousarray(query_keys),
            )
        else:
            parent_index, parent_type = _accelerated.lookup_width3_numba(
                sorted_keys,
                sorted_parent_index,
                sorted_type,
                np.ascontiguousarray(query_keys),
            )
    else:
        parent_index, parent_type = _lookup_parent_signatures_numpy(
            sorted_keys,
            sorted_parent_index,
            sorted_type,
            query_keys,
        )

    missing = np.flatnonzero(parent_index == 0)
    if missing.size:
        raise RuntimeError(f"{item_name} {int(missing[0])} has no parent {item_name}")
    return parent_index, parent_type


def _lookup_parent_signatures_numpy(
    signature_keys: np.ndarray,
    parent_index_values: np.ndarray,
    type_values: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signature_view = _row_view(signature_keys)
    query_view = _row_view(query_keys)
    positions = np.searchsorted(signature_view, query_view)
    valid = positions < signature_view.shape[0]
    found = np.zeros(query_view.shape[0], dtype=bool)
    found[valid] = signature_view[positions[valid]] == query_view[valid]
    parent_index = np.zeros(query_view.shape[0], dtype=np.int32)
    parent_type = np.zeros(query_view.shape[0], dtype=np.int32)
    parent_index[found] = parent_index_values[positions[found]]
    parent_type[found] = type_values[positions[found]]
    return parent_index, parent_type


def _row_view(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    dtype = np.dtype(
        {
            "names": [f"f{index}" for index in range(contiguous.shape[1])],
            "formats": [contiguous.dtype] * contiguous.shape[1],
        }
    )
    return contiguous.view(dtype).reshape(-1)


def _parent_cell_fields(
    cells: np.ndarray,
    parent_vertex_index: np.ndarray,
    parent: IconGrid | _GlobalParentData | BisectionProvenance,
    accelerator: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Map each fine cell to its parent cell and ICON child-cell type code."""
    parent_cells = parent.cells.astype(np.int64, copy=False) + 1
    parent_edges = parent.cell_edges.astype(np.int64, copy=False) + 1
    a = parent_cells[:, 0]
    b = parent_cells[:, 1]
    c = parent_cells[:, 2]
    e_ab = parent_edges[:, 0]
    e_bc = parent_edges[:, 1]
    e_ca = parent_edges[:, 2]

    signature_keys = np.empty((parent.cells.shape[0] * 4, 3), dtype=np.int64)
    signature_keys[0::4] = np.column_stack((a, -e_ab, -e_ca))
    signature_keys[1::4] = np.column_stack((b, -e_ab, -e_bc))
    signature_keys[2::4] = np.column_stack((c, -e_ca, -e_bc))
    signature_keys[3::4] = np.column_stack((-e_ab, -e_bc, -e_ca))
    signature_keys.sort(axis=1)

    parent_indices = np.repeat(
        np.arange(1, parent.cells.shape[0] + 1, dtype=np.int32),
        4,
    )
    child_types = np.tile(
        np.array(
            [
                CHILD_CELL_TYPE_AT_VERTEX_0,
                CHILD_CELL_TYPE_AT_VERTEX_1,
                CHILD_CELL_TYPE_AT_VERTEX_2,
                CHILD_CELL_TYPE_CENTER,
            ],
            dtype=np.int32,
        ),
        parent.cells.shape[0],
    )
    query_keys = np.sort(parent_vertex_index[cells].astype(np.int64), axis=1)
    return _lookup_parent_signatures(
        signature_keys,
        parent_indices,
        child_types,
        query_keys,
        accelerator,
        "cell",
    )


def _parent_edge_fields(
    edges: np.ndarray,
    parent_vertex_index: np.ndarray,
    parent: IconGrid | _GlobalParentData | BisectionProvenance,
    accelerator: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Map each fine edge to its parent edge and ICON child-edge type code."""
    parent_edges = parent.edges.astype(np.int64, copy=False) + 1
    edge_ids = np.arange(1, parent.edges.shape[0] + 1, dtype=np.int64)
    midpoints = -edge_ids

    parent_cell_edges = parent.cell_edges.astype(np.int64, copy=False) + 1
    e_ab = parent_cell_edges[:, 0]
    e_bc = parent_cell_edges[:, 1]
    e_ca = parent_cell_edges[:, 2]

    edge_signature_count = parent.edges.shape[0] * 2
    cell_signature_count = parent.cells.shape[0] * 3
    signature_keys = np.empty(
        (edge_signature_count + cell_signature_count, 2),
        dtype=np.int64,
    )
    parent_indices = np.empty(signature_keys.shape[0], dtype=np.int32)
    edge_types = np.empty(signature_keys.shape[0], dtype=np.int32)

    signature_keys[0:edge_signature_count:2] = np.column_stack(
        (parent_edges[:, 0], midpoints)
    )
    signature_keys[1:edge_signature_count:2] = np.column_stack(
        (parent_edges[:, 1], midpoints)
    )
    parent_indices[:edge_signature_count] = np.repeat(edge_ids.astype(np.int32), 2)
    edge_types[0:edge_signature_count:2] = EDGE_CHILD_TYPE_FROM_VERTEX_0
    edge_types[1:edge_signature_count:2] = EDGE_CHILD_TYPE_FROM_VERTEX_1

    offset = edge_signature_count
    signature_keys[offset + 0 :: 3] = np.column_stack((-e_ab, -e_ca))
    signature_keys[offset + 1 :: 3] = np.column_stack((-e_ab, -e_bc))
    signature_keys[offset + 2 :: 3] = np.column_stack((-e_ca, -e_bc))
    parent_indices[offset + 0 :: 3] = e_bc.astype(np.int32)
    parent_indices[offset + 1 :: 3] = e_ca.astype(np.int32)
    parent_indices[offset + 2 :: 3] = e_ab.astype(np.int32)
    edge_types[offset + 0 :: 3] = EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_0
    edge_types[offset + 1 :: 3] = EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_1
    edge_types[offset + 2 :: 3] = EDGE_CHILD_TYPE_IN_CELL_OPPOSITE_VERTEX_2

    signature_keys.sort(axis=1)
    query_keys = np.sort(parent_vertex_index[edges].astype(np.int64), axis=1)
    return _lookup_parent_signatures(
        signature_keys,
        parent_indices,
        edge_types,
        query_keys,
        accelerator,
        "edge",
    )


def _metadata(
    spec: Any,
    options: IconGridOptions,
    geometry: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "uuidOfHGrid": _spec_uuid(spec, options),
        "uuidOfParHGrid": "00000000-0000-0000-0000-000000000000",
        "grid_root": getattr(spec, "root", 0),
        "grid_level": getattr(spec, "bisections", 0),
        "sphere_radius": options.sphere_radius,
        "grid_geometry": 1,
        "grid_cell_type": 3,
        "number_of_grid_used": 1,
        "center": 255,
        "subcenter": 255,
        "centre": 255,
        "subcentre": 255,
        "crs_id": 0,
        "crs_name": "Spherical Earth",
        "grid_mapping_name": "latitude_longitude",
        "ellipsoid_name": "sphere",
        "semi_major_axis": options.sphere_radius,
        "inverse_flattening": 0.0,
    }
    if isinstance(spec, GlobalGridSpec):
        metadata.update(
            {
                "centre": options.global_grid.centre,
                "subcentre": options.global_grid.subcentre,
                "center": options.global_grid.centre,
                "subcenter": options.global_grid.subcentre,
                "number_of_grid_used": options.global_grid.number_of_grid_used,
                "spring_beta": options.global_grid.beta_spring,
                "spring_maxit": options.global_grid.maxit,
                "indexing_algorithm": options.global_grid.indexing_algorithm,
                "grid_mapping_name": "lat_long_on_sphere",
                "global_grid": 1,
            }
        )
        metadata["global_optimization"] = options.global_optimization.method
        if options.global_optimization.method != "none":
            metadata.update(
                {
                    "global_optimization_iterations": options.global_optimization.iterations,
                }
            )
    if isinstance(spec, TorusGridSpec):
        metadata.update(
            {
                "grid_geometry": 2,
                "periodic": 1,
                "crs_name": "Planar torus",
                "grid_mapping_name": "cartesian",
                "domain_length": spec.domain_length,
                "domain_height": spec.domain_height,
                "torus_nx": spec.nx,
                "torus_ny": spec.ny,
                "torus_edge_length": spec.edge_length,
            }
        )
    elif isinstance(spec, _PLANAR_GRID_SPEC_TYPES):
        metadata.update(
            {
                "grid_geometry": 2,
                "periodic": int(getattr(spec, "periodic", False)),
                "crs_name": "Planar",
                "grid_mapping_name": "cartesian",
                "planar_grid_type": spec.__class__.__name__,
                "planar_nx": spec.nx,
                "planar_ny": spec.ny,
            }
        )
        if hasattr(spec, "domain_length"):
            metadata["domain_length"] = spec.domain_length
            metadata["domain_height"] = spec.domain_height
        elif hasattr(spec, "edge_length"):
            metadata["planar_edge_length"] = spec.edge_length
        else:
            metadata["planar_dx"] = spec.dx
            metadata["planar_dy"] = spec.dy
    elif isinstance(spec, LimitedAreaGridSpec):
        metadata.update(
            {
                "grid_geometry": 3,
                "parent_grid_name": spec.parent_grid_name,
                "limited_area_region": _region_class_name(spec.region),
                "boundary_depth_index": spec.boundary_depth,
            }
        )
    elif isinstance(spec, CutGridSpec):
        metadata.update(
            {
                "grid_geometry": 3,
                "grid_mapping_name": "cartesian",
                "cut_mode": spec.mode,
                "boundary_depth_index": spec.boundary_depth,
                "smoothing_depth": spec.smoothing_depth,
                "cut_region_count": len(spec.regions),
            }
        )
    if geometry:
        metadata.update(
            {
                "mean_edge_length": float(np.mean(geometry["edge_length"])),
                "mean_dual_edge_length": float(np.mean(geometry["dual_edge_length"])),
                "mean_cell_area": float(np.mean(geometry["cell_area"])),
                "mean_dual_cell_area": float(np.mean(geometry["dual_area"])),
            }
        )
    return metadata


def _spec_uuid(
    spec: Any,
    options: IconGridOptions,
) -> str:
    if isinstance(spec, GlobalGridSpec):
        payload = {
            "generator": "grid_generator",
            "grid": spec.name,
            "sphere_radius": _canonical_float(options.sphere_radius),
            "global_grid": _canonicalize_payload(asdict(options.global_grid)),
            "global_optimization": _canonicalize_payload(
                asdict(options.global_optimization)
            ),
        }
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
        )
    payload: dict[str, Any] = {
        "generator": "grid_generator",
        "grid": spec.name,
        "sphere_radius": _canonical_float(options.sphere_radius),
    }
    if isinstance(spec, TorusGridSpec):
        payload.update(
            {
                "family": "torus",
                "nx": spec.nx,
                "ny": spec.ny,
                "edge_length": _canonical_float(spec.edge_length),
            }
        )
    elif isinstance(spec, LimitedAreaGridSpec):
        payload.update(
            {
                "family": "limited_area",
                "parent_grid_name": spec.parent_grid_name,
                "region": _canonicalize_payload(asdict(spec)["region"]),
                "boundary_depth": spec.boundary_depth,
            }
        )
    elif isinstance(spec, _PLANAR_GRID_SPEC_TYPES):
        spec_payload = asdict(spec)
        spec_payload.pop("periodic", None)
        payload.update(
            {
                "family": "planar",
                "kind": spec.__class__.__name__,
                "parameters": _canonicalize_payload(spec_payload),
            }
        )
    elif isinstance(spec, CutGridSpec):
        payload.update(
            {
                "family": "cut",
                "mode": spec.mode,
                "boundary_depth": spec.boundary_depth,
                "smoothing_depth": spec.smoothing_depth,
                "regions": _canonicalize_payload(asdict(spec)["regions"]),
            }
        )
    else:
        payload.update({"family": "unknown"})
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    )


def _canonicalize_payload(value: Any) -> Any:
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, dict):
        return {key: _canonicalize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_payload(item) for item in value]
    return value


def grid_uuid(
    grid_name: str,
    *,
    sphere_radius: float = EARTH_RADIUS_M,
    options: IconGridOptions | Mapping[str, Any] | None = None,
) -> str:
    canonical_sphere_radius = finite_float_option("sphere_radius", sphere_radius)
    if canonical_sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be positive")
    grid_options = _resolve_options(options)
    if sphere_radius != EARTH_RADIUS_M:
        grid_options = replace(grid_options, sphere_radius=canonical_sphere_radius)
    payload = {
        "generator": "grid_generator",
        "grid": parse_grid_spec(grid_name).name,
        "sphere_radius": _canonical_float(canonical_sphere_radius),
        "global_grid": _canonicalize_payload(asdict(grid_options.global_grid)),
        "global_optimization": _canonicalize_payload(asdict(grid_options.global_optimization)),
    }
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    )


def _canonical_float(value: float) -> float:
    return float(f"{float(value):.17g}")


def _sort_around_vertex(
    vertices: np.ndarray,
    vertex: int,
    ids: list[int],
    *,
    points: np.ndarray | None = None,
) -> list[int]:
    if points is None:
        points = _normalize_rows(vertices)
    origin = _normalize(vertices[vertex])
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(origin, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    axis_1 = reference - np.dot(reference, origin) * origin
    axis_1 = axis_1 / np.linalg.norm(axis_1)
    axis_2 = np.cross(origin, axis_1)

    def angle(one_based_id: int) -> float:
        point = points[one_based_id - 1]
        tangent = point - np.dot(point, origin) * origin
        return float(np.arctan2(np.dot(tangent, axis_2), np.dot(tangent, axis_1)))

    ordered = sorted(ids, key=angle)
    if not ordered:
        return ordered
    start = ordered.index(min(ordered))
    return ordered[start:] + ordered[:start]


def _edge_centers(vertices: np.ndarray, edges: np.ndarray, radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    centers = unit_vertices[edges].mean(axis=1)
    return _normalize_rows(centers) * radius


def _cell_areas(vertices: np.ndarray, cells: np.ndarray, sphere_radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    triangles = unit_vertices[cells]
    angles = np.empty((triangles.shape[0], 3), dtype=np.float64)
    for index in range(3):
        a = triangles[:, index]
        b = triangles[:, (index + 1) % 3]
        c = triangles[:, (index + 2) % 3]
        normal_b = _normalize_rows(np.cross(a, b))
        normal_c = _normalize_rows(np.cross(a, c))
        angles[:, index] = np.arccos(np.clip(np.sum(normal_b * normal_c, axis=1), -1.0, 1.0))
    excess = angles.sum(axis=1) - np.pi
    return excess * sphere_radius**2


def _dual_areas(
    n_vertices: int,
    cells: np.ndarray,
    cell_areas: np.ndarray,
    cell_center_xyz: np.ndarray | None = None,
    ordered_cells_of_vertex: np.ndarray | None = None,
    sphere_radius: float | None = None,
) -> np.ndarray:
    if cell_center_xyz is not None and ordered_cells_of_vertex is not None and sphere_radius is not None:
        dual = _geometric_dual_areas(n_vertices, cell_center_xyz, ordered_cells_of_vertex, sphere_radius)
        dual_sum = float(dual.sum())
        if dual_sum > 0.0:
            dual *= float(cell_areas.sum()) / dual_sum
        return dual

    dual = np.zeros(n_vertices, dtype=np.float64)
    for cell_index, cell in enumerate(cells):
        dual[cell] += cell_areas[cell_index] / 3.0
    return dual


def _geometric_dual_areas(
    n_vertices: int,
    cell_center_xyz: np.ndarray,
    ordered_cells_of_vertex: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    unit_centers = _normalize_rows(cell_center_xyz)
    dual = np.zeros(n_vertices, dtype=np.float64)
    valid = ordered_cells_of_vertex > 0
    counts = valid.sum(axis=1)
    for count in np.unique(counts):
        if count < 3:
            continue
        rows = np.flatnonzero(counts == count)
        cell_indices = ordered_cells_of_vertex[rows, :count] - 1
        polygons = unit_centers[cell_indices]
        anchors = np.repeat(polygons[:, :1, :], count - 2, axis=1)
        area = _spherical_triangle_areas(
            anchors.reshape(-1, 3),
            polygons[:, 1:-1, :].reshape(-1, 3),
            polygons[:, 2:, :].reshape(-1, 3),
        ).reshape(rows.size, count - 2)
        dual[rows] = area.sum(axis=1) * sphere_radius**2
    return dual


def _spherical_triangle_areas(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    normal_ab = _normalize_rows(np.cross(a, b))
    normal_ac = _normalize_rows(np.cross(a, c))
    normal_ba = -normal_ab
    normal_bc = _normalize_rows(np.cross(b, c))
    normal_ca = -normal_ac
    normal_cb = -normal_bc
    angles = np.column_stack(
        (
            np.arccos(np.clip(np.sum(normal_ab * normal_ac, axis=1), -1.0, 1.0)),
            np.arccos(np.clip(np.sum(normal_ba * normal_bc, axis=1), -1.0, 1.0)),
            np.arccos(np.clip(np.sum(normal_ca * normal_cb, axis=1), -1.0, 1.0)),
        )
    )
    return angles.sum(axis=1) - np.pi


def _spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(_spherical_triangle_areas(a[np.newaxis, :], b[np.newaxis, :], c[np.newaxis, :])[0])


def _edge_lengths(vertices: np.ndarray, edges: np.ndarray, sphere_radius: float) -> np.ndarray:
    unit_vertices = _normalize_rows(vertices)
    edge_vertices = unit_vertices[edges]
    angles = np.arccos(
        np.clip(np.sum(edge_vertices[:, 0] * edge_vertices[:, 1], axis=1), -1.0, 1.0)
    )
    return angles * sphere_radius


def _dual_edge_lengths(
    cell_center_xyz: np.ndarray,
    edge_cells: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    centers = _normalize_rows(cell_center_xyz)
    adjacent_centers = centers[edge_cells]
    angles = np.arccos(
        np.clip(np.sum(adjacent_centers[:, 0] * adjacent_centers[:, 1], axis=1), -1.0, 1.0)
    )
    return angles * sphere_radius


def _edge_cell_distances(
    cell_center_xyz: np.ndarray,
    edge_cells: np.ndarray,
    edge_center_xyz: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    edge_centers = _normalize_rows(edge_center_xyz)
    cell_centers = _normalize_rows(cell_center_xyz)
    adjacent_centers = cell_centers[edge_cells]
    dots = np.sum(adjacent_centers * edge_centers[:, np.newaxis, :], axis=2)
    return np.arccos(np.clip(dots, -1.0, 1.0)) * sphere_radius


def _zero_based_with_skip(one_based: np.ndarray) -> np.ndarray:
    return np.where(one_based == 0, -1, one_based - 1).astype(np.int32)
