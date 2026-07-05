"""Pure Python ICON-style geodesic grid generation.

The generator accepts ICON R<n>B<k> grid names and canonicalizes them to the
zero-padded form commonly used in ICON grid file names. It creates triangular
spherical grids with the topology, metric, orientation, normal-vector, and
refinement-provenance fields needed to write a compact ICON grid NetCDF file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import getpass
from pathlib import Path
import platform
from typing import Any, Mapping
import re
import json
import uuid

import numpy as np

from ._geometry import SphericalIcosahedralGeometry
from ._io import IconNetcdfWriter
from ._limited_area import LimitedAreaExtractor
from ._metrics import SphericalMetricsBuilder
from ._optimization import (
    GlobalOptimizationOptions,
    GlobalGridOptions,
    optimize_global_grid,
    resolve_global_optimization_options,
)
from ._ordering import IconOrderingBuilder
from ._planar import (
    PlanarRefinementBuilder,
    PlanarTriangularGeometry,
    PlanarTriangularMetricsBuilder,
    PlanarTriangularTopologyBuilder,
)
from ._refinement import GlobalRefinementBuilder
from ._topology import GlobalTopologyBuilder
from ._torus import (
    PeriodicTopologyBuilder,
    PlanarTorusGeometry,
    PlanarTorusMetricsBuilder,
    TorusRefinementBuilder,
)
from ._types import BisectionProvenance, GeometryData, TopologyData
from ._validation import finite_float_option, validate_grid_options
from . import _accelerated

IconNetcdfField = tuple[str, tuple[str, ...], Any, dict[str, Any]]


@dataclass
class _GlobalGenerationContext:
    """Shared internal state for one global generation request."""

    grids: dict[tuple[int, int], "IconGrid"] = field(default_factory=dict)
    parent_data: dict[tuple[int, int], "_GlobalParentData"] = field(default_factory=dict)
    parent_vertex_indices: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def key(self, spec: "GlobalGridSpec") -> tuple[int, int]:
        return spec.root, spec.bisections


@dataclass(frozen=True)
class _GlobalParentData:
    """Geometry and topology needed for child ordering and provenance."""

    spec: "GlobalGridSpec"
    vertices: np.ndarray
    cells: np.ndarray
    edges: np.ndarray
    cell_edges: np.ndarray
    edge_center_xyz: np.ndarray

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
CELL_COORD_ATTRS = {
    "coordinates": "clon clat",
    "grid_type": "unstructured",
    "number_of_grid_in_reference": 1,
}
EDGE_COORD_ATTRS = {"coordinates": "elon elat"}
VERTEX_COORD_ATTRS = {"coordinates": "vlon vlat"}
ICON_VARIABLE_ATTRS: dict[str, dict[str, Any]] = {
    "clon": {
        "bounds": "clon_vertices",
        "long_name": "center longitude",
        "standard_name": "grid_longitude",
    },
    "clat": {
        "bounds": "clat_vertices",
        "long_name": "center latitude",
        "standard_name": "grid_latitude",
    },
    "vlon": {"long_name": "vertex longitude", "standard_name": "grid_longitude"},
    "vlat": {"long_name": "vertex latitude", "standard_name": "grid_latitude"},
    "elon": {
        "bounds": "elon_vertices",
        "long_name": "edge midpoint longitude",
        "standard_name": "grid_longitude",
    },
    "elat": {
        "bounds": "elat_vertices",
        "long_name": "edge midpoint latitude",
        "standard_name": "grid_latitude",
    },
    "lon_cell_centre": {**CELL_COORD_ATTRS, "long_name": "longitude of cell centre"},
    "lat_cell_centre": {**CELL_COORD_ATTRS, "long_name": "latitude of cell centre"},
    "longitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "longitude of vertices"},
    "latitude_vertices": {**VERTEX_COORD_ATTRS, "long_name": "latitude of vertices"},
    "lon_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "longitudes of edge midpoints"},
    "lat_edge_centre": {**EDGE_COORD_ATTRS, "long_name": "latitudes of edge midpoints"},
    "edge_of_cell": {"long_name": "edges of each cell"},
    "vertex_of_cell": {"long_name": "vertices of each cell"},
    "neighbor_cell_index": {"long_name": "cell neighbor index"},
    "adjacent_cell_of_edge": {"long_name": "cells adjacent to each edge"},
    "edge_vertices": {"long_name": "vertices at the end of each edge"},
    "cells_of_vertex": {"long_name": "cells around each vertex"},
    "edges_of_vertex": {"long_name": "edges around each vertex"},
    "vertices_of_vertex": {"long_name": "vertices around each vertex"},
    "cell_area": {
        **CELL_COORD_ATTRS,
        "long_name": "area of grid cell",
        "standard_name": "area",
    },
    "dual_area": {
        **VERTEX_COORD_ATTRS,
        "long_name": "areas of dual hexagonal/pentagonal cells",
        "standard_name": "area",
    },
    "cell_area_p": {**CELL_COORD_ATTRS, "long_name": "area of grid cell"},
    "dual_area_p": {"long_name": "areas of dual hexagonal/pentagonal cells"},
    "edge_length": {**EDGE_COORD_ATTRS, "long_name": "lengths of edges of triangular cells"},
    "dual_edge_length": {
        **EDGE_COORD_ATTRS,
        "long_name": "lengths of dual edges (distances between triangular cell circumcenters)",
    },
    "edge_cell_distance": {
        "long_name": "distances between edge midpoint and adjacent triangle midpoints",
    },
    "edge_vert_distance": {
        "long_name": "distances between edge midpoint and vertices of that edge",
    },
    "edgequad_area": {
        **EDGE_COORD_ATTRS,
        "long_name": "area around the edge formed by the two adjacent triangles",
    },
    "orientation_of_normal": {"long_name": "orientations of normals to triangular cell edges"},
    "edge_system_orientation": {**EDGE_COORD_ATTRS, "long_name": "edge system orientation"},
    "edge_orientation": {"long_name": "edge orientation"},
    "refin_c_ctrl": {"long_name": "refinement control flag for cells"},
    "refin_e_ctrl": {"long_name": "refinement control flag for edges"},
    "refin_v_ctrl": {"long_name": "refinement control flag for vertices"},
    "start_idx_c": {"long_name": "list of start indices for each refinement control level for cells"},
    "end_idx_c": {"long_name": "list of end indices for each refinement control level for cells"},
    "start_idx_e": {"long_name": "list of start indices for each refinement control level for edges"},
    "end_idx_e": {"long_name": "list of end indices for each refinement control level for edges"},
    "start_idx_v": {"long_name": "list of start indices for each refinement control level for vertices"},
    "end_idx_v": {"long_name": "list of end indices for each refinement control level for vertices"},
    "cell_elevation": {**CELL_COORD_ATTRS, "long_name": "elevation at the cell centers"},
    "edge_elevation": {**EDGE_COORD_ATTRS, "long_name": "elevation at the edge centers"},
    "cell_sea_land_mask": {
        **CELL_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "edge_sea_land_mask": {
        **EDGE_COORD_ATTRS,
        "long_name": "sea (-2 inner, -1 boundary) land (2 inner, 1 boundary) mask for the cell",
        "units": "2,1,-1,-",
    },
    "cartesian_x_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate x on unit sp",
    },
    "cartesian_y_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate y on unit sp",
    },
    "cartesian_z_vertices": {
        **VERTEX_COORD_ATTRS,
        "long_name": "vertex cartesian coordinate z on unit sp",
    },
    "cell_circumcenter_cartesian_x": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate x",
    },
    "cell_circumcenter_cartesian_y": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate y",
    },
    "cell_circumcenter_cartesian_z": {
        **CELL_COORD_ATTRS,
        "long_name": "cartesian position of the prime cell circumcenter on the unit sphere, coordinate z",
    },
    "edge_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate x on unit sphere",
    },
    "edge_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate y on unit sphere",
    },
    "edge_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "prime edge center cartesian coordinate z on unit sphere",
    },
    "phys_cell_id": {**CELL_COORD_ATTRS, "long_name": "physical domain ID of cell"},
    "phys_edge_id": {**EDGE_COORD_ATTRS, "long_name": "physical domain ID of edge"},
    "cell_index": {"long_name": "cell index"},
    "edge_index": {"long_name": "edge index"},
    "vertex_index": {"long_name": "vertices index"},
    "edge_dual_middle_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate x on unit sphere",
    },
    "edge_dual_middle_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate y on unit sphere",
    },
    "edge_dual_middle_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "dual edge center cartesian coordinate z on unit sphere",
    },
    "edge_primal_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate x",
    },
    "edge_primal_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate y",
    },
    "edge_primal_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the prime edge 3D vector, coordinate z",
    },
    "edge_dual_normal_cartesian_x": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate x",
    },
    "edge_dual_normal_cartesian_y": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate y",
    },
    "edge_dual_normal_cartesian_z": {
        **EDGE_COORD_ATTRS,
        "long_name": "unit normal to the dual edge 3D vector, coordinate z",
    },
    "zonal_normal_primal_edge": {"long_name": "zonal component of normal to primal edge"},
    "meridional_normal_primal_edge": {
        "long_name": "meridional component of normal to primal edge",
    },
    "zonal_normal_dual_edge": {"long_name": "zonal component of normal to dual edge"},
    "meridional_normal_dual_edge": {
        "long_name": "meridional component of normal to dual edge",
    },
    "parent_cell_index": {**CELL_COORD_ATTRS, "long_name": "parent cell index"},
    "parent_cell_type": {"long_name": "parent cell type"},
    "edge_parent_type": {"long_name": "edge parent type"},
    "parent_edge_index": {"long_name": "parent edge index"},
    "parent_vertex_index": {"long_name": "parent vertex index"},
    "child_cell_index": {"long_name": "child cell index"},
    "child_cell_id": {"long_name": "domain ID of child cell"},
    "child_edge_index": {"long_name": "child edge index"},
    "child_edge_id": {"long_name": "domain ID of child edge"},
}


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

    parent_grid_name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    boundary_depth: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        parent = parse_grid_spec(self.parent_grid_name)
        lon_min = _finite_float_option("lon_min", self.lon_min)
        lon_max = _finite_float_option("lon_max", self.lon_max)
        lat_min = _finite_float_option("lat_min", self.lat_min)
        lat_max = _finite_float_option("lat_max", self.lat_max)
        if not -180.0 <= lon_min <= 180.0:
            raise ValueError("lon_min must be within [-180, 180]")
        if not -180.0 <= lon_max <= 180.0:
            raise ValueError("lon_max must be within [-180, 180]")
        if not -90.0 <= lat_min <= 90.0 or not -90.0 <= lat_max <= 90.0:
            raise ValueError("lat bounds must be within [-90, 90]")
        if lat_min > lat_max:
            raise ValueError("lat_min must be less than or equal to lat_max")
        if not isinstance(self.boundary_depth, int) or isinstance(self.boundary_depth, bool):
            raise TypeError("boundary_depth must be a non-negative integer")
        if self.boundary_depth < 0:
            raise ValueError("boundary_depth must be non-negative")
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
class LonLatBoxRegion:
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
class CircleRegion:
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
class OrientedRectangleRegion:
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
class PolygonRegion:
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


@dataclass(frozen=True)
class CutGridSpec:
    """Selection options for extracting a cut grid from an existing grid."""

    regions: tuple[LonLatBoxRegion | CircleRegion | OrientedRectangleRegion | PolygonRegion, ...]
    mode: str = "keep"
    boundary_depth: int = 0
    smoothing_depth: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions:
            raise ValueError("cut grid spec requires at least one region")
        supported_region_types = (
            LonLatBoxRegion,
            CircleRegion,
            OrientedRectangleRegion,
            PolygonRegion,
        )
        if not all(isinstance(region, supported_region_types) for region in regions):
            raise TypeError("cut grid regions must be supported region spec instances")
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
    global_grid: GlobalGridOptions = field(default_factory=GlobalGridOptions)
    global_optimization: GlobalOptimizationOptions = field(
        default_factory=lambda: GlobalOptimizationOptions(
            method="spring",
            iterations=2000,
        )
    )
    radius: float = 1.0
    sphere_radius: float = EARTH_RADIUS_M

    def __post_init__(self) -> None:
        global_grid = self.global_grid
        if isinstance(global_grid, Mapping):
            global_grid = GlobalGridOptions(**dict(global_grid))
        if not isinstance(global_grid, GlobalGridOptions):
            raise TypeError("global_grid must be a GlobalGridOptions instance or mapping")
        object.__setattr__(
            self,
            "global_grid",
            global_grid,
        )
        global_optimization = resolve_global_optimization_options(self.global_optimization)
        if (
            global_optimization.method == "spring"
            and global_optimization.iterations == 2000
        ):
            global_optimization = GlobalOptimizationOptions(
                method="spring",
                iterations=global_grid.maxit,
            )
        if (
            global_optimization.method == "none"
            and global_optimization.iterations == 2000
        ):
            global_optimization = GlobalOptimizationOptions(method="none", iterations=0)
        object.__setattr__(
            self,
            "global_optimization",
            global_optimization,
        )


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
    explicit_global_optimization = (
        isinstance(options, Mapping) and "global_optimization" in options
    )
    if (
        not isinstance(grid_spec, GlobalGridSpec)
        and resolved_options.global_optimization.method == "spring"
        and not explicit_global_optimization
    ):
        resolved_options = IconGridOptions(
            max_cells=resolved_options.max_cells,
            accelerator=resolved_options.accelerator,
            global_grid=resolved_options.global_grid,
            global_optimization=GlobalOptimizationOptions(method="none"),
            radius=resolved_options.radius,
            sphere_radius=resolved_options.sphere_radius,
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


def _write_icon_grid(
    grid: IconGrid,
    path: str | Path,
    *,
    sphere_radius: float | None = None,
) -> Path:
    """Write a compact ICON-style NetCDF grid file."""
    _require_complete_icon_grid(grid)
    if sphere_radius is None:
        sphere_radius = grid.options.sphere_radius
    if not np.isclose(sphere_radius, grid.options.sphere_radius):
        raise ValueError(
            "sphere_radius must match the value used by generate_grid(); "
            "pass options={'sphere_radius': ...} when generating the grid"
        )

    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ModuleNotFoundError("NetCDF export requires the netCDF4 package") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        _write_icon_dimensions(dataset, grid)
        _write_icon_attributes(dataset, grid, path)
        for name, dims, data, attrs in _icon_fields(grid):
            variable = dataset.createVariable(name, np.asarray(data).dtype, dims)
            variable[:] = data
            for attr_name, attr_value in attrs.items():
                variable.setncattr(attr_name, attr_value)

    return path


def _require_complete_icon_grid(grid: IconGrid) -> None:
    for name, fields in {
        "icon_connectivity": grid.icon_connectivity,
        "geometry": grid.geometry,
        "refinement": grid.refinement,
    }.items():
        if not fields:
            raise ValueError(f"ICON NetCDF export requires populated {name}")


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


def _generate_grid(
    spec: GlobalGridSpec,
    options: IconGridOptions,
    context: _GlobalGenerationContext | None = None,
) -> IconGrid:
    if context is None:
        context = _GlobalGenerationContext()
    cache_key = context.key(spec)
    cached = context.grids.get(cache_key)
    if cached is not None:
        return cached

    geometry = SphericalIcosahedralGeometry().build(spec, options)
    geometry = IconOrderingBuilder(context).order_spherical_bisection(spec, options, geometry)
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
    metadata = _metadata(spec, options, metrics.fields)

    grid = IconGrid(
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
    if options.global_optimization.method != "none":
        grid = optimize_global_grid(grid, options.global_optimization)
    context.grids[cache_key] = grid
    return grid


def _adjust_global_edge_orientation(
    spec: GlobalGridSpec,
    options: IconGridOptions,
    geometry: GeometryData,
    topology: TopologyData,
    context: _GlobalGenerationContext,
) -> TopologyData:
    if spec.bisections == 0:
        return topology
    parent_spec = GlobalGridSpec(root=spec.root, bisections=spec.bisections - 1)
    parent = _generate_grid(parent_spec, options, context)
    provenance = geometry.bisection_provenance
    if provenance is None:
        parent_vertex_index = _parent_vertex_indices(geometry.vertices, parent)
        parent_for_mapping: IconGrid | _GlobalParentData | BisectionProvenance = parent
        parent_normals = parent.geometry["edge_primal_normal_cartesian"]
    else:
        parent_vertex_index = provenance.parent_vertex_index
        parent_for_mapping = provenance
        parent_vertex_count = int(np.max(parent_vertex_index[parent_vertex_index > 0]))
        provenance_edge_centers = _edge_centers(
            geometry.vertices[:parent_vertex_count],
            provenance.edges,
            options.radius,
        )
        parent_edge_map = _nearest_unit_point_indices(
            provenance_edge_centers,
            parent.edge_center_xyz,
        )
        parent_normals = parent.geometry["edge_primal_normal_cartesian"][parent_edge_map]
    parent_edge_index, _ = _parent_edge_fields(
        topology.edges,
        parent_vertex_index,
        parent_for_mapping,
        options.accelerator,
    )

    edge_system_orientation = _edge_system_orientation(
        geometry.vertices,
        geometry.cell_center_xyz,
        topology.edges,
        topology.edge_cells,
        topology.edge_center_xyz,
    )
    child_normals = _edge_normal_fields(
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

    edge_center_xyz = _edge_centers(geometry.vertices, edges, options.radius)
    edge_lon, edge_lat = _lon_lat(edge_center_xyz)
    icon_connectivity = _icon_connectivity(
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
        connectivity=_public_connectivity(
            geometry.cells,
            edges,
            edge_cells,
            icon_connectivity,
        ),
        neighbor_tables=_neighbor_tables(
            geometry.cells,
            edges,
            edge_cells,
            icon_connectivity,
        ),
        source_edge_index=topology.source_edge_index,
    )


def _nearest_unit_point_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    unit_source = _normalize_rows(source)
    unit_target = _normalize_rows(target)
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


def _parent_grid(
    spec: GlobalGridSpec,
    options: IconGridOptions,
    context: _GlobalGenerationContext,
) -> IconGrid | _GlobalParentData:
    if spec.bisections == 0:
        raise ValueError("grid has no bisection parent")
    parent_spec = GlobalGridSpec(root=spec.root, bisections=spec.bisections - 1)
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
    spec: GlobalGridSpec,
    options: IconGridOptions,
    vertices: np.ndarray,
    context: _GlobalGenerationContext,
) -> tuple[np.ndarray, IconGrid | _GlobalParentData]:
    parent = _parent_grid(spec, options, context)
    cache_key = context.key(spec)
    parent_vertex_index = context.parent_vertex_indices.get(cache_key)
    if parent_vertex_index is None:
        parent_vertex_index = _parent_vertex_indices(vertices, parent)
        context.parent_vertex_indices[cache_key] = parent_vertex_index
    return parent_vertex_index, parent


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


def cut_grid(grid: IconGrid, spec: CutGridSpec) -> IconGrid:
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


def _apply_global_grid_rotation(points: np.ndarray, options: GlobalGridOptions) -> np.ndarray:
    """Apply compatible pole placement and north-pole rotation."""

    if (
        options.north_pole_lon == 0.0
        and options.north_pole_lat == 90.0
        and options.rotation_angle_degrees == 0.0
    ):
        return points.copy()
    matrix = _global_grid_rotation_matrix(options)
    return _normalize_rows(points @ matrix.T)


def _global_grid_rotation_matrix(options: GlobalGridOptions) -> np.ndarray:
    default = _global_grid_seed_vertices(GlobalGridOptions())
    target = _global_grid_seed_vertices(options)
    u, _, vt = np.linalg.svd(default.T @ target)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0.0:
        vt[-1] *= -1.0
        matrix = vt.T @ u.T
    return matrix


def _global_grid_seed_vertices(options: GlobalGridOptions) -> np.ndarray:
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


def _raw_global_grid_rotation(points: np.ndarray, options: GlobalGridOptions) -> np.ndarray:
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
    new_cells = np.empty((new_cell_count, 3), dtype=np.int32)
    raw_cell_edges = np.empty((new_cell_count, 3), dtype=np.int32)
    new_edges = np.empty((new_edge_count, 2), dtype=np.int32)
    split_edge_index = (
        2 * np.arange(old_edge_count, dtype=np.int32)[:, np.newaxis]
        + np.array([0, 1], dtype=np.int32)
    )
    inner_edge_index = (
        2 * old_edge_count
        + 3 * np.arange(old_cell_count, dtype=np.int32)[:, np.newaxis]
        + np.array([0, 1, 2], dtype=np.int32)
    )

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
            common_vertex = _common_edge_vertex(edge_vertices[first_edge], edge_vertices[second_edge])
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

    raw_edge_cells = _edge_cells_from_cell_edges(raw_cell_edges, new_edge_count)
    new_edges = _edge_vertices_from_cell_edges(new_cells, raw_cell_edges, raw_edge_cells)
    new_cells, new_cell_edges = _order_cells_by_edges(
        new_vertices,
        new_cells,
        new_edges,
        raw_cell_edges,
        raw_edge_cells,
    )

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
    )
    return (
        _normalize_rows(new_vertices.astype(np.float64, copy=False)),
        new_cells,
        provenance,
    )


def _common_edge_vertex(first_edge: np.ndarray, second_edge: np.ndarray) -> int:
    common = set(map(int, first_edge)) & set(map(int, second_edge))
    if len(common) != 1:
        raise RuntimeError("parent cell edges do not share exactly one vertex")
    return common.pop()


def _local_vertex_position(cell: np.ndarray, vertex: int) -> int:
    matches = np.flatnonzero(cell == vertex)
    if matches.size != 1:
        raise RuntimeError("parent vertex is not present exactly once in parent cell")
    return int(matches[0])


def _edge_endpoint_slot(edge: np.ndarray, vertex: int) -> int:
    if int(edge[0]) == vertex:
        return 0
    if int(edge[1]) == vertex:
        return 1
    raise RuntimeError("vertex is not an endpoint of parent edge")


def _edge_cells_from_cell_edges(cell_edges: np.ndarray, edge_count: int) -> np.ndarray:
    edge_cells = np.full((edge_count, 2), -1, dtype=np.int32)
    for cell_index, edges in enumerate(cell_edges):
        for edge_index in edges:
            if edge_cells[edge_index, 0] < 0:
                edge_cells[edge_index, 0] = cell_index
            elif edge_cells[edge_index, 1] < 0:
                first = int(edge_cells[edge_index, 0])
                edge_cells[edge_index, 0] = min(first, cell_index)
                edge_cells[edge_index, 1] = max(first, cell_index)
            else:
                raise RuntimeError(f"edge {int(edge_index)} has more than two adjacent cells")
    open_edges = np.flatnonzero(edge_cells[:, 1] < 0)
    if open_edges.size:
        raise RuntimeError(f"edge {int(open_edges[0])} has 1 adjacent cells, expected 2")
    return edge_cells


def _edge_vertices_from_cell_edges(
    cells: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
) -> np.ndarray:
    edges = np.empty((edge_cells.shape[0], 2), dtype=np.int32)
    for edge_index, adjacent_cells in enumerate(edge_cells):
        cell_index = int(adjacent_cells[1])
        if cell_index < 0:
            cell_index = int(adjacent_cells[0])
        local_positions = np.flatnonzero(cell_edges[cell_index] == edge_index)
        if local_positions.size != 1:
            raise RuntimeError("edge is not present exactly once in adjacent cell")
        first = int(local_positions[0])
        edges[edge_index] = (cells[cell_index, first], cells[cell_index, (first + 1) % 3])
        if adjacent_cells[1] < 0:
            edges[edge_index] = edges[edge_index, ::-1]
    return edges


def _order_cells_by_edges(
    vertices: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    cell_edges: np.ndarray,
    edge_cells: np.ndarray,
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
        used_edges = {first_edge}
        for output_index in range(1, 3):
            edge_index, following_vertex = _next_cell_edge(
                edges,
                edges_for_cell,
                used_edges,
                current_vertex,
            )
            ordered_cell_edges[cell_index, output_index] = edge_index
            ordered_cells[cell_index, output_index] = current_vertex
            used_edges.add(edge_index)
            current_edge = edge_index
            current_vertex = following_vertex
        if current_vertex != start_vertex or current_edge < 0:
            raise RuntimeError("cell edges do not form a closed triangle")
    return ordered_cells, ordered_cell_edges


def _next_cell_edge(
    edges: np.ndarray,
    cell_edges: np.ndarray,
    used_edges: set[int],
    vertex: int,
) -> tuple[int, int]:
    for edge_index in map(int, cell_edges):
        if edge_index in used_edges:
            continue
        first, second = map(int, edges[edge_index])
        if first == vertex:
            return edge_index, second
        if second == vertex:
            return edge_index, first
    raise RuntimeError("could not find next cell edge")


def _cell_edge_indices(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges, cell_edges, _ = _build_edges(cells)
    return edges, cell_edges


def _orient_cells_outward(cells: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    triangles = vertices[cells]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward = np.sum(normals * triangles.sum(axis=1), axis=1) < 0.0
    if not np.any(inward):
        return cells

    oriented = cells.copy()
    flipped = oriented[inward]
    flipped[:, [1, 2]] = flipped[:, [2, 1]]
    oriented[inward] = flipped
    return oriented


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


def _write_icon_dimensions(dataset: Any, grid: IconGrid) -> None:
    dataset.createDimension("cell", grid.dims["cell"])
    dataset.createDimension("vertex", grid.dims["vertex"])
    dataset.createDimension("edge", grid.dims["edge"])
    for name, size in FIXED_DIMS.items():
        dataset.createDimension(name, size)


def _write_icon_attributes(dataset: Any, grid: IconGrid, path: Path) -> None:
    external_attrs = {
        "revision": "pure-python",
        "history": f"grid.to_netcdf {path}",
        "date": datetime.now().strftime("%Y%m%d at %H%M%S"),
        "user_name": getpass.getuser(),
        "os_name": platform.platform(),
        "grid_ID": 1,
        "parent_grid_ID": 0,
        "no_of_subgrids": 1,
        "start_subgrid_id": 0,
        "max_childdom": 1,
        "boundary_depth_index": 0,
        "rotation_vector": np.zeros(3, dtype=np.float64),
        "domain_length": grid.metadata.get(
            "domain_length",
            2.0 * np.pi * grid.options.sphere_radius,
        ),
        "domain_height": grid.metadata.get(
            "domain_height",
            2.0 * np.pi * grid.options.sphere_radius,
        ),
        "domain_cartesian_center": np.zeros(3, dtype=np.float64),
    }
    attrs = {
        "title": f"Pure Python ICON grid {grid.name}",
        "institution": "grid_generator",
        "source": "grid_generator Python ICON grid generator",
        "ICON_grid_file_uri": str(path),
        **external_attrs,
        **grid.metadata,
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _icon_fields(grid: IconGrid) -> list[IconNetcdfField]:
    fields = (
        _coordinate_fields(grid)
        + _connectivity_fields(grid)
        + _metric_fields(grid)
        + _refinement_fields_for_netcdf(grid)
        + _static_surface_fields(grid)
        + _cartesian_fields(grid)
        + _normal_vector_fields(grid)
        + _hierarchy_fields(grid)
    )
    return [
        (name, dims, data, _with_icon_variable_attrs(name, attrs))
        for name, dims, data, attrs in fields
    ]


def _coordinate_fields(grid: IconGrid) -> list[IconNetcdfField]:
    edge_bounds_lon, edge_bounds_lat = _edge_lon_lat_bounds(grid)
    return [
        ("clon", ("cell",), np.radians(grid.lon), {"units": "radian"}),
        ("clat", ("cell",), np.radians(grid.lat), {"units": "radian"}),
        ("clon_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lon), {"units": "radian"}),
        ("clat_vertices", ("cell", "nv"), np.radians(grid.cell_vertex_lat), {"units": "radian"}),
        ("vlon", ("vertex",), np.radians(grid.vertex_lon), {"units": "radian"}),
        ("vlat", ("vertex",), np.radians(grid.vertex_lat), {"units": "radian"}),
        ("elon", ("edge",), np.radians(grid.edge_lon), {"units": "radian"}),
        ("elat", ("edge",), np.radians(grid.edge_lat), {"units": "radian"}),
        ("elon_vertices", ("edge", "no"), edge_bounds_lon, {"units": "radian"}),
        ("elat_vertices", ("edge", "no"), edge_bounds_lat, {"units": "radian"}),
        ("lon_cell_centre", ("cell",), np.radians(grid.lon), {"units": "radian"}),
        ("lat_cell_centre", ("cell",), np.radians(grid.lat), {"units": "radian"}),
        ("longitude_vertices", ("vertex",), np.radians(grid.vertex_lon), {"units": "radian"}),
        ("latitude_vertices", ("vertex",), np.radians(grid.vertex_lat), {"units": "radian"}),
        ("lon_edge_centre", ("edge",), np.radians(grid.edge_lon), {"units": "radian"}),
        ("lat_edge_centre", ("edge",), np.radians(grid.edge_lat), {"units": "radian"}),
    ]


def _connectivity_fields(grid: IconGrid) -> list[IconNetcdfField]:
    connectivity = grid.icon_connectivity
    return [
        ("edge_of_cell", ("nv", "cell"), connectivity["c2e"].T + 1, {}),
        ("vertex_of_cell", ("nv", "cell"), grid.cells.T + 1, {}),
        ("neighbor_cell_index", ("nv", "cell"), connectivity["c2c"].T + 1, {}),
        ("adjacent_cell_of_edge", ("nc", "edge"), grid.edge_cells.T + 1, {}),
        ("edge_vertices", ("nc", "edge"), grid.edges.T + 1, {}),
        ("cells_of_vertex", ("ne", "vertex"), connectivity["v2c"].T, {}),
        ("edges_of_vertex", ("ne", "vertex"), connectivity["v2e"].T, {}),
        ("vertices_of_vertex", ("ne", "vertex"), connectivity["v2v"].T, {}),
    ]


def _metric_fields(grid: IconGrid) -> list[IconNetcdfField]:
    geometry = grid.geometry
    edgequad_normalizer = (
        1.0 if grid.metadata.get("grid_geometry") == 2 else grid.options.sphere_radius**2
    )
    return [
        ("cell_area", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("cell_area_p", ("cell",), geometry["cell_area"], {"units": "m2"}),
        ("dual_area_p", ("vertex",), geometry["dual_area"], {"units": "m2"}),
        ("edge_length", ("edge",), geometry["edge_length"], {"units": "m"}),
        ("dual_edge_length", ("edge",), geometry["dual_edge_length"], {"units": "m"}),
        ("edge_cell_distance", ("nc", "edge"), geometry["edge_cell_distance"].T, {"units": "m"}),
        ("edge_vert_distance", ("nc", "edge"), geometry["edge_vert_distance"].T, {"units": "m"}),
        (
            "edgequad_area",
            ("edge",),
            geometry["edgequad_area"] / edgequad_normalizer,
            {"units": "m2"},
        ),
        ("orientation_of_normal", ("nv", "cell"), geometry["orientation_of_normal"].T, {}),
        ("edge_system_orientation", ("edge",), geometry["edge_system_orientation"], {}),
        ("edge_orientation", ("ne", "vertex"), geometry["edge_orientation"].T, {}),
    ]


def _refinement_fields_for_netcdf(grid: IconGrid) -> list[IconNetcdfField]:
    refinement = grid.refinement
    return [
        ("refin_c_ctrl", ("cell",), refinement["refin_c_ctrl"], {}),
        ("refin_e_ctrl", ("edge",), refinement["refin_e_ctrl"], {}),
        ("refin_v_ctrl", ("vertex",), refinement["refin_v_ctrl"], {}),
        ("start_idx_c", ("max_chdom", "cell_grf"), refinement["start_idx_c"], {}),
        ("end_idx_c", ("max_chdom", "cell_grf"), refinement["end_idx_c"], {}),
        ("start_idx_e", ("max_chdom", "edge_grf"), refinement["start_idx_e"], {}),
        ("end_idx_e", ("max_chdom", "edge_grf"), refinement["end_idx_e"], {}),
        ("start_idx_v", ("max_chdom", "vert_grf"), refinement["start_idx_v"], {}),
        ("end_idx_v", ("max_chdom", "vert_grf"), refinement["end_idx_v"], {}),
    ]


def _static_surface_fields(grid: IconGrid) -> list[IconNetcdfField]:
    zeros_cell = np.zeros(grid.dims["cell"], dtype=np.float64)
    zeros_edge = np.zeros(grid.dims["edge"], dtype=np.float64)
    return [
        ("cell_elevation", ("cell",), zeros_cell, {"units": "m"}),
        ("edge_elevation", ("edge",), zeros_edge, {"units": "m"}),
        ("cell_sea_land_mask", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("edge_sea_land_mask", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
    ]


def _cartesian_fields(grid: IconGrid) -> list[IconNetcdfField]:
    if grid.metadata.get("grid_geometry") == 2:
        unit_vertices = grid.vertices
        unit_centers = grid.cell_center_xyz
        unit_edge_centers = grid.edge_center_xyz
    else:
        unit_vertices = _normalize_rows(grid.vertices)
        unit_centers = _normalize_rows(grid.cell_center_xyz)
        unit_edge_centers = _normalize_rows(grid.edge_center_xyz)
    return [
        ("cartesian_x_vertices", ("vertex",), unit_vertices[:, 0], {"units": "meters"}),
        ("cartesian_y_vertices", ("vertex",), unit_vertices[:, 1], {"units": "meters"}),
        ("cartesian_z_vertices", ("vertex",), unit_vertices[:, 2], {"units": "meters"}),
        ("cell_circumcenter_cartesian_x", ("cell",), unit_centers[:, 0], {"units": "meters"}),
        ("cell_circumcenter_cartesian_y", ("cell",), unit_centers[:, 1], {"units": "meters"}),
        ("cell_circumcenter_cartesian_z", ("cell",), unit_centers[:, 2], {"units": "meters"}),
        ("edge_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], {"units": "meters"}),
        ("edge_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], {"units": "meters"}),
        ("edge_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], {"units": "meters"}),
        ("phys_cell_id", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("phys_edge_id", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("cell_index", ("cell",), np.arange(1, grid.dims["cell"] + 1, dtype=np.int32), {}),
        ("edge_index", ("edge",), np.arange(1, grid.dims["edge"] + 1, dtype=np.int32), {}),
        ("vertex_index", ("vertex",), np.arange(1, grid.dims["vertex"] + 1, dtype=np.int32), {}),
        ("edge_dual_middle_cartesian_x", ("edge",), unit_edge_centers[:, 0], {"units": "meters"}),
        ("edge_dual_middle_cartesian_y", ("edge",), unit_edge_centers[:, 1], {"units": "meters"}),
        ("edge_dual_middle_cartesian_z", ("edge",), unit_edge_centers[:, 2], {"units": "meters"}),
    ]


def _normal_vector_fields(grid: IconGrid) -> list[IconNetcdfField]:
    geometry = grid.geometry
    return [
        (
            "edge_primal_normal_cartesian_x",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 0],
            {"units": "meters"},
        ),
        (
            "edge_primal_normal_cartesian_y",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 1],
            {"units": "meters"},
        ),
        (
            "edge_primal_normal_cartesian_z",
            ("edge",),
            geometry["edge_primal_normal_cartesian"][:, 2],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_x",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 0],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_y",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 1],
            {"units": "meters"},
        ),
        (
            "edge_dual_normal_cartesian_z",
            ("edge",),
            geometry["edge_dual_normal_cartesian"][:, 2],
            {"units": "meters"},
        ),
        ("zonal_normal_primal_edge", ("edge",), geometry["zonal_normal_primal_edge"], {"units": "radian"}),
        (
            "meridional_normal_primal_edge",
            ("edge",),
            geometry["meridional_normal_primal_edge"],
            {"units": "radian"},
        ),
        ("zonal_normal_dual_edge", ("edge",), geometry["zonal_normal_dual_edge"], {"units": "radian"}),
        (
            "meridional_normal_dual_edge",
            ("edge",),
            geometry["meridional_normal_dual_edge"],
            {"units": "radian"},
        ),
    ]


def _hierarchy_fields(grid: IconGrid) -> list[IconNetcdfField]:
    refinement = grid.refinement
    return [
        ("parent_cell_index", ("cell",), refinement["parent_cell_index"], {}),
        ("parent_cell_type", ("cell",), refinement["parent_cell_type"], {}),
        ("edge_parent_type", ("edge",), refinement["edge_parent_type"], {}),
        ("parent_edge_index", ("edge",), refinement["parent_edge_index"], {}),
        ("parent_vertex_index", ("vertex",), refinement["parent_vertex_index"], {}),
        ("child_cell_index", ("no", "cell"), np.zeros((4, grid.dims["cell"]), dtype=np.int32), {}),
        ("child_cell_id", ("cell",), np.zeros(grid.dims["cell"], dtype=np.int32), {}),
        ("child_edge_index", ("no", "edge"), np.zeros((4, grid.dims["edge"]), dtype=np.int32), {}),
        ("child_edge_id", ("edge",), np.zeros(grid.dims["edge"], dtype=np.int32), {}),
    ]


def _with_icon_variable_attrs(name: str, attrs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(ICON_VARIABLE_ATTRS.get(name, {}))
    merged.update(attrs)
    return merged


def _edge_lon_lat_bounds(grid: IconGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return ICON-style four-point edge bounds in radians.

    The upstream grid generator stores bounds for each edge as a quadrilateral:
    first edge vertex, second adjacent cell center, second edge vertex, first
    adjacent cell center.
    """
    edge_vertices = np.asarray(grid.edges, dtype=np.int32)
    edge_cells = np.asarray(grid.edge_cells, dtype=np.int32)
    lon = np.empty((grid.dims["edge"], 4), dtype=np.float64)
    lat = np.empty((grid.dims["edge"], 4), dtype=np.float64)

    lon[:, 0] = grid.vertex_lon[edge_vertices[:, 0]]
    lat[:, 0] = grid.vertex_lat[edge_vertices[:, 0]]
    second_cell = edge_cells[:, 1]
    second_cell_lon = np.where(second_cell >= 0, grid.lon[np.maximum(second_cell, 0)], grid.edge_lon)
    second_cell_lat = np.where(second_cell >= 0, grid.lat[np.maximum(second_cell, 0)], grid.edge_lat)
    lon[:, 1] = second_cell_lon
    lat[:, 1] = second_cell_lat
    lon[:, 2] = grid.vertex_lon[edge_vertices[:, 1]]
    lat[:, 2] = grid.vertex_lat[edge_vertices[:, 1]]
    first_cell = edge_cells[:, 0]
    first_cell_lon = np.where(first_cell >= 0, grid.lon[np.maximum(first_cell, 0)], grid.edge_lon)
    first_cell_lat = np.where(first_cell >= 0, grid.lat[np.maximum(first_cell, 0)], grid.edge_lat)
    lon[:, 3] = first_cell_lon
    lat[:, 3] = first_cell_lat

    pole_mask = np.isclose(np.abs(lat), 90.0)
    lon[pole_mask] = np.repeat(grid.edge_lon[:, np.newaxis], 4, axis=1)[pole_mask]
    return np.radians(lon), np.radians(lat)


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
                "lon_min": spec.lon_min,
                "lon_max": spec.lon_max,
                "lat_min": spec.lat_min,
                "lat_max": spec.lat_max,
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
                "bounds": [
                    _canonical_float(spec.lon_min),
                    _canonical_float(spec.lon_max),
                    _canonical_float(spec.lat_min),
                    _canonical_float(spec.lat_max),
                ],
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
    global_grid: GlobalGridOptions | Mapping[str, Any] | None = None,
    global_optimization: GlobalOptimizationOptions | Mapping[str, Any] | str | None = None,
) -> str:
    canonical_sphere_radius = finite_float_option("sphere_radius", sphere_radius)
    if canonical_sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be positive")
    grid_options = global_grid
    if grid_options is None:
        grid_options = GlobalGridOptions()
    elif isinstance(grid_options, Mapping):
        grid_options = GlobalGridOptions(**dict(grid_options))
    if not isinstance(grid_options, GlobalGridOptions):
        raise TypeError("global_grid must be None, a GlobalGridOptions instance, or a mapping")
    if global_optimization is None:
        optimization_options = GlobalOptimizationOptions(
            method="spring",
            iterations=grid_options.maxit,
        )
    else:
        optimization_options = resolve_global_optimization_options(global_optimization)
    payload = {
        "generator": "grid_generator",
        "grid": parse_grid_spec(grid_name).name,
        "sphere_radius": _canonical_float(canonical_sphere_radius),
        "global_grid": _canonicalize_payload(asdict(grid_options)),
        "global_optimization": _canonicalize_payload(asdict(optimization_options)),
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
