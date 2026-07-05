"""Refinement-control and parent-provenance field computation."""

from __future__ import annotations

from typing import Any

from ._types import GeometryData, RefinementData, TopologyData


class GlobalRefinementBuilder:
    """Compute parent-provenance fields for global bisection refinement."""

    def __init__(self, context: Any | None = None) -> None:
        self.context = context

    def build(
        self,
        spec: Any,
        options: Any,
        geometry: GeometryData,
        topology: TopologyData,
    ) -> RefinementData:
        from . import grid_generator as gg

        return RefinementData(
            fields=gg._refinement_fields(
                spec,
                options,
                geometry,
                topology.edges,
                self.context,
            )
        )
