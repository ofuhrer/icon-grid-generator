"""Grid cutting helpers.

Use ``cut_grid(grid, region)`` for a simple single-region cut, or
``cut_grid(grid, CutGridSpec(...))`` when multiple regions or advanced cut
options are needed.
"""

from .grid_generator import CutGridSpec, cut_grid

__all__ = [
    "CutGridSpec",
    "cut_grid",
]
