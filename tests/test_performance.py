from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        os.environ.get("GRID_GENERATOR_PERF_TESTS") != "1",
        reason="set GRID_GENERATOR_PERF_TESTS=1 or run make perf-check",
    ),
]


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PerformanceCase:
    grid: str
    accelerator: str
    attempts: int
    max_best_seconds: float
    max_best_rss_mib: float


def _run_generation_once(grid: str, accelerator: str) -> dict[str, float | int | str]:
    code = f"""
import json
import resource
import time

from grid_generator import generate_grid

start = time.perf_counter()
grid = generate_grid(
    {grid!r},
    options={{"max_cells": None, "optimize_global": False, "accelerator": {accelerator!r}}},
)
elapsed = time.perf_counter() - start
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({{
    "grid": grid.name,
    "accelerator": {accelerator!r},
    "seconds": elapsed,
    "rss_mib": rss / (1024 * 1024),
    "cells": grid.dims["cell"],
    "edges": grid.dims["edge"],
    "vertices": grid.dims["vertex"],
}}))
"""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _best_of(case: PerformanceCase) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    results = [_run_generation_once(case.grid, case.accelerator) for _ in range(case.attempts)]
    return min(results, key=lambda row: float(row["seconds"])), results


@pytest.mark.parametrize(
    "case",
    [
        PerformanceCase(
            grid="R02B04",
            accelerator="auto",
            attempts=5,
            max_best_seconds=1.25,
            max_best_rss_mib=180.0,
        ),
        PerformanceCase(
            grid="R02B05",
            accelerator="auto",
            attempts=5,
            max_best_seconds=3.25,
            max_best_rss_mib=360.0,
        ),
        PerformanceCase(
            grid="R01B07",
            accelerator="auto",
            attempts=4,
            max_best_seconds=11.0,
            max_best_rss_mib=1_350.0,
        ),
    ],
    ids=lambda case: f"{case.grid}-{case.accelerator}",
)
def test_raw_global_generation_performance_regression(case: PerformanceCase):
    best, results = _best_of(case)

    assert best["seconds"] <= case.max_best_seconds, results
    assert best["rss_mib"] <= case.max_best_rss_mib, results

