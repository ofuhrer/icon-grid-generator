"""Optional acceleration helpers.

The package must remain usable without Numba. Keep this module importable in a
plain NumPy environment and import Numba only inside functions that need it.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util

import numpy as np


ACCELERATOR_AUTO = "auto"
ACCELERATOR_NUMBA = "numba"
ACCELERATOR_NUMPY = "numpy"
SUPPORTED_ACCELERATORS = {
    ACCELERATOR_AUTO,
    ACCELERATOR_NUMBA,
    ACCELERATOR_NUMPY,
}
AUTO_NUMBA_MIN_LOOKUP_ROWS = 1_000_000


def is_numba_available() -> bool:
    return importlib.util.find_spec("numba") is not None


def should_use_numba(accelerator: str, work_items: int | None = None) -> bool:
    if accelerator == ACCELERATOR_NUMPY:
        return False
    if accelerator == ACCELERATOR_NUMBA:
        if not is_numba_available():
            raise ModuleNotFoundError(
                "Numba acceleration requires installing the 'accelerate' extra"
            )
        return True
    if work_items is None or work_items < AUTO_NUMBA_MIN_LOOKUP_ROWS:
        return False
    return is_numba_available()


def lookup_width2_numba(
    signature_keys: np.ndarray,
    parent_index_values: np.ndarray,
    type_values: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _compiled_lookup_width2()(
        signature_keys,
        parent_index_values,
        type_values,
        query_keys,
    )


def lookup_width3_numba(
    signature_keys: np.ndarray,
    parent_index_values: np.ndarray,
    type_values: np.ndarray,
    query_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _compiled_lookup_width3()(
        signature_keys,
        parent_index_values,
        type_values,
        query_keys,
    )


@lru_cache(maxsize=1)
def _compiled_lookup_width2():
    from numba import njit

    @njit
    def lookup(signature_keys, parent_index_values, type_values, query_keys):
        out_parent = np.empty(query_keys.shape[0], dtype=np.int32)
        out_type = np.empty(query_keys.shape[0], dtype=np.int32)
        for row in range(query_keys.shape[0]):
            q0 = query_keys[row, 0]
            q1 = query_keys[row, 1]
            low = 0
            high = signature_keys.shape[0]
            found = -1
            while low < high:
                mid = (low + high) // 2
                k0 = signature_keys[mid, 0]
                k1 = signature_keys[mid, 1]
                if k0 < q0 or (k0 == q0 and k1 < q1):
                    low = mid + 1
                elif k0 > q0 or (k0 == q0 and k1 > q1):
                    high = mid
                else:
                    found = mid
                    break
            if found < 0:
                out_parent[row] = 0
                out_type[row] = 0
            else:
                out_parent[row] = parent_index_values[found]
                out_type[row] = type_values[found]
        return out_parent, out_type

    return lookup


@lru_cache(maxsize=1)
def _compiled_lookup_width3():
    from numba import njit

    @njit
    def lookup(signature_keys, parent_index_values, type_values, query_keys):
        out_parent = np.empty(query_keys.shape[0], dtype=np.int32)
        out_type = np.empty(query_keys.shape[0], dtype=np.int32)
        for row in range(query_keys.shape[0]):
            q0 = query_keys[row, 0]
            q1 = query_keys[row, 1]
            q2 = query_keys[row, 2]
            low = 0
            high = signature_keys.shape[0]
            found = -1
            while low < high:
                mid = (low + high) // 2
                k0 = signature_keys[mid, 0]
                k1 = signature_keys[mid, 1]
                k2 = signature_keys[mid, 2]
                if k0 < q0 or (k0 == q0 and (k1 < q1 or (k1 == q1 and k2 < q2))):
                    low = mid + 1
                elif k0 > q0 or (k0 == q0 and (k1 > q1 or (k1 == q1 and k2 > q2))):
                    high = mid
                else:
                    found = mid
                    break
            if found < 0:
                out_parent[row] = 0
                out_type[row] = 0
            else:
                out_parent[row] = parent_index_values[found]
                out_type[row] = type_values[found]
        return out_parent, out_type

    return lookup
