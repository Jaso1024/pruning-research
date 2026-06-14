from __future__ import annotations

import numpy as np


def causal_ema_filter(source: np.ndarray, beta: float, *, axis: int) -> np.ndarray:
    """Apply y[t] = x[t] + beta * y[t - 1] along one axis.

    This matches scipy.signal.lfilter([1.0], [1.0, -beta], source, axis=axis)
    for the causal exponential filters used by the attention-fit scripts, while
    avoiding a hard SciPy dependency for simple CLI/import smoke checks.
    """
    values = np.asarray(source, dtype=np.float32)
    moved = np.moveaxis(values, axis, 0)
    out = np.empty_like(moved, dtype=np.float32)
    running = np.zeros_like(moved[0], dtype=np.float32)
    beta32 = np.float32(beta)
    for index in range(moved.shape[0]):
        running = moved[index] + beta32 * running
        out[index] = running
    return np.moveaxis(out, 0, axis)
