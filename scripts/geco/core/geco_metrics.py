"""
Shared evaluation metrics for GECO benchmarks: reproducible RNG and line-level recovery.
"""
import hashlib
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed for NumPy; stable across processes (unlike built-in hash())."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "little")


def get_deterministic_seed(*parts: object) -> int:
    """
    Alias for stable_seed().
    Use this name in experiment scripts to emphasize reproducibility of injected noise.
    """
    return stable_seed(*parts)


def word_line_ids_from_layout(df_layout: pd.DataFrame, gap_px: float = 22.0) -> np.ndarray:
    """
    Infer typographic line index per word by sorting on true_y and splitting when vertical gap > gap_px.
    """
    y = df_layout["true_y"].astype(float).values
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(y)
    line_by_word = np.zeros(n, dtype=np.int32)
    lid = 0
    line_by_word[order[0]] = 0
    for i in range(1, n):
        if y[order[i]] - y[order[i - 1]] > gap_px:
            lid += 1
        line_by_word[order[i]] = lid
    return line_by_word


def line_recovery_rate(
    target_indices: Sequence[int],
    predicted_indices: Sequence[int],
    line_by_word: np.ndarray,
) -> float:
    """Fraction of steps where predicted word lies on the same typographic line as the target word."""
    total = len(target_indices)
    if total == 0:
        return 0.0
    nlines = len(line_by_word)
    ok = 0
    for t, p in zip(target_indices, predicted_indices):
        ti = int(t)
        pi = int(p)
        if 0 <= ti < nlines and 0 <= pi < nlines and line_by_word[ti] == line_by_word[pi]:
            ok += 1
    return 100.0 * ok / total


def drift_alignment_rate(estimated_drift_y: float, true_drift_y: float, tol_px: float = 15.0) -> float:
    """Binary score: 100 if vertical drift estimate is within tol_px of injected drift, else 0."""
    return 100.0 if abs(float(estimated_drift_y) - float(true_drift_y)) < tol_px else 0.0


def evaluate_word_and_recovery(
    target_indices: Sequence[int],
    predicted_indices: Sequence[int],
    line_by_word: np.ndarray,
    estimated_drift_y: Optional[float],
    true_drift_y: float,
) -> Tuple[float, float, float, float]:
    """
    Returns (word_acc, top3_acc, line_recovery, drift_alignment).
    drift_alignment is NaN if estimated_drift_y is None (non-EM decoders).
    """
    total = len(target_indices)
    if total == 0:
        return 0.0, 0.0, 0.0, float("nan")

    acc = sum(1 for t, p in zip(target_indices, predicted_indices) if int(t) == int(p)) / total * 100
    top3 = (
        sum(1 for t, p in zip(target_indices, predicted_indices) if abs(int(t) - int(p)) <= 1) / total * 100
    )
    lrec = line_recovery_rate(target_indices, predicted_indices, line_by_word)
    if estimated_drift_y is None:
        dalign = float("nan")
    else:
        dalign = drift_alignment_rate(estimated_drift_y, true_drift_y)
    return acc, top3, lrec, dalign
