
import numpy as np
from typing import Tuple


def tpcv_decision(
    scores: np.ndarray,
    thresholds: np.ndarray,
    delta: float = 0.0,
    k: int = 2,
) -> np.ndarray:

    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    diff = scores - thresholds  # >0 表示超出阈值

    min_diff = diff.min(axis=1)
    count_strong = (diff >= delta).sum(axis=1)

    decision = np.full(scores.shape[0], 2, dtype=np.int64)  
    decision[min_diff < 0.0] = 0  
    mask_alert = (min_diff >= 0.0) & (count_strong >= k)
    decision[mask_alert] = 1       
    return decision


def anomaly_score_from_diff(diff: np.ndarray) -> np.ndarray:

    diff = np.asarray(diff, dtype=np.float64)
    return diff.max(axis=1).astype(np.float32)
