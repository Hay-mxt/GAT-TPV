# core/tpcv.py
# -*- coding: utf-8 -*-
"""
TPCV k-of-N 共识决策。

给定多视角分数 s'_i(x) 以及对应的样本级阈值 T'_i(x)，输出三类决策：

    0: KNOWN
    1: ALERT
    2: ANOMALY (to observe)
"""

import numpy as np
from typing import Tuple


def tpcv_decision(
    scores: np.ndarray,
    thresholds: np.ndarray,
    delta: float = 0.0,
    k: int = 2,
) -> np.ndarray:
    """
    参数
    ----
    scores : np.ndarray[N, V]
        校正后的多视角分数 s'_i(x)。
    thresholds : np.ndarray[N, V]
        对应的样本级阈值 T'_i(x)。
    delta : float
        强异常 margin，论文中记作 δ。
    k : int
        至少有 k 个视角“强异常”时升级为 ALERT。

    返回
    ----
    decision : np.ndarray[N]
        每个样本的决策类别：0=KNOWN, 1=ALERT, 2=ANOMALY。
    """
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    diff = scores - thresholds  # >0 表示超出阈值

    min_diff = diff.min(axis=1)
    count_strong = (diff >= delta).sum(axis=1)

    decision = np.full(scores.shape[0], 2, dtype=np.int64)  # 默认 ANOMALY
    decision[min_diff < 0.0] = 0  # 有视角不过阈值 => KNOWN
    mask_alert = (min_diff >= 0.0) & (count_strong >= k)
    decision[mask_alert] = 1       # 所有视角至少达阈，且 >=k 个超出 δ => ALERT
    return decision


def anomaly_score_from_diff(diff: np.ndarray) -> np.ndarray:
    """
    给定 diff = s'_i(x) - T'_i(x) 的矩阵，返回一个标量异常分数，
    这里简单取 max_i diff_i(x)。
    """
    diff = np.asarray(diff, dtype=np.float64)
    return diff.max(axis=1).astype(np.float32)
