# core/gaml.py
# -*- coding: utf-8 -*-
"""
GAML 的简化实现：对每个视角做稳健 z-score 对齐，并输出“异常置信度”。

接口两种模式：

1) 拟合参数（fit 阶段）：
    params = calibrate_scores_with_gaml(scores_dict, labels)

   - scores_dict: {view_name: np.ndarray[N]}，通常为 val split 的分数
   - labels: np.ndarray[N]，当前版本不使用（预留）

2) 使用已拟合参数做校正（transform 阶段）：
    s_calib_dict, conf_dict = calibrate_scores_with_gaml(
        scores_dict, labels=None, params=params, return_conf=True
    )

   - s_calib_dict: {view: z_scores}，校正后的稳健 z-score（可跨视角比较）
   - conf_dict:    {view: c}，异常置信度 c_i(x) ∈ [0,1)
                 定义为：c = |z| / (1 + |z|)
                 即：越偏离 normal（|z| 越大）越“异常确信”

重要：
- 本项目的 A-EVT 设计假设 c 表示“异常置信度”：
    c 高 -> 阈值下调 -> 对明显异常更敏感
    c 低 -> 阈值上调 -> 抑制不确定噪声
"""

from typing import Dict, Optional
import numpy as np


def _robust_zscore(scores: np.ndarray, med: float, mad: float) -> np.ndarray:
    """基于 median + MAD 的稳健 z-score。"""
    scores = np.asarray(scores, dtype=np.float64)
    denom = 1.4826 * (float(mad) + 1e-8)
    return (scores - float(med)) / denom


def calibrate_scores_with_gaml(
    scores_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Dict[str, float]]] = None,
    return_conf: bool = False,
):
    """
    GAML 主入口。

    参数
    ----
    scores_dict : dict
        {view_name: scores}，scores 为 [N] 的一维数组。
    labels : np.ndarray 或 None
        可选，目前未使用（保留接口以便未来加入先验相关性）。
    params : dict 或 None
        None => fit 模式；否则 transform 模式。
    return_conf : bool
        仅 transform 模式下生效，返回每个视角的异常置信度 c_i(x)。

    返回
    ----
    fit 模式：params = {view: {"med": med, "mad": mad}}
    transform 模式：
        - return_conf=False: {view: z_scores}
        - return_conf=True : ({view: z_scores}, {view: conf})
    """
    view_names = list(scores_dict.keys())

    # ---------- fit 模式 ----------
    if params is None:
        params = {}
        for v in view_names:
            s = np.asarray(scores_dict[v], dtype=np.float64)
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-8
            params[v] = {"med": float(med), "mad": float(mad)}
        return params

    # ---------- transform 模式 ----------
    s_calib: Dict[str, np.ndarray] = {}
    conf: Dict[str, np.ndarray] = {}

    for v in view_names:
        s = np.asarray(scores_dict[v], dtype=np.float64)
        med = params[v]["med"]
        mad = params[v]["mad"]
        z = _robust_zscore(s, med, mad).astype(np.float32)
        s_calib[v] = z

        # 异常置信度：|z| 越大越异常
        absz = np.abs(z).astype(np.float32)
        # 数值稳定：避免极端 absz 导致 inf
        absz = np.minimum(absz, np.float32(1e6))
        c = absz / (1.0 + absz)
        conf[v] = c.astype(np.float32)

    if return_conf:
        return s_calib, conf
    return s_calib
