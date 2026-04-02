# core/a_evt.py
# -*- coding: utf-8 -*-
"""
条件极值阈值（Adaptive EVT, A-EVT）的简化实现。

思路：
    - 对每个视角的校正分数 s'_i(x) 在验证集上做高分分位数得到基准阈值 T_i；
    - 同时记录该视角的 MAD_i 作为尺度；
    - 推理期结合置信度 c_i(x) 调节阈值，得到样本级阈值 T'_i(x)：

        T'_i(x) = clip(
            T_i + [β(1 - c_i(x)) - γ c_i(x)] MAD_i,
            T_i - γ MAD_i,
            T_i + β MAD_i
        )

    与你论文 3.5 节中公式一致（只是这里没有显式拟合 POT/GPD）。
"""

from typing import Dict

import numpy as np


def fit_evt_for_view(
    scores: np.ndarray,
    q: float = 0.995,
    beta: float = 1.0,
    gamma: float = 3.0,
) -> Dict[str, float]:
    """
    在验证集上为单个视角拟合 EVT 基准阈值。

    参数
    ----
    scores : np.ndarray[N]
        校正后的分数 s'_i(x)，数值越大越异常。
    q : float
        用于确定阈值的分位数，0.995 表示 top 0.5%。
    beta, gamma : float
        控制上下调幅度的超参。

    返回
    ----
    params : dict
        {"T": T_i, "MAD": MAD_i, "beta": beta, "gamma": gamma}
    """
    scores = np.asarray(scores, dtype=np.float64)
    T = float(np.quantile(scores, q))
    med = float(np.median(scores))
    MAD = float(np.median(np.abs(scores - med)) + 1e-8)
    return {"T": T, "MAD": MAD, "beta": beta, "gamma": gamma}


def compute_aevt_threshold(
    conf: np.ndarray,
    evt_params: Dict[str, float],
) -> np.ndarray:
    """
    根据 EVT 参数和置信度 c_i(x) 计算样本级阈值 T'_i(x)。

    参数
    ----
    conf : np.ndarray[N]
        该视角的置信度 c_i(x) ∈ [0,1]。
    evt_params : dict
        由 fit_evt_for_view 返回。

    返回
    ----
    T_prime : np.ndarray[N]
        每个样本对应的阈值 T'_i(x)。
    """
    c = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    T = evt_params["T"]
    MAD = evt_params["MAD"]
    beta = evt_params.get("beta", 1.0)
    gamma = evt_params.get("gamma", 3.0)

    lower = T - gamma * MAD
    upper = T + beta * MAD

    T_prime = T + (beta * (1.0 - c) - gamma * c) * MAD
    T_prime = np.clip(T_prime, lower, upper)
    return T_prime.astype(np.float32)
