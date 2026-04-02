# core/unk_cluster.py
# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN, KMeans


def cluster_unknowns(
    features: np.ndarray,
    method: str = "dbscan",
    params: Optional[Dict[str, Any]] = None,
    sample_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    对 UNKNOWN 样本的表示做聚类，返回簇编号。

    参数
    ----
    features : np.ndarray, shape [N, D]
        UNKNOWN 样本的特征表示（例如 AE 潜表示）。
    method   : str
        使用的聚类方法，目前支持:
          - "dbscan"
          - "kmeans"
    params   : dict 或 None
        聚类器的超参数字典。
        通用键：
            "max_samples" : int,   参加聚类的最大样本数，默认 None=全量
            "seed"        : int,   随机种子，默认 0
        DBSCAN 额外键:
            "eps"         : float, 近邻半径，默认 0.5
            "min_samples" : int,   最小样本数，默认 5
        KMeans 额外键:
            "n_clusters"  : int,   聚类簇数，默认 5
    sample_scores : np.ndarray 或 None, shape [N]
        若提供，则用作“重要性采样”的权重（越大越容易被采到），
        用于模仿 Trident 中按重构误差进行采样的思路。
        若为 None，则使用均匀随机采样。

    返回
    ----
    labels : np.ndarray[int], shape [N]
        每个样本所属簇的编号。DBSCAN 中 -1 表示噪声点。
        若 features 为空，则返回 shape [0] 的 int 数组。

    说明
    ----
    为了避免在大规模 UNKNOWN 上运行 DBSCAN/KMeans 时爆内存/时间，
    若 N > max_samples，则先对 UNKNOWN 做子采样再聚类：
      - 若 sample_scores 为 None，则均匀随机采样；
      - 若 sample_scores 不为 None，则按 scores 做加权采样
        （scores 越大，越容易被选中）。
    未被采样的样本统一标记为簇 -1（视作“未聚类/噪声”）。
    """
    features = np.asarray(features, dtype=np.float32)
    N = features.shape[0]
    if N == 0:
        return np.array([], dtype=int)

    if params is None:
        params = {}

    if sample_scores is not None:
        sample_scores = np.asarray(sample_scores, dtype=np.float32)
        if sample_scores.shape[0] != N:
            raise ValueError(
                f"sample_scores 长度 {sample_scores.shape[0]} "
                f"与 features 数量 {N} 不一致。"
            )

    method = method.lower()
    max_samples = params.get("max_samples", None)  # 默认不采样，全量
    seed = params.get("seed", 0)

    # ========== 下采样（uniform 或 score-based） ==========
    if max_samples is not None and N > max_samples:
        rng = np.random.RandomState(seed)
        idx_all = np.arange(N)

        if sample_scores is None:
            # 均匀随机采样
            sel_idx = rng.choice(idx_all, size=max_samples, replace=False)
        else:
            # 按 scores 加权采样（scores 越大，被选中的概率越高）
            scores = sample_scores.astype(np.float64)
            scores = scores - scores.min()
            total = scores.sum()
            if total <= 0:
                # 极端情况：scores 全一样或全零，退化为均匀采样
                sel_idx = rng.choice(idx_all, size=max_samples, replace=False)
            else:
                p = scores / total
                sel_idx = rng.choice(idx_all, size=max_samples, replace=False, p=p)
    else:
        sel_idx = np.arange(N)

    feats_used = features[sel_idx]

    # ========== 聚类 ==========
    if method == "dbscan":
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels_small = model.fit_predict(feats_used)  # [len(sel_idx)]

    elif method == "kmeans":
        n_clusters = params.get("n_clusters", 5)
        model = KMeans(n_clusters=n_clusters, random_state=seed)
        labels_small = model.fit_predict(feats_used)

    else:
        raise ValueError(f"未知聚类方法: {method}. 目前支持 'dbscan' 和 'kmeans'。")

    # 把未参与聚类的样本统一标记为 -1
    labels = np.full(N, -1, dtype=int)
    labels[sel_idx] = labels_small.astype(int)

    return labels
