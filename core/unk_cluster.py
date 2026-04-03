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
    max_samples = params.get("max_samples", None)  
    seed = params.get("seed", 0)

    if max_samples is not None and N > max_samples:
        rng = np.random.RandomState(seed)
        idx_all = np.arange(N)

        if sample_scores is None:
            
            sel_idx = rng.choice(idx_all, size=max_samples, replace=False)
        else:
            
            scores = sample_scores.astype(np.float64)
            scores = scores - scores.min()
            total = scores.sum()
            if total <= 0:
                
                sel_idx = rng.choice(idx_all, size=max_samples, replace=False)
            else:
                p = scores / total
                sel_idx = rng.choice(idx_all, size=max_samples, replace=False, p=p)
    else:
        sel_idx = np.arange(N)

    feats_used = features[sel_idx]

   
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

    labels = np.full(N, -1, dtype=int)
    labels[sel_idx] = labels_small.astype(int)

    return labels
