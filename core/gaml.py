
from typing import Dict, Optional
import numpy as np


def _robust_zscore(scores: np.ndarray, med: float, mad: float) -> np.ndarray:

    scores = np.asarray(scores, dtype=np.float64)
    denom = 1.4826 * (float(mad) + 1e-8)
    return (scores - float(med)) / denom


def calibrate_scores_with_gaml(
    scores_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Dict[str, float]]] = None,
    return_conf: bool = False,
):

    view_names = list(scores_dict.keys())

    if params is None:
        params = {}
        for v in view_names:
            s = np.asarray(scores_dict[v], dtype=np.float64)
            med = np.median(s)
            mad = np.median(np.abs(s - med)) + 1e-8
            params[v] = {"med": float(med), "mad": float(mad)}
        return params

    s_calib: Dict[str, np.ndarray] = {}
    conf: Dict[str, np.ndarray] = {}

    for v in view_names:
        s = np.asarray(scores_dict[v], dtype=np.float64)
        med = params[v]["med"]
        mad = params[v]["mad"]
        z = _robust_zscore(s, med, mad).astype(np.float32)
        s_calib[v] = z

        absz = np.abs(z).astype(np.float32)

        absz = np.minimum(absz, np.float32(1e6))
        c = absz / (1.0 + absz)
        conf[v] = c.astype(np.float32)

    if return_conf:
        return s_calib, conf
    return s_calib
