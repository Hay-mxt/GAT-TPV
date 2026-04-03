# GAT-TPV: Consensus-Driven Multi-View One-Class Framework

This repository provides a **minimal implementation** of the core methodology proposed in the paper:

**“GAT-TPV: A Consensus-Driven Multi-View One-Class Framework for Network Traffic Anomaly Detection”**

---

## Overview

GAT-TPV is a multi-view anomaly detection framework designed for **normal-only training** in open environments.
It combines three complementary views of traffic data:

**Static view (AE)** — feature-level reconstruction
**Temporal view (RNN)** — sequential dependency modeling
**Structural view (GNN)** — interaction topology modeling

The framework introduces a **consensus-driven verification pipeline** to improve robustness under distribution shift and achieve low false-positive detection.

---

## Core Pipeline

The method follows a four-stage pipeline:

```
AE / RNN / GNN → anomaly scores
        ↓
Robust Alignment (GAML)
        ↓
Adaptive Thresholding (A-EVT)
        ↓
Consensus Verification (TPCV)
        ↓
Tri-state Output (KNOWN / ALERT / ANOMALY)
```

Additionally, samples labeled as **ANOMALY** can be clustered into fine-grained unknown groups (UNK).

---

## Repository Structure

```
├── models/
│   ├── ae_oc.py        # Autoencoder-based one-class model (static view)
│   ├── rnn_oc.py       # RNN-based one-class model (temporal view)
│   └── gnn_oc.py       # GNN-based one-class model (structural view)
│
├── core/
│   ├── gaml.py         # Robust score alignment + confidence estimation
│   ├── a_evt.py        # Adaptive EVT-inspired thresholding
│   ├── tpcv.py         # k-of-N consensus verification
│   └── unk_cluster.py  # Offline clustering for unknown patterns
```

---

## Key Components

### 1. Multi-View One-Class Models

Each model independently learns normal patterns and produces anomaly scores:

* Reconstruction error (AE, GNN)
* Sequence deviation (RNN)

---

### 2. GAML (Gated Alignment & Multi-view Reliability)

* Aligns anomaly scores across views using **median and MAD**
* Computes confidence:

[
c_v(x) = \frac{|s_v(x)|}{1 + |s_v(x)|}
]

---

### 3. A-EVT (Adaptive Thresholding)

* Uses high-quantile statistics for low-FPR operation
* Dynamically adjusts thresholds based on confidence

---

### 4. TPCV (Consensus Verification)

* Applies **k-of-N rule** across views:

  * `KNOWN`: no view detects anomaly
  * `ALERT`: weak consensus
  * `ANOMALY`: strong consensus (≥ k views)

---

### 5. UNK Clustering (Optional)

* Clusters confirmed anomalies into unknown pattern groups
* Supports DBSCAN / K-Means

---

## Usage (Minimal Example)

Since this repository focuses on **core methodology**, users need to provide:

* Input features `x`
* Pretrained or trained models

Example workflow:

Step 1: get anomaly scores from each model

Step 2: align scores and compute confidence

Step 3: adaptive thresholds

Step 4: consensus decision

---

## Notes on Reproducibility

This repository **does NOT include data preprocessing pipelines** (e.g., pcap parsing, flow construction).
It focuses on the **core algorithmic components** of GAT-TPV.
Users can integrate their own datasets and feature extraction methods.

---

## Requirements

Python 3.8+

NumPy / SciPy / scikit-learn

---


## Disclaimer

This is a minimal and research-oriented implementation.
Some engineering details (e.g., large-scale data processing, deployment optimizations) are omitted for clarity.

---
