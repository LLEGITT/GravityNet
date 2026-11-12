# GravityNet-3D: Trainable Gravity Collision-Graph Clustering

A lightweight physics-inspired neural architecture (**6,200 parameters**) for 3D point cloud part segmentation via learned dynamical systems and collision-graph clustering.

![Results Overview](results.png)

## Overview

Instead of relying on heavy point-based backbones (e.g., PointNet, DGCNN), **GravityNet-3D** learns a bank of $T = 100$ parallel harmonic potential fields ("terrains"). Input 3D points slide through each terrain under damped Newtonian dynamics, settling into part-consistent attractors. Pairwise soft-collision affinities are then aggregated across all terrains and partitioned via spectral clustering with automatic cluster count estimation (eigengap heuristic — **no oracle $k$ required**).

### Key Highlights
* **Ultra-compact:** Only **6,200 trainable parameters** (frequencies, amplitudes, phases, plus per-terrain collision radius $r$ and temperature $\tau$).
* **Oracle-free inference:** Automatically estimates the number of parts $k$ per shape via the normalized Laplacian eigengap, yet outperforms baselines given the ground-truth $k$.
* **Strong generalization:** Trained on 150 chairs for 10 epochs and evaluated on 200 unseen test chairs from **ShapeNetPart** (`03001627`).

---

## Method

1. **Learnable Potential Fields:** Each terrain $t \in \{1, \dots, T\}$ defines a smooth potential $V_t(\mathbf{x})$ parameterized by $B = 12$ sinusoidal basis functions.
2. **Damped Particle Simulation:** Points evolve for $S = 6$ steps ($\Delta t = 0.15$, friction $\gamma = 0.75$):
   $$\mathbf{v}^{(s+1)} = \gamma \mathbf{v}^{(s)} - \Delta t \tanh(\nabla_{\mathbf{x}} V_t(\mathbf{x}^{(s)}))$$
   $$\mathbf{x}^{(s+1)} = \mathbf{x}^{(s)} + \Delta t \mathbf{v}^{(s+1)}$$
3. **Soft Collision Affinity:** After simulation, pairwise Euclidean distances $D_t$ are converted into collision probabilities using learned positive parameters $(r_t, \tau_t)$ and averaged across terrains:
   $$A = \frac{1}{T} \sum_{t=1}^{T} \sigma\left(\frac{r_t - D_t}{\tau_t}\right)$$
4. **Graph Partitioning:** A mutual $k$-NN graph ($k = 15$) prunes spurious long-range edges, and Spectral Clustering extracts the final part labels.

---

## Benchmark Results (ShapeNetPart Chairs)

Evaluated on **200 unseen test chairs** (512 points per shape, 3D coordinates only). Mean $\pm$ standard error across shapes:

| Method | Oracle $k$? | ARI $\uparrow$ | NMI $\uparrow$ | Purity $\uparrow$ | Avg. #Clusters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gravity-100 (Ours)** | **No** (auto) | **0.578 ± 0.011** | **0.632 ± 0.008** | **0.872 ± 0.006** | 4.4 |
| K-Means | Yes | 0.372 ± 0.010 | 0.439 ± 0.010 | 0.722 ± 0.007 | 3.3 |
| Ward Hierarchical | Yes | 0.358 ± 0.010 | 0.432 ± 0.009 | 0.716 ± 0.007 | 3.3 |

* **Paired ARI gain vs. K-Means:** `+0.207 ± 0.011` (GravityNet wins on **94%** of test chairs, 0% ties).
* **Paired ARI gain vs. Ward:** `+0.221 ± 0.012` (GravityNet wins on **90%** of test chairs).
* **Learned Affinity Separation (held-out chair):** Same-part mean affinity = `0.891` vs. different-part = `0.374`.
* **Learned Physical Parameters:** Mean collision radius $r = 0.9181$ (`[0.1598, 2.5749]`), mean temperature $\tau = 0.1353$ (`[0.0096, 2.1045]`).

---

## Quickstart

### Requirements
```bash
pip install torch numpy scipy scikit-learn matplotlib