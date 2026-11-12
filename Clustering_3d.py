"""
Trainable 100-layer Gravity Collision-Graph Clustering (FULL 3D + normals) vs KMeans on ShapeNetPart chairs.
Data: ShapeNetPart "normal" (folder 03001627, .txt files: x y z nx ny nz label).
"""
import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------
DATA_DIR = None         # None = auto-search in /kaggle/input, ./data, .
TRAIN_SHAPES = 150      # Chairs to train terrains + (r, tau)
TEST_SHAPES = 200       # Chairs for final comparison vs KMeans
EPOCHS = 10
LR = 0.005
LR_RTAU = 0.03   # faster learning rate for radius / temperature

# Model saving / loading
MODEL_PATH = "/kaggle/working/gravitynet_3d.pt" if os.path.isdir("/kaggle/working") else "gravitynet_3d.pt"
LOAD_MODEL = False   # True = skip training and load the weights saved at MODEL_PATH

N_POINTS = 512          # Fast evaluation on GPU / CPU
NUM_TERRAINS = 100      # Exactly 100 terrain layers
STEPS = 6               # Simulation steps per terrain
DT = 0.15
FRICTION = 0.75
MIN_SIZE = 12

USE_NORMALS = False       # append surface normals (nx, ny, nz) as 3 extra coordinates -> 6D features
NORMAL_W = 0.5            # weight of normals relative to positions (positions live in [-1, 1])
FEAT_DIM = 6 if USE_NORMALS else 3
KMEANS_WITH_NORMALS = True  # fair baseline: KMeans sees exactly the same features as gravity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def find_chair_dir():
    if DATA_DIR and os.path.isdir(DATA_DIR):
        return DATA_DIR
    for root in ["/kaggle/input", "./data", "."]:
        for dirpath, dirnames, _ in os.walk(root):
            if "03001627" in dirnames:
                return os.path.join(dirpath, "03001627")
    raise FileNotFoundError("Folder 03001627 (Chair) not found: set DATA_DIR.")


_RAW = {}


def load_shape(path):
    if path not in _RAW:
        _RAW[path] = np.loadtxt(path)   # parsed once, reused every epoch
    arr = _RAW[path]
    pos, nrm, y = arr[:, :3], arr[:, 3:6], arr[:, -1].astype(int)
    if len(pos) > N_POINTS:
        idx = np.random.permutation(len(pos))[:N_POINTS]
        pos, nrm, y = pos[idx], nrm[idx], y[idx]
    pos = pos - pos.mean(axis=0, keepdims=True)
    pos = pos / np.abs(pos).max()
    if USE_NORMALS:
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)   # unit normals
        feats = np.concatenate([pos, NORMAL_W * nrm], axis=1)            # (N, 6)
    else:
        feats = pos                                                       # (N, 3)
    return torch.tensor(feats, dtype=torch.float32), y


# ----------------------------------------------------------------------
# 2. 100 TRAINABLE TERRAINS WITH LINKED (r, tau)
# ----------------------------------------------------------------------
def inv_softplus(y):
    return float(np.log(np.expm1(y)))


R_MIN, TAU_MIN = 0.02, 0.005     # only LOWER bounds (avoid division by zero); no upper bound
R_INIT, TAU_INIT = 0.15, 0.05


class Gravity100Network(nn.Module):
    """
    Vectorized bank of T terrains: all parameters carry a leading T dimension and the
    whole simulation runs as batched tensor ops (no Python loop over terrains).
    r and tau use softplus: strictly positive and UNBOUNDED above, with a non-dead gradient.
    """
    def __init__(self, num_terrains=NUM_TERRAINS, num_bases=12, dt=DT, friction=FRICTION):
        super().__init__()
        self.dt, self.friction = dt, friction
        T = num_terrains
        self.freqs = nn.Parameter(torch.randn(T, num_bases, FEAT_DIM) * 2.0)
        self.amps = nn.Parameter(torch.randn(T, num_bases) * 0.5)
        self.phases = nn.Parameter(torch.rand(T, num_bases) * 2.0 * np.pi)
        # raw parameters (names kept so the optimizer groups still match "log_r" / "log_tau")
        self.log_r = nn.Parameter(torch.full((T,), inv_softplus(R_INIT - R_MIN)))
        self.log_tau = nn.Parameter(torch.full((T,), inv_softplus(TAU_INIT - TAU_MIN)))

    def get_r(self):
        return R_MIN + F.softplus(self.log_r)        # shape (T,)

    def get_tau(self):
        return TAU_MIN + F.softplus(self.log_tau)    # shape (T,)

    def gradient(self, xyz):                          # xyz: (T, N, 3)
        proj = torch.einsum("tnd,tbd->tnb", xyz, self.freqs) + self.phases.unsqueeze(1)
        w = torch.cos(proj) * self.amps.unsqueeze(1)
        return torch.einsum("tnb,tbd->tnd", w, self.freqs)

    def forward(self, pts3):                          # pts3: (N, 3)
        T = self.freqs.shape[0]
        xyz = pts3.unsqueeze(0).expand(T, -1, -1).contiguous()
        v = torch.zeros_like(xyz)
        for _ in range(STEPS):
            g = torch.tanh(self.gradient(xyz))
            v = self.friction * v - self.dt * g
            xyz = xyz + self.dt * v
        D = torch.cdist(xyz, xyz)                     # (T, N, N)
        r = self.get_r().view(T, 1, 1)
        tau = self.get_tau().view(T, 1, 1)
        return torch.sigmoid((r - D) / tau).mean(dim=0)


# ----------------------------------------------------------------------
# 3. TRAINING & GRAPH CLUSTERING
# ----------------------------------------------------------------------
def contrastive_loss(A_pred, true_labels, margin=0.15):
    """
    Prevents collapse to 1 or N clusters:
    - Same-part pairs should have affinity close to 1.
    - Different-part pairs should be pushed below 'margin'.
    """
    y = torch.as_tensor(true_labels, device=A_pred.device)
    same_part = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff_part = 1.0 - same_part

    # Balanced: positives and negatives each averaged over their own pairs, diagonal excluded.
    eye = torch.eye(len(y), device=A_pred.device)
    same_part = same_part - eye
    diff_part = 1.0 - same_part - eye
    pos_loss = ((1.0 - A_pred).pow(2) * same_part).sum() / same_part.sum().clamp(min=1.0)
    neg_loss = (F.relu(A_pred - margin).pow(2) * diff_part).sum() / diff_part.sum().clamp(min=1.0)
    return pos_loss + neg_loss


def absorb_small(lab, pts, min_size=MIN_SIZE):
    ids, counts = np.unique(lab, return_counts=True)
    big = ids[counts >= min_size]
    if len(big) == 0:
        return np.zeros_like(lab)
    mask = np.isin(lab, big)
    if mask.all():
        return lab
    D = torch.cdist(pts[torch.tensor(~mask)], pts[torch.tensor(mask)])
    lab = lab.copy()
    lab[~mask] = lab[mask][D.argmin(dim=1).numpy()]
    return lab


from sklearn.cluster import SpectralClustering

K_MAX = 6  # max number of parts considered by the eigengap

def gravity_predict(model, pts3, n_clusters="auto"):
    model.eval()
    with torch.no_grad():
        A = model(pts3.to(DEVICE)).cpu().numpy()
    
    # 1. Zero out self-connections
    np.fill_diagonal(A, 0.0)
    
    # 2. Build a mutual k-NN graph from the learned affinity to kill bridging edges
    k = 15  # local neighborhood size
    n = A.shape[0]
    knn_mask = np.zeros_like(A, dtype=bool)
    
    top = np.argpartition(A, -k, axis=1)[:, -k:]
    knn_mask[np.arange(n)[:, None], top] = True
        
    # Symmetrize (mutual or standard k-NN)
    A_sparse = np.where(knn_mask | knn_mask.T, A, 0.0)
    
    # 3. Predict clusters:
    # If oracle k is available (like KMeans), use Spectral Clustering on the affinity
    # Otherwise, run connected components on the sparse graph
    if n_clusters == "auto":
        # Eigengap heuristic on the normalised Laplacian: k is estimated, not given (no oracle).
        deg = A_sparse.sum(axis=1) + 1e-8
        dm = 1.0 / np.sqrt(deg)
        L = np.eye(n) - dm[:, None] * A_sparse * dm[None, :]
        evals = np.linalg.eigvalsh(L)[:K_MAX + 1]
        gaps = np.diff(evals)
        n_clusters = int(np.argmax(gaps[1:K_MAX]) + 2)   # k in [2, K_MAX]

    if n_clusters is not None:
        clustering = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed', 
            assign_labels='kmeans', 
            random_state=0
        )
        lab = clustering.fit_predict(A_sparse + 1e-5)
    else:
        # Connected components on the strictly pruned local graph
        thresh = np.percentile(A_sparse[A_sparse > 0], 50) if np.any(A_sparse > 0) else 0.5
        adj = (A_sparse >= thresh).astype(int)
        _, lab = connected_components(csr_matrix(adj), directed=False)
        lab = absorb_small(lab, pts3)
        
    return lab


def ward_labels(pts, true):
    k = len(np.unique(true))  # Oracle k, same as KMeans
    X = pts.numpy() if KMEANS_WITH_NORMALS else pts.numpy()[:, :3]
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def kmeans_labels(pts, true):
    k = len(np.unique(true))  # Oracle k
    X = pts.numpy() if KMEANS_WITH_NORMALS else pts.numpy()[:, :3]
    return KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)


# ----------------------------------------------------------------------
# 4. METRICS
# ----------------------------------------------------------------------
def purity(pred, true):
    return sum(np.unique(true[pred == c], return_counts=True)[1].max() for c in np.unique(pred)) / len(true)


def metrics(pred, true):
    return (adjusted_rand_score(true, pred), normalized_mutual_info_score(true, pred),
            purity(pred, true), len(np.unique(pred)))


def mean_se(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


# ----------------------------------------------------------------------
# 5. FIGURE (Preserved Layout)
# ----------------------------------------------------------------------
def _plot_cloud(ax, pts, lab, title):
    _, lab = np.unique(lab, return_inverse=True)
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=lab % 20, cmap="tab20", s=6)
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def plot_results(show, ours, km, wd=None):
    pts, true, lab_g, lab_k = show
    pts = pts.numpy()
    ari_o, ari_k = ours[:, 0], km[:, 0]
    d = ari_o - ari_k

    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    _plot_cloud(ax, pts, true, f"Ground truth ({len(np.unique(true))} parts)\n(chair with median ARI difference)")
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    _plot_cloud(ax, pts, lab_g, f"chair with median gravity ARI {'3D+normals' if USE_NORMALS else '3D'} (100 Terrains)\n"
                                f"ARI={adjusted_rand_score(true, lab_g):.2f} | {len(np.unique(lab_g))} clusters")
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    _plot_cloud(ax, pts, lab_k, f"KMeans (k = true k)\n"
                                f"ARI={adjusted_rand_score(true, lab_k):.2f} | {len(np.unique(lab_k))} clusters")

    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(ari_k, ari_o, s=12, alpha=0.5)
    ax.plot([-0.2, 1], [-0.2, 1], "k--", lw=1)
    ax.set_xlabel("KMeans ARI"); ax.set_ylabel("Gravity ARI")
    ax.set_title("Per chair (above the line = gravity better)")

    ax = fig.add_subplot(2, 3, 5)
    ax.hist(d, bins=30, color="steelblue")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(d.mean(), color="red", ls="--", label=f"mean {d.mean():+.3f}")
    ax.set_xlabel("ARI(gravity) - ARI(KMeans)"); ax.set_title(f"Paired difference ({len(d)} chairs)")
    ax.legend()

    ax = fig.add_subplot(2, 3, 6)
    for j, (name, idx) in enumerate([("ARI", 0), ("NMI", 1)]):
        series = [("Gravity-100", ours, "steelblue"), ("KMeans", km, "darkorange")]
        if wd is not None:
            series.append(("Ward", wd, "seagreen"))
        w = 0.8 / len(series)
        for i, (lab, M, c) in enumerate(series):
            m_, se_ = mean_se(M[:, idx])
            ax.bar(j + (i - (len(series) - 1) / 2) * w, m_, w, yerr=se_, color=c, label=lab if j == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ARI", "NMI"])
    ax.set_ylim(0, 1); ax.set_title("Mean on test chairs (±SE)"); ax.legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=120)
    plt.show()


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    chair_dir = find_chair_dir()
    files = sorted(glob.glob(os.path.join(chair_dir, "**", "*.txt"), recursive=True))
    print(f"{len(files)} chairs found in {chair_dir}")

    rng = np.random.RandomState(0)
    rng.shuffle(files)
    train_files = files[:TRAIN_SHAPES]
    test_files = files[-TEST_SHAPES:]   # fixed test set, independent of TRAIN_SHAPES
    print(f"Device: {DEVICE} | Train: {len(train_files)} chairs | Test: {len(test_files)} chairs")

    model = Gravity100Network(num_terrains=NUM_TERRAINS).to(DEVICE)
    print(f"Features: {FEAT_DIM}D | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    rtau = [p for n, p in model.named_parameters() if "log_r" in n or "log_tau" in n]
    others = [p for n, p in model.named_parameters() if not ("log_r" in n or "log_tau" in n)]
    optimizer = torch.optim.Adam([{"params": others, "lr": LR},
                                  {"params": rtau, "lr": LR_RTAU}])

    if LOAD_MODEL:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        assert ckpt["config"]["feat_dim"] == FEAT_DIM, "Checkpoint was trained with a different feature size"
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded model from {MODEL_PATH}")

    # ---- Step 1: Train 100 terrains + 100 (r, tau) ----
    print(f"\n--- Training 100-layer Gravity Model ({EPOCHS} Epochs) ---")
    for epoch in range(0 if LOAD_MODEL else EPOCHS):
        model.train()
        total_loss = 0.0
        for f in train_files:
            pts, true = load_shape(f)
            pts_dev = pts.to(DEVICE)

            optimizer.zero_grad()
            A_pred = model(pts_dev)
            loss = contrastive_loss(A_pred, true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss / len(train_files):.4f} | Elapsed: {time.time()-t0:.0f}s")
        torch.save(model.state_dict(), "ckpt_last.pt")   # safety checkpoint after every epoch

    if not LOAD_MODEL:
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"feat_dim": FEAT_DIM, "num_terrains": NUM_TERRAINS, "steps": STEPS,
                       "dt": DT, "friction": FRICTION, "use_normals": USE_NORMALS,
                       "normal_w": NORMAL_W, "train_shapes": TRAIN_SHAPES, "epochs": EPOCHS},
        }, MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")

    # Inspect trained parameter distributions
    r_vals = model.get_r().tolist()
    tau_vals = model.get_tau().tolist()
    print(f"\nTrained radius r: mean={np.mean(r_vals):.4f} [min={min(r_vals):.4f}, max={max(r_vals):.4f}]")
    print(f"Trained tau:      mean={np.mean(tau_vals):.4f} [min={min(tau_vals):.4f}, max={max(tau_vals):.4f}]")

    # Diagnostic: affinity separation on one held-out chair
    pts_d, true_d = load_shape(test_files[0])
    model.eval()
    with torch.no_grad():
        A_d = model(pts_d.to(DEVICE)).cpu().numpy()
    y_d = np.asarray(true_d)
    same = (y_d[:, None] == y_d[None, :]) & ~np.eye(len(y_d), dtype=bool)
    print(f"Affinity same-part: {A_d[same].mean():.3f} | diff-part: {A_d[~same & ~np.eye(len(y_d), dtype=bool)].mean():.3f}")

    # ---- Step 2: Evaluate on Test Chairs ----
    print(f"\n--- Evaluating on {len(test_files)} Test Chairs ---")
    np.random.seed(0)   # same point subsampling every evaluation
    ours, km, wd, store = [], [], [], []
    t1 = time.time()
    for i, f in enumerate(test_files):
        pts, true = load_shape(f)
        lg = gravity_predict(model, pts)
        lk = kmeans_labels(pts, true)
        lw = ward_labels(pts, true)

        ours.append(metrics(lg, true))
        km.append(metrics(lk, true))
        wd.append(metrics(lw, true))
        store.append((pts, true))

        if (i + 1) % 25 == 0 or i + 1 == len(test_files):
            el = time.time() - t1
            eta = el / (i + 1) * (len(test_files) - i - 1)
            print(f"[test] {i+1}/{len(test_files)} | elapsed {el:.0f}s | ETA {eta:.0f}s")

    ours, km, wd = np.array(ours), np.array(km), np.array(wd)

    # ---- Step 3: Print Table ----
    print(f"\n=========== 100-LAYER GRAVITY vs KMEANS on {len(test_files)} test chairs (mean ± SE) ===========")
    print(f"{'method':22s} {'ARI':>16s} {'NMI':>16s} {'purity':>16s} {'#clusters':>12s}")
    for name, M in [("gravity-100", ours), ("kmeans (k=oracle)", km), ("ward (k=oracle)", wd)]:
        cells = []
        for j in range(3):
            m, s = mean_se(M[:, j])
            cells.append(f"{m:.3f} ± {s:.3f}")
        print(f"{name:22s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s} {M[:, 3].mean():12.1f}")

    d = ours[:, 0] - km[:, 0]
    m, se = mean_se(d)
    print(f"\nPaired ARI(gravity) - ARI(kmeans): {m:+.3f} ± {se:.3f}")
    print(f"Gravity better: {np.mean(d > 0)*100:.0f}%, KMeans better: {np.mean(d < 0)*100:.0f}%, Ties: {np.mean(d == 0)*100:.0f}%")
    dw = ours[:, 0] - wd[:, 0]
    mw, sew = mean_se(dw)
    print(f"Paired ARI(gravity) - ARI(ward):   {mw:+.3f} ± {sew:.3f}")
    print(f"Gravity better than Ward: {np.mean(dw > 0)*100:.0f}%, Ward better: {np.mean(dw < 0)*100:.0f}%")
    print(f"Total time: {time.time()-t0:.0f}s")

    # ---- Step 4: Render Figure ----
    idx = int(np.argmin(np.abs(ours[:, 0] - np.median(ours[:, 0]))))    
    pts, true = store[idx]
    show = (pts, true, gravity_predict(model, pts), kmeans_labels(pts, true))
    plot_results(show, ours, km, wd)