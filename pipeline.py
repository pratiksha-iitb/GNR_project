"""
Core pipeline for the GNR607 project:
Decorrelation contrast enhancement + K-Means clustering comparison.

All functions are taken from the original notebook (GNR607_test_.ipynb),
with Colab-specific parts removed and a few parameters exposed.
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage import transform
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------
# Default parameters (same values as the notebook)
# --------------------------------------------------------------
DEFAULTS = {
    "max_side": 512,       # max image dimension (px)
    "n_init": 10,          # K-Means initialisations
    "max_iter": 300,       # K-Means max iterations
    "seed": 42,            # random seed
    "sample_size": 10000,  # pixels used for DBI search + quality metrics
    "k_min": 2,            # DBI search start
    "k_max": 20,           # DBI search end
    "manual_k": 6,         # used only in manual-K mode
    "auto_extend": True,   # extend search by +10 (up to 50) if best K is at the upper edge
}
ABSOLUTE_MAX_K = 50


# --------------------------------------------------------------
# 1. Image loading
# --------------------------------------------------------------
def load_image(source, max_side=512):
    """Load a path / file-like object as an 8-bit RGB array, resized so that
    the longer side is at most `max_side` (aspect ratio preserved)."""
    im = Image.open(source)
    im = ImageOps.exif_transpose(im)

    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.array(im).astype(np.float64)
        arr = arr / max(arr.max(), 1.0) * 255.0
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = np.array(im.convert("RGB"))

    img = np.clip(arr, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = transform.resize(
            img, (int(h * scale), int(w * scale)),
            preserve_range=True, anti_aliasing=True,
        )
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# --------------------------------------------------------------
# 2. PCA-based decorrelation stretch
# --------------------------------------------------------------
def decorrelation_stretch(img):
    h, w, _ = img.shape
    X = img.reshape(-1, 3).astype(np.float64)

    mu = X.mean(axis=0)
    cov_rgb_original = np.cov(X, rowvar=False)

    eigval, eigvec = np.linalg.eigh(cov_rgb_original)
    order = np.argsort(eigval)[::-1]
    eigval = np.maximum(eigval[order], 1e-12)
    eigvec = eigvec[:, order]

    pc_before = (X - mu) @ eigvec
    cov_pc_before = np.cov(pc_before, rowvar=False)
    pc_std_before = np.sqrt(eigval)

    scale_factors = np.array([
        1.0,
        np.sqrt(eigval[0] / eigval[1]),
        np.sqrt(eigval[0] / eigval[2]),
    ])
    pc_after = pc_before * scale_factors
    cov_pc_after = np.cov(pc_after, rowvar=False)
    pc_std_after = pc_after.std(axis=0)

    Y = pc_after @ eigvec.T + mu
    cov_rgb_transformed = np.cov(Y, rowvar=False)

    outside = (Y < 0) | (Y > 255)
    clipped_any_pct = float(outside.any(axis=1).mean() * 100)
    clipped_channel_pct = (outside.mean(axis=0) * 100).tolist()

    Y_final = np.clip(Y, 0, 255).astype(np.uint8)
    cov_rgb_final = np.cov(Y_final, rowvar=False)
    Y_final = Y_final.reshape(h, w, 3)

    info = {
        "mean": mu,
        "eigenvalues": eigval,
        "eigenvectors": eigvec,
        "scale_factors": scale_factors,
        "pc_std_before": pc_std_before,
        "pc_std_after": pc_std_after,
        "cov_rgb_original": cov_rgb_original,
        "cov_rgb_transformed": cov_rgb_transformed,
        "cov_rgb_final": cov_rgb_final,
        "cov_pc_before": cov_pc_before,
        "cov_pc_after": cov_pc_after,
        "clipped_any_pct": clipped_any_pct,
        "clipped_channel_pct": clipped_channel_pct,
    }
    return Y_final, info, pc_before, pc_after


def average_absolute_correlation(corr):
    return float(np.mean(np.abs(corr[np.triu_indices(3, k=1)])))


# --------------------------------------------------------------
# 3. K-Means helpers
# --------------------------------------------------------------
def run_kmeans(X, k, n_init, max_iter, seed):
    km = KMeans(
        n_clusters=k, n_init=n_init, max_iter=max_iter,
        random_state=seed, algorithm="lloyd",
    )
    labels = km.fit_predict(X)
    return labels, km


def find_optimal_k(X_sample, name, k_min, k_max, n_init, max_iter, seed,
                   auto_extend=True, log=None):
    """Davies-Bouldin search. Same logic as the notebook: if the minimum lands
    on the upper boundary, the search is extended by 10 (up to K = 50)."""
    k_max = min(k_max, len(X_sample) - 1)
    abs_max = max(ABSOLUTE_MAX_K, k_max)
    ks, dbis = [], []
    cur_max, next_k = k_max, k_min

    while True:
        for k in range(next_k, cur_max + 1):
            if log:
                log(f"{name}: testing K = {k}")
            labels, _ = run_kmeans(X_sample, k, n_init, max_iter, seed)
            ks.append(k)
            dbis.append(davies_bouldin_score(X_sample, labels))
        next_k = cur_max + 1

        best_k = ks[int(np.argmin(dbis))]
        if auto_extend and best_k == cur_max and cur_max < abs_max:
            cur_max = min(cur_max + 10, abs_max)
            if log:
                log(f"{name}: minimum at upper edge, extending to K = {cur_max}")
        else:
            break

    i = int(np.argmin(dbis))
    df = pd.DataFrame({"K": ks, "Davies-Bouldin Index": dbis})
    return ks[i], dbis[i], df


# --------------------------------------------------------------
# 4. Comparison helpers
# --------------------------------------------------------------
def create_cluster_difference_map(lab_o, lab_d, k, shape):
    """Hungarian label matching, then black = same cluster, white = changed."""
    contingency = np.bincount(lab_o * k + lab_d, minlength=k * k).reshape(k, k)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = np.zeros(k, dtype=np.int64)
    mapping[col_ind] = row_ind
    same = lab_o == mapping[lab_d]
    diff_map = np.where(same, 0, 255).astype(np.uint8).reshape(shape)
    pct_same = float(same.mean() * 100)
    return diff_map, pct_same, 100 - pct_same


def create_segmented_image(labels, km, shape):
    centres = km.cluster_centers_ * 255
    seg = centres[labels].reshape(*shape, 3)
    return np.clip(seg, 0, 255).astype(np.uint8)


def label_map_rgb(labels, k, shape):
    """Colour a label map with nipy_spectral (same colormap as the notebook)."""
    import matplotlib
    cmap = matplotlib.colormaps["nipy_spectral"]
    norm = labels / max(k - 1, 1)
    return (cmap(norm)[:, :3] * 255).astype(np.uint8).reshape(*shape, 3)


def calculate_cluster_sizes(labels, k, image_name):
    counts = np.bincount(labels, minlength=k)
    return pd.DataFrame({
        "Image Version": image_name,
        "K": k,
        "Cluster": np.arange(k),
        "Pixel Count": counts,
        "Percentage (%)": counts / len(labels) * 100,
    })


def evaluate_clustering(X, labels, indices, image_name, k):
    Xs, ls = X[indices], labels[indices]
    return {
        "Image Version": image_name,
        "K": k,
        "Silhouette Score": silhouette_score(Xs, ls),
        "Davies-Bouldin Index": davies_bouldin_score(Xs, ls),
        "Calinski-Harabasz Index": calinski_harabasz_score(Xs, ls),
    }


# --------------------------------------------------------------
# 5. Full pipeline
# --------------------------------------------------------------
def run_all(img, params, k_mode="auto", log=None):
    """
    k_mode = "auto"   : optimal K is found with the Davies-Bouldin Index on the
                        DECORRELATED (enhanced) image only.
    k_mode = "manual" : K is taken from params["manual_k"]; DBI search skipped.
    In both modes the original and decorrelated image are clustered with the
    same K, so the comparison is like for like.

    params : dict with keys of DEFAULTS
    """
    p = {**DEFAULTS, **params}
    h, w, _ = img.shape
    say = log or (lambda m: None)

    say("Applying decorrelation stretch...")
    dcs, info, pc_before, pc_after = decorrelation_stretch(img)

    corr_o = np.corrcoef(img.reshape(-1, 3).astype(np.float64), rowvar=False)
    corr_d = np.corrcoef(dcs.reshape(-1, 3).astype(np.float64), rowvar=False)

    X_o = img.reshape(-1, 3).astype(np.float64) / 255.0
    X_d = dcs.reshape(-1, 3).astype(np.float64) / 255.0

    rng = np.random.default_rng(p["seed"])
    n_sample = min(p["sample_size"], len(X_o))
    idx = rng.choice(len(X_o), size=n_sample, replace=False)

    # ---- choose K ----
    if k_mode == "auto":
        k, best_dbi, df_dbi = find_optimal_k(
            X_d[idx], "Decorrelated", p["k_min"], p["k_max"], p["n_init"],
            p["max_iter"], p["seed"], p["auto_extend"], say)
    else:
        k, best_dbi, df_dbi = int(p["manual_k"]), None, None

    res = {
        "img": img, "dcs": dcs, "info": info,
        "pc_before": pc_before, "pc_after": pc_after,
        "corr_o": corr_o, "corr_d": corr_d,
        "avg_corr_o": average_absolute_correlation(corr_o),
        "avg_corr_d": average_absolute_correlation(corr_d),
        "params": p, "shape": (h, w), "k_mode": k_mode,
        "k": k, "best_dbi": best_dbi, "dbi": df_dbi,
    }

    # ---- cluster both images at that K ----
    say(f"Clustering both images at K = {k}...")
    lab_o, km_o = run_kmeans(X_o, k, p["n_init"], p["max_iter"], p["seed"])
    lab_d, km_d = run_kmeans(X_d, k, p["n_init"], p["max_iter"], p["seed"])
    diff, same, changed = create_cluster_difference_map(lab_o, lab_d, k, (h, w))

    res["run"] = {
        "lab_o": lab_o, "lab_d": lab_d,
        "map_o": label_map_rgb(lab_o, k, (h, w)),
        "map_d": label_map_rgb(lab_d, k, (h, w)),
        "seg_o": create_segmented_image(lab_o, km_o, (h, w)),
        "seg_d": create_segmented_image(lab_d, km_d, (h, w)),
        "diff": diff, "pct_same": same, "pct_changed": changed,
        "sizes_o": calculate_cluster_sizes(lab_o, k, "Original"),
        "sizes_d": calculate_cluster_sizes(lab_d, k, "Decorrelated"),
    }
    res["cluster_sizes"] = pd.concat(
        [res["run"]["sizes_o"], res["run"]["sizes_d"]], ignore_index=True)
    res["metrics"] = pd.DataFrame([
        evaluate_clustering(X_o, lab_o, idx, "Original", k),
        evaluate_clustering(X_d, lab_d, idx, "Decorrelated", k),
    ])
    say("Done.")
    return res
