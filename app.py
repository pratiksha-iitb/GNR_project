"""
GNR607 - Decorrelation Contrast Enhancement and K-Means Clustering
Streamlit front-end.   Run with:   streamlit run app.py
"""
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data as skdata

import pipeline as pl

st.set_page_config(page_title="Decorrelation + K-Means", layout="wide")

D = pl.DEFAULTS

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def fig_pc_hist(pc_before, pc_after):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        both = np.concatenate([pc_before[:, i], pc_after[:, i]])
        bins = np.linspace(both.min(), both.max(), 101)
        ax[i].hist(pc_before[:, i], bins=bins, color="royalblue", alpha=0.55,
                   label="Before", edgecolor="black", linewidth=0.3)
        ax[i].hist(pc_after[:, i], bins=bins, color="darkorange", alpha=0.55,
                   label="After", edgecolor="black", linewidth=0.3)
        ax[i].set_title(f"PC{i+1}")
        ax[i].set_xlabel("Principal component value")
        ax[i].set_ylabel("Frequency")
        ax[i].legend()
        ax[i].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_corr(corr_o, corr_d):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for a, c, t in zip(ax, (corr_o, corr_d), ("Original", "Decorrelated")):
        a.imshow(c, vmin=-1, vmax=1, cmap="coolwarm")
        a.set_title(f"{t} RGB correlation")
        a.set_xticks(range(3)); a.set_yticks(range(3))
        a.set_xticklabels("RGB"); a.set_yticklabels("RGB")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    return fig


def fig_dbi(res):
    df, k = res["dbi"], res["k"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["K"], df["Davies-Bouldin Index"], marker="o")
    ax.axvline(k, linestyle="--", label=f"Optimal K = {k}")
    ax.set_title("Decorrelated image")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Davies-Bouldin index")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    return fig


def fig_sizes(run, k):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, key, c, t in ((ax[0], "sizes_o", "royalblue", "Original"),
                         (ax[1], "sizes_d", "darkorange", "Decorrelated")):
        s = run[key]
        a.bar(s["Cluster"], s["Percentage (%)"], color=c)
        a.set_title(f"{t} - K={k}")
        a.set_xlabel("Cluster"); a.set_ylabel("% of pixels")
        a.set_xticks(s["Cluster"]); a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def build_zip(res, figs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("original.png", png_bytes(res["img"]))
        z.writestr("decorrelated.png", png_bytes(res["dcs"]))
        for name, b in figs.items():
            z.writestr(name, b)
        k, r = res["k"], res["run"]
        z.writestr(f"K{k}/labelmap_original.png", png_bytes(r["map_o"]))
        z.writestr(f"K{k}/labelmap_decorrelated.png", png_bytes(r["map_d"]))
        z.writestr(f"K{k}/segmentation_original.png", png_bytes(r["seg_o"]))
        z.writestr(f"K{k}/segmentation_decorrelated.png", png_bytes(r["seg_d"]))
        z.writestr(f"K{k}/difference_map.png", png_bytes(r["diff"]))
        z.writestr("cluster_sizes.csv", res["cluster_sizes"].to_csv(index=False))
        z.writestr("clustering_quality_metrics.csv", res["metrics"].to_csv(index=False))
        z.writestr("dbi_decorrelated.csv", res["dbi"].to_csv(index=False))
        z.writestr("parameters.csv",
                   pd.DataFrame(list(res["params"].items()),
                                columns=["Parameter", "Value"]).to_csv(index=False))
    return buf.getvalue()


def reset_defaults():
    for key, val in D.items():
        st.session_state[f"p_{key}"] = val


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
for key, val in D.items():
    st.session_state.setdefault(f"p_{key}", val)

with st.sidebar:
    st.title("Inputs")
    uploaded = st.file_uploader("Input image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"])
    use_sample = st.checkbox("Use built-in sample image instead", value=False)

    st.info("K is chosen automatically: the Davies-Bouldin Index is searched on the "
            "decorrelated image, and both images are clustered with that K.")

    with st.expander("Advanced settings"):
        st.number_input("Max image side (px)", 64, 2000, step=64, key="p_max_side",
                        help="Larger images are down-scaled. Bigger = slower.")
        st.number_input("K-Means initialisations (n_init)", 1, 50, key="p_n_init")
        st.number_input("K-Means max iterations", 10, 1000, key="p_max_iter",
                        help="K-Means has no epochs; this is the iteration cap.")
        st.number_input("Random seed", 0, 1_000_000, key="p_seed")
        st.number_input("Sample size for DBI / metrics (pixels)", 1000, 100000,
                        step=1000, key="p_sample_size")
        c1, c2 = st.columns(2)
        c1.number_input("K min", 2, 49, key="p_k_min")
        c2.number_input("K max", 3, 50, key="p_k_max")
        st.checkbox("Auto-extend K range if best K is at the edge", key="p_auto_extend")
        st.button("Reset to defaults", on_click=reset_defaults)

    run = st.button("Run analysis", type="primary", width="stretch")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("Decorrelation contrast enhancement and K-Means clustering")
st.caption("How does decorrelating an RGB image change the clusters K-Means finds? "
           "Upload an image, keep the defaults, and press Run.")

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if run:
    p = {k: st.session_state[f"p_{k}"] for k in D}
    if p["k_min"] >= p["k_max"]:
        st.error("K min must be smaller than K max.")
        st.stop()
    if uploaded is None and not use_sample:
        st.error("Please upload an image (or tick the sample-image box).")
        st.stop()

    try:
        source = skdata.astronaut() if (uploaded is None) else uploaded
        if uploaded is None:
            tmp = io.BytesIO()
            Image.fromarray(source).save(tmp, format="PNG"); tmp.seek(0)
            source = tmp
        img = pl.load_image(source, p["max_side"])
    except Exception as e:
        st.error(f"Could not read the image: {e}")
        st.stop()

    with st.status("Running...", expanded=True) as status:
        try:
            res = pl.run_all(img, p, log=lambda m: st.write(m))
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Pipeline error: {e}")
            st.stop()
        status.update(label="Finished", state="complete", expanded=False)

    figs = {"pc_histograms.png": fig_pc_hist(res["pc_before"], res["pc_after"]),
            "rgb_correlation.png": fig_corr(res["corr_o"], res["corr_d"]),
            "dbi_vs_k.png": fig_dbi(res)}
    res["figs"] = figs
    res["zip"] = build_zip(res, {n: fig_bytes(f) for n, f in figs.items()})
    st.session_state["res"] = res

# ---------------------------------------------------------------
# Results
# ---------------------------------------------------------------
res = st.session_state.get("res")
if res is None:
    st.info("No results yet. Choose an image in the sidebar and press **Run analysis**.")
    st.stop()

h, w = res["shape"]
st.write(f"Image: **{w} x {h}** px ({h*w:,} pixels) | "
         f"Optimal K (from decorrelated image): **{res['k']}**")

tabs = st.tabs(["1. Enhancement", "2. Choosing K", "3. Cluster maps",
                "4. Comparison", "5. Download"])

# ---- Tab 1 ----
with tabs[0]:
    c1, c2 = st.columns(2)
    c1.image(res["img"], caption="Original", width="stretch")
    c2.image(res["dcs"], caption="Decorrelated", width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg |RGB correlation|, original", f"{res['avg_corr_o']:.4f}")
    m2.metric("Avg |RGB correlation|, decorrelated", f"{res['avg_corr_d']:.4f}",
              delta=f"{(res['avg_corr_d']/res['avg_corr_o']-1)*100:.1f}%", delta_color="off")
    m3.metric("Pixels clipped to 0-255", f"{res['info']['clipped_any_pct']:.1f}%")
    st.caption("Correlation is not exactly zero after enhancement because the final image is "
               "clipped to 0-255 and rounded to 8 bit, which breaks the exact rotation.")

    info = res["info"]
    ev = info["eigenvalues"]
    pc_table = pd.DataFrame({
        "Direction (R, G, B)": [", ".join(f"{v:.2f}" for v in info["eigenvectors"][:, i]) for i in range(3)],
        "Variance share (%)": ev / ev.sum() * 100,
        "Std before": info["pc_std_before"],
        "Scale factor": info["scale_factors"],
        "Std after": info["pc_std_after"],
    }, index=["PC1", "PC2", "PC3"])
    st.subheader("Principal components")
    st.dataframe(pc_table.style.format({"Variance share (%)": "{:.1f}", "Std before": "{:.2f}",
                                        "Scale factor": "{:.2f}", "Std after": "{:.2f}"}),
                 width="stretch")

    c1, c2 = st.columns([1, 1.4])
    c1.pyplot(res["figs"]["rgb_correlation.png"])
    c2.pyplot(res["figs"]["pc_histograms.png"])

    with st.expander("Covariance matrices"):
        rgb, pcs = ["R", "G", "B"], ["PC1", "PC2", "PC3"]
        for title, key, lab in (("RGB covariance, original", "cov_rgb_original", rgb),
                                ("PC covariance, before stretch", "cov_pc_before", pcs),
                                ("PC covariance, after stretch", "cov_pc_after", pcs),
                                ("RGB covariance, after stretch (before clipping)", "cov_rgb_transformed", rgb),
                                ("RGB covariance, final 8-bit image", "cov_rgb_final", rgb)):
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(info[key], index=lab, columns=lab).round(3))

# ---- Tab 2 ----
with tabs[1]:
    st.metric("Optimal K (decorrelated image)", res["k"],
              help=f"Davies-Bouldin index = {res['best_dbi']:.4f}")
    st.caption("The Davies-Bouldin search runs on the decorrelated image only. "
               "Both images are then clustered with this K.")
    st.pyplot(res["figs"]["dbi_vs_k.png"])
    with st.expander("DBI values"):
        st.dataframe(res["dbi"].round(4), width="stretch")

# ---- Tab 3 ----
with tabs[2]:
    k, r = res["k"], res["run"]
    st.caption("Cluster colours are labels only and are not matched between the two images.")
    c1, c2 = st.columns(2)
    c1.image(r["map_o"], caption=f"Original: cluster map (K={k})", width="stretch")
    c2.image(r["map_d"], caption=f"Decorrelated: cluster map (K={k})", width="stretch")
    c1.image(r["seg_o"], caption=f"Original: colour segmentation (K={k})", width="stretch")
    c2.image(r["seg_d"], caption=f"Decorrelated: colour segmentation (K={k})", width="stretch")

# ---- Tab 4 ----
with tabs[3]:
    k, r = res["k"], res["run"]
    c1, c2 = st.columns([1, 1.3])
    c1.image(r["diff"], caption="Black = same cluster (after label matching), white = changed",
             width="stretch")
    c1.write(f"**{r['pct_same']:.2f}%** same, **{r['pct_changed']:.2f}%** changed")
    c2.pyplot(fig_sizes(r, k))

    st.subheader("Clustering quality")
    st.dataframe(res["metrics"].style.format({
        "Silhouette Score": "{:.4f}", "Davies-Bouldin Index": "{:.4f}",
        "Calinski-Harabasz Index": "{:.2f}"}), width="stretch", hide_index=True)
    st.caption(f"Metrics use the same {min(res['params']['sample_size'], h*w):,} sampled pixels, "
               "each image in its own feature space. Silhouette and Calinski-Harabasz: higher is better. "
               "Davies-Bouldin: lower is better.")

    m = res["metrics"]
    o = m[m["Image Version"] == "Original"].iloc[0]
    d = m[m["Image Version"] == "Decorrelated"].iloc[0]
    st.markdown("**Which image scores better**")
    st.dataframe(pd.DataFrame([{
        "K": k,
        "Silhouette": "Original" if o["Silhouette Score"] > d["Silhouette Score"] else "Decorrelated",
        "Davies-Bouldin": "Original" if o["Davies-Bouldin Index"] < d["Davies-Bouldin Index"] else "Decorrelated",
        "Calinski-Harabasz": "Original" if o["Calinski-Harabasz Index"] > d["Calinski-Harabasz Index"] else "Decorrelated",
    }]), hide_index=True)

    with st.expander("Cluster size table"):
        st.dataframe(res["cluster_sizes"].round(2), width="stretch", hide_index=True)

# ---- Tab 5 ----
with tabs[4]:
    st.download_button("Download everything (ZIP)", res["zip"], file_name="gnr607_results.zip",
                       mime="application/zip", type="primary")
    st.download_button("Decorrelated image (PNG)", png_bytes(res["dcs"]),
                       file_name="decorrelated.png", mime="image/png")
    st.download_button("Quality metrics (CSV)", res["metrics"].to_csv(index=False),
                       file_name="clustering_quality_metrics.csv", mime="text/csv")
    st.caption("The ZIP holds the original and decorrelated images, all figures, label maps, "
               "colour segmentations, difference maps, and CSVs for metrics, cluster sizes, DBI and parameters.")
