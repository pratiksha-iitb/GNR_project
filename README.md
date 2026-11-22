# Decorrelation Contrast Enhancement and K-Means Clustering (GNR607)

A small web app that wraps the project notebook. Upload an RGB image and the app:

1. applies PCA-based decorrelation contrast enhancement,
2. picks K with the Davies-Bouldin Index (or uses a K you choose),
3. clusters the original and decorrelated images with K-Means,
4. compares cluster maps, cluster sizes, pixel-level changes and quality metrics
   (Silhouette, Davies-Bouldin, Calinski-Harabasz).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `pipeline.py` | All processing code (functions taken from the notebook) |
| `requirements.txt` | Python dependencies |

## Run it locally

You need Python 3.9 or newer.

```bash
# 1. Go to the project folder
cd gnr607_app

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. If it does not, open that address manually.
Stop the app with `Ctrl + C` in the terminal.

## How to use

**With defaults (quickest):** upload an image in the sidebar and click **Run analysis**.
The "Use built-in sample image" box lets you test without a file.

**With your own settings:** choose Auto or Manual K and open **Advanced settings**.
**Reset to defaults** restores everything.

| Setting | Default | Meaning |
|---|---|---|
| K selection | Auto (Davies-Bouldin) | Auto finds the best K for each image and runs all combinations. Manual uses one K for both images and skips the DBI search (faster). |
| Manual K | 6 | Only used in Manual mode |
| Max image side | 512 px | Larger images are down-scaled |
| n_init | 10 | K-Means random restarts |
| Max iterations | 300 | K-Means iteration cap (K-Means has no epochs) |
| Random seed | 42 | Makes results reproducible |
| Sample size | 10,000 px | Pixels used for the DBI search and quality metrics |
| K min / K max | 2 / 20 | DBI search range |
| Auto-extend | on | If the best K is at K max, the search extends by 10 (up to 50) |

## Output tabs

1. **Enhancement**: original vs decorrelated image, correlation before/after, % clipped pixels, principal components table, correlation heatmaps, PC histograms, covariance matrices.
2. **Choosing K**: DBI vs K curves and the chosen K for each image (Auto mode).
3. **Cluster maps**: label maps and colour segmentations, original vs decorrelated, at each K.
4. **Comparison**: pixel-change map, cluster size charts, quality metrics table, and which image wins each metric.
5. **Download**: ZIP with all images, figures and CSVs.

## Notes

- Runtime grows with image size, `n_init` and the K range. If it is slow, lower **Max image side**, **n_init**, or use **Manual** K.
- In Auto mode, if both images pick the same K, only one K is clustered.
- The notebook's default `MAX_SIDE` is 512 px; the PPT mentions 1080 px as a loader limit. Change it in Advanced settings if you want to match the PPT.
- Colours in cluster maps are arbitrary labels and are not matched between images; the pixel-change map does match labels (Hungarian algorithm) before comparing.

## Deploy online (optional, free)

1. Push the three files to a GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, and select the repo and `app.py`.
3. Streamlit Community Cloud installs `requirements.txt` and gives you a public link.
