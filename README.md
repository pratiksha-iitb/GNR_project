# Decorrelation Contrast Enhancement and K-Means Clustering

**GNR607 Project**
Adrika (23b2171), Anousha (23b2249), Avasthi (23b2227), Pratiksha (23b2111)

**Live app: https://gnrproject.streamlit.app/**

This app answers one question: *how does decorrelating an RGB image change the clusters K-Means finds?*
It enhances your image, clusters the original and the enhanced version, and compares the two.

---

## How to use

1. Open the link: https://gnrproject.streamlit.app/
   (If the page says the app is asleep, click **Yes, get this app back up!** and wait a minute.)
2. In the left sidebar, click **Browse files** and upload an image (JPG, PNG, etc.).
   No image? Tick **Use built-in sample image instead**.
3. Click **Run analysis**.
4. Wait for the run to finish (usually under a minute for a default-size image).
5. Look through the tabs at the top of the page.

That is all. The default settings are the ones used in our project, so you do not need to change anything.

---

## What you see in each tab

| Tab | What it shows |
|---|---|
| **1. Enhancement** | Original vs decorrelated image, correlation before and after, principal components, histograms |
| **2. Choosing K** | Davies-Bouldin curve and the optimal K (Auto mode only) |
| **3. Cluster maps** | Cluster maps and colour segmentation, original vs decorrelated, at the chosen K |
| **4. Comparison** | Pixel-change map, cluster sizes, quality scores (Silhouette, Davies-Bouldin, Calinski-Harabasz) |
| **5. Download** | ZIP with all images, figures and CSV files |

---

## Settings (optional)

**K selection** (sidebar)
- **Auto (optimal K)**: default. The app finds the best K using the Davies-Bouldin Index on the
  **decorrelated (enhanced) image**, then clusters both the original and the decorrelated image with that K.
- **Manual**: you choose K with the slider (2 to 20). Both images are clustered with your K and the
  Davies-Bouldin search is skipped, so it is faster.

**Advanced settings** (click the expander in the sidebar). Defaults are shown.

| Setting | Default |
|---|---|
| Max image side | 512 px |
| K-Means initialisations (n_init) | 10 |
| K-Means max iterations | 300 |
| Random seed | 42 |
| Sample size for metrics | 10,000 pixels |
| K min / K max | 2 / 20 |

Click **Reset to defaults** to restore all of these.

---

## Tips

- If the app is slow, use **Manual** K, or lower **Max image side** / **n_init**.
- Very large images are resized to the max side (512 px by default).
- Colours in the cluster maps are only labels. They are not matched between the two images.

---

## Run on your own computer (optional)

Needs Python 3. Open Terminal in the project folder and run:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: the user interface
- `pipeline.py`: the image processing and clustering code
- `requirements.txt`: Python packages needed
