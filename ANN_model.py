# ============================================================
# NORMAL ANN–MLP LANDSLIDE SUSCEPTIBILITY MODEL
# WITH PROPER BOUNDARY MASKING
# ============================================================

!pip install rasterio geopandas scikit-learn matplotlib tqdm --quiet

from google.colab import drive
drive.mount("/content/drive")

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = "/content/drive/MyDrive/Rudraprayag"

TRAIN_CSV = f"{BASE}/dataset_final/Training_LS_FINAL_UTM.csv"
TEST_CSV  = f"{BASE}/dataset_final/Testing_LS_FINAL_UTM.csv"
RE_DIR    = f"{BASE}/Re"
BOUNDARY  = f"{BASE}/study_area/boundary.shp"

OUT_DIR = f"{BASE}/model_output_ANN"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TIF = f"{OUT_DIR}/ANN_LSM.tif"

# ------------------------------------------------------------
# Conditioning Factors
# ------------------------------------------------------------
PARAMS = [
    "Aspect","Curvature","DTL","DTR","DTS","DTT","Elevation","Geomorphology","Lithology",
    "Slope Length","LULC","NDVI","NDWI","Plan Curv","Prof Curv","Roughness","RSP","Slope",
    "Soil Texture","Soil Depletion","Soil Moisture","Solar Radiation","SPI","TPI","TWI",
    "VD","VDCN"
]

TARGET = "grid_code"

# ------------------------------------------------------------
# Load Training & Testing Data
# ------------------------------------------------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

X_tr = train[PARAMS].apply(pd.to_numeric, errors="coerce")
X_te = test[PARAMS].apply(pd.to_numeric, errors="coerce")

medians = X_tr.median()

X_tr = X_tr.fillna(medians)
X_te = X_te.fillna(medians)

y_tr = train[TARGET].astype(int)
y_te = test[TARGET].astype(int)

# ------------------------------------------------------------
# Feature Scaling (VERY IMPORTANT FOR ANN)
# ------------------------------------------------------------
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# ------------------------------------------------------------
# Train Normal ANN (Simple Architecture)
# ------------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=400,
    random_state=42
)

model.fit(X_tr_scaled, y_tr)

# ------------------------------------------------------------
# Model Validation
# ------------------------------------------------------------
y_prob = model.predict_proba(X_te_scaled)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_te, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_te, y_pred, digits=4))

# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_te, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ANN (AUC={roc_auc_score(y_te,y_prob):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – ANN")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Raster Prediction Section
# ------------------------------------------------------------

def norm(s):
    return re.sub(r"[^a-z0-9]+","",s.lower())

tifs = [f for f in os.listdir(RE_DIR) if f.endswith(".tif")]
RASTER_MAP = {p: next(f for f in tifs if norm(p) in norm(f)) for p in PARAMS}

ref = rasterio.open(os.path.join(RE_DIR, RASTER_MAP[PARAMS[0]]))
profile = ref.profile.copy()

H, W = ref.height, ref.width
transform = ref.transform
crs = ref.crs

profile.update(dtype="float32", count=1, nodata=-9999)

# ------------------------------------------------------------
# Proper Boundary Mask Creation
# ------------------------------------------------------------
gdf = gpd.read_file(BOUNDARY)

if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

boundary_mask = rasterize(
    [(geom,1) for geom in gdf.geometry],
    out_shape=(H, W),
    transform=transform,
    fill=0,
    all_touched=True
).astype(bool)

# ------------------------------------------------------------
# Open Raster Layers
# ------------------------------------------------------------
rasters = {
    p: rasterio.open(os.path.join(RE_DIR, RASTER_MAP[p]))
    for p in PARAMS
}

# ------------------------------------------------------------
# Block-wise Prediction with Boundary Mask
# ------------------------------------------------------------
with rasterio.open(OUT_TIF, "w", **profile) as dst:

    for row in tqdm(range(0, H, 256), desc="ANN Prediction"):
        for col in range(0, W, 256):

            height = min(256, H - row)
            width  = min(256, W - col)
            window = Window(col, row, width, height)

            stack = []
            for p in PARAMS:
                arr = rasters[p].read(1, window=window)
                arr = np.where(np.isfinite(arr), arr, medians[p])
                stack.append(arr.reshape(-1))

            X_block = np.stack(stack, axis=1)
            X_block_scaled = scaler.transform(X_block)

            probs = model.predict_proba(X_block_scaled)[:,1]

            mask_block = boundary_mask[row:row+height, col:col+width].reshape(-1)

            out_block = np.full(probs.shape, -9999, dtype="float32")
            out_block[mask_block] = probs[mask_block]

            dst.write(out_block.reshape(height, width), 1, window=window)

for r in rasters.values():
    r.close()

print("✅ ANN Landslide Susceptibility Map saved at:")
print(OUT_TIF)

# ------------------------------------------------------------
# Final Map Visualization
# ------------------------------------------------------------
with rasterio.open(OUT_TIF) as src:
    arr = src.read(1)
    arr = np.where(arr == -9999, np.nan, arr)

plt.figure(figsize=(9,6))
plt.imshow(arr, cmap="plasma", vmin=0, vmax=1)
plt.colorbar(label="ANN Landslide Susceptibility")
plt.title("ANN Landslide Susceptibility Map (Boundary Clipped)")
plt.axis("off")
plt.tight_layout()
plt.show()
