# ==========================================================
# CATBOOST + SHAP COMPLETE WORKFLOW (GOOGLE DRIVE VERSION)
# ==========================================================

!pip install catboost shap joblib scipy matplotlib --quiet

from google.colab import drive
drive.mount("/content/drive")

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# ---------------- PATHS ----------------
BASE = "/content/drive/MyDrive/Rudraprayag"

TRAIN_CSV = f"{BASE}/dataset_final/Training_LS_FINAL_UTM.csv"
TEST_CSV  = f"{BASE}/dataset_final/Testing_LS_FINAL_UTM.csv"

OUT_DIR = f"{BASE}/model_output_CatBoost_SHAP"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = f"{OUT_DIR}/CatBoost_model.joblib"

# ---------------- PARAMETERS ----------------
PARAMS = [
    "Aspect","Curvature","DTL","DTR","DTS","DTT","Elevation","Geomorphology","Lithology",
    "Slope Length","LULC","NDVI","NDWI","Plan Curv","Prof Curv","Roughness","RSP","Slope",
    "Soil Texture","Soil Depletion","Soil Moisture","Solar Radiation",
    "SPI","TPI","TWI","VD","VDCN"
]

TARGET = "grid_code"

# ==========================================================
# LOAD DATA
# ==========================================================

df_tr = pd.read_csv(TRAIN_CSV)
df_te = pd.read_csv(TEST_CSV)

X_tr = df_tr[PARAMS].apply(pd.to_numeric, errors="coerce")
X_te = df_te[PARAMS].apply(pd.to_numeric, errors="coerce")

medians = X_tr.median()

X_tr = X_tr.fillna(medians).values.astype("float32")
X_te = X_te.fillna(medians).values.astype("float32")

y_tr = df_tr[TARGET].astype(int).values
y_te = df_te[TARGET].astype(int).values

# ==========================================================
# TRAIN CATBOOST
# ==========================================================

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)

model.fit(X_tr, y_tr)

joblib.dump(model, MODEL_PATH)

# ==========================================================
# VALIDATION
# ==========================================================

proba_te = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba_te)

print(f"\n✅ CatBoost ROC-AUC: {auc:.4f}")

# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_tr)

# ---------------- SHAP SUMMARY ----------------
plt.figure()
shap.summary_plot(shap_values, X_tr, feature_names=PARAMS, show=False)
plt.savefig(f"{OUT_DIR}/SHAP_summary.png", dpi=500, bbox_inches="tight")
plt.close()

# ---------------- GLOBAL IMPORTANCE ----------------
mean_shap = np.abs(shap_values).mean(axis=0)

df_imp = pd.DataFrame({
    "Factor": PARAMS,
    "Mean_|SHAP|": mean_shap
}).sort_values("Mean_|SHAP|", ascending=False)

plt.figure(figsize=(8,10))
plt.barh(df_imp["Factor"], df_imp["Mean_|SHAP|"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP value|")
plt.title("Global Factor Importance (CatBoost)")

for i, v in enumerate(df_imp["Mean_|SHAP|"]):
    plt.text(v, i, f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/SHAP_global_importance.png", dpi=500, bbox_inches="tight")
plt.close()

# ==========================================================
# CUSTOM DEPENDENCE FUNCTION
# ==========================================================

def shap_dependence_plot_custom(
    X, shap_vals, proba, feature_name, feature_idx,
    bins=25, save_dir=None
):

    x = X[:, feature_idx]
    y_shap = shap_vals[:, feature_idx]

    mask_pos = proba >= 0.5
    mask_neg = proba < 0.5

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x[mask_neg], y_shap[mask_neg],
        c=y_shap[mask_neg], cmap="Blues",
        s=28, alpha=0.85, label="Proba < 0.5"
    )

    plt.scatter(
        x[mask_pos], y_shap[mask_pos],
        c=y_shap[mask_pos], cmap="Reds",
        s=28, alpha=0.85, label="Proba ≥ 0.5"
    )

    bin_med, bin_edges, _ = binned_statistic(
        x, y_shap, statistic="median", bins=bins
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.plot(
        bin_centers, bin_med,
        color="black", linewidth=2,
        label="Median SHAP (binned)"
    )

    zero_idx = np.nanargmin(np.abs(bin_med))
    zero_x = bin_centers[zero_idx]

    plt.axvline(
        zero_x, color="green",
        linestyle="--", linewidth=1.8,
        label=f"Zero-cross ≈ {zero_x:.2f}"
    )

    plt.axhline(0, color="gray", linestyle=":", linewidth=1)

    plt.xlabel(feature_name)
    plt.ylabel("SHAP value")
    plt.title(f"SHAP Dependence Plot: {feature_name}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    cbar = plt.colorbar()
    cbar.set_label("SHAP value")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"SHAP_dependence_{feature_name}.png"),
            dpi=500, bbox_inches="tight"
        )

    plt.close()

# ==========================================================
# GENERATE DEPENDENCE PLOTS FOR ALL FACTORS
# ==========================================================

proba_tr = model.predict_proba(X_tr)[:, 1]

DEP_PLOT_DIR = f"{OUT_DIR}/SHAP_dependence_plots"
os.makedirs(DEP_PLOT_DIR, exist_ok=True)

for i, p in enumerate(PARAMS):
    shap_dependence_plot_custom(
        X=X_tr,
        shap_vals=shap_values,
        proba=proba_tr,
        feature_name=p,
        feature_idx=i,
        bins=25,
        save_dir=DEP_PLOT_DIR
    )

print("\n✅ All SHAP plots saved successfully.")
print(f"📁 Output directory: {OUT_DIR}")
