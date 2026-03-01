!pip -q install jenks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# 0) YOUR EXACT LULC TABLE (given)
# -------------------------
lulc_manual = pd.DataFrame({
    "LULC_Class": [
        "Snow/glacier",
        "Dense Vegetation",
        "Grassland",
        "Barrenland",
        "Cropland",
        "Urban",
        "Waterbody",
        "Road"
    ],
    "Total_pixels": [175043, 804244, 315577, 245067, 181352, 116536, 108969, 225886],
    "Landslide_pixels": [160, 2457, 2213, 3181, 824, 958, 1268, 1406]
})

# compute FR from manual LULC table
_total_all = lulc_manual["Total_pixels"].sum()
_total_ls  = lulc_manual["Landslide_pixels"].sum()
lulc_manual["FR"] = ((lulc_manual["Landslide_pixels"]/_total_ls) /
                     (lulc_manual["Total_pixels"]/_total_all))
lulc_manual_out = lulc_manual.copy()

# -------------------------
# 1) PATHS (EDIT IF NEEDED)
# -------------------------
BASE = Path("/content/drive/MyDrive/Rudraprayag")
CSV_PATH = BASE / "dataset_final" / "Training_LS_FINAL_UTM.csv"
TARGET_COL = "grid_code"

# -------------------------
# 2) LOAD
# -------------------------
df = pd.read_csv(CSV_PATH)

# drop Shape if exists
if "Shape" in df.columns:
    df = df.drop(columns=["Shape"])

# ensure target 0/1
df[TARGET_COL] = df[TARGET_COL].replace({True:1, False:0, "Yes":1, "No":0, "yes":1, "no":0})
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

# -------------------------
# 3) PARAMETERS (keep only those present)
# -------------------------
PARAMS = [
    "Aspect","Curvature","DTL","DTR","DTS","DTT","Elevation","Geomorphology","Lithology",
    "Slope Length","LULC","NDVI","NDWI","Plan Curv","Prof Curv","Roughness","RSP","Slope",
    "Soil Texture","Soil Depletion","Soil Moisture","Solar Radiation","SPI","TPI","TWI",
    "VD","VDCN"
]
PARAMS = [p for p in PARAMS if p in df.columns]

# -------------------------
# 4) CLEAN -9999 / nodata (for numeric columns)
# -------------------------
NODATA_VALUES = [-9999, -9999.0, -9999.00, 9999, 9999.0]

for c in df.columns:
    if c == TARGET_COL:
        continue
    if df[c].dtype != "object":
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace(NODATA_VALUES, np.nan).replace([np.inf, -np.inf], np.nan)

# -------------------------
# 5) CLASS NAME MAPPINGS
# -------------------------
lulc_map = {
    1:"Snow/glacier",
    2:"Dense Vegetation",
    3:"Grassland",
    4:"Barrenland",
    5:"Cropland",
    6:"Urban",
    7:"Waterbody",
    8:"Road"
}

soil_texture_map = {
    1218: "Sandy Loam",
    1227: "Sandy Clay Loam",
    1592: "Clay Loam",
    31657: "Loam"
}

geomorph_map = {
    36:"WatBod - Lake",
    43:"FluOri - Active Flood plain",
    185:"Point Bar",
    188:"DenOri - Mass Wasting Products",
    211:"GlaOri - Glacial Terrain",
    215:"Lateral Moraine",
    249:"Kame Terrace",
    258:"Nunatak",
    264:"Lake",
    294:"Lateral Bar",
    296:"Medial Moraine",
    485:"Channel Bar",
    561:"Road cutting",
    737:"Talus / Scree",
    1701:"Landslide",
    3022:"Terrace",
    5834:"Valley",
    12007:"Ridge",
    12269:"FluOri - Piedmont Alluvial Plain",
    15812:"Valley Glacier",
    33720:"DenOri - Piedmont Slope",
    34177:"WatBod - River",
    52994:"StrOri - Moderately Dissected Hills and Valleys",
    281638:"Snow Cover",
    1715478:"StrOri - Highly Dissected Hills and Valleys"
}

# If Lithology is numeric codes, fill mapping here. If already text, keep empty.
lithology_map = {}

# -------------------------
# 6) FIXED LEGEND BINS (Aspect + Soil Depletion)
# -------------------------
aspect_bins = [
    (-1.0, -1.0, "Flat (-1)"),
    (0.0, 22.5, "North (0–22.5)"),
    (22.5, 67.5, "Northeast (22.5–67.5)"),
    (67.5, 112.5, "East (67.5–112.5)"),
    (112.5, 157.5, "Southeast (112.5–157.5)"),
    (157.5, 202.5, "South (157.5–202.5)"),
    (202.5, 247.5, "Southwest (202.5–247.5)"),
    (247.5, 292.5, "West (247.5–292.5)"),
    (292.5, 337.5, "Northwest (292.5–337.5)"),
    (337.5, 360.0, "North (337.5–360)")
]

soil_depletion_bins = [
    (-9.1, 27.5,  "Minor (<10 ha)"),
    (27.6, 43.5,  "Normal (10–20 ha)"),
    (43.6, 78.8,  "Significant (20–30 ha)"),
    (78.9, 116.6, "Alarming (30–40 ha)"),
    (117.0, 222.0,"Hazardous (>40 ha)")
]

# -------------------------
# 7) HELPERS
# -------------------------
def frequency_ratio_table(cats, y, order=None):
    s = pd.DataFrame({"cat": cats, "y": y}).dropna()
    total_all = len(s)
    ls_all = int(s["y"].sum())

    if total_all == 0 or ls_all == 0:
        return pd.DataFrame(columns=["Class", "Total", "Landslide", "FR"])

    if order is None:
        order = s["cat"].value_counts().index.tolist()

    rows = []
    for lab in order:
        sub = s[s["cat"] == lab]
        tot = len(sub)
        ls = int(sub["y"].sum())
        if tot == 0:
            continue
        fr = (ls / ls_all) / (tot / total_all)
        rows.append([lab, tot, ls, fr])

    out = pd.DataFrame(rows, columns=["Class", "Total", "Landslide", "FR"])
    return out

def map_codes(series, mapping):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(NODATA_VALUES, np.nan)
    return s.map(mapping)

def aspect_to_class(series):
    a = pd.to_numeric(series, errors="coerce").replace(NODATA_VALUES, np.nan)
    out = pd.Series(index=a.index, dtype="object")
    for lo, hi, name in aspect_bins:
        if lo == hi:
            out[a == lo] = name
        else:
            out[(a > lo) & (a <= hi)] = name
    return out

def soil_depletion_to_class(series):
    x = pd.to_numeric(series, errors="coerce").replace(NODATA_VALUES, np.nan)
    out = pd.Series(index=x.index, dtype="object")
    for lo, hi, name in soil_depletion_bins:
        out[(x >= lo) & (x <= hi)] = name
    return out

def bin_continuous_5(series):
    x = pd.to_numeric(series, errors="coerce")
    x = x.replace(NODATA_VALUES, np.nan).replace([np.inf, -np.inf], np.nan)
    x_clean = x.dropna()
    if x_clean.empty:
        return None, None

    # Try Jenks, else Quantiles
    try:
        from jenks import jenks_breaks
        breaks = jenks_breaks(x_clean.values, 5)
        breaks = np.unique(breaks)
    except Exception:
        breaks = np.unique(np.quantile(x_clean.values, np.linspace(0, 1, 6)))

    if len(breaks) < 3:
        breaks = np.unique(np.quantile(x_clean.values, np.linspace(0, 1, 6)))

    if len(breaks) < 3:
        return None, None

    labels = [f"{breaks[i]:.2f}–{breaks[i+1]:.2f}" for i in range(len(breaks)-1)]
    binned = pd.cut(x, bins=breaks, include_lowest=True)
    cats = binned.cat.codes.map({i: labels[i] for i in range(len(labels))})
    return cats, labels

# -------------------------
# 8) BUILD TABLES (ALL FACTORS)
# -------------------------
tables = {}
y = df[TARGET_COL]

for p in PARAMS:
    # ---- LULC: USE YOUR MANUAL TABLE (not from CSV)
    if p == "LULC":
        tab = lulc_manual_out.rename(columns={"LULC_Class":"Class",
                                              "Total_pixels":"Total",
                                              "Landslide_pixels":"Landslide"})[["Class","Total","Landslide","FR"]]
        tables[p] = tab.copy()
        continue

    # ---- Soil Texture
    if p == "Soil Texture":
        cats = map_codes(df[p], soil_texture_map)
        order = ["Sandy Loam","Sandy Clay Loam","Clay Loam","Loam"]
        tab = frequency_ratio_table(cats, y, order=order)

    # ---- Geomorphology
    elif p == "Geomorphology":
        if df[p].dtype == "object":
            cats = df[p].astype(str).replace(["Unknown","unknown","UNKNOWN","nan","None"], np.nan)
            cats = cats.replace("UNMAPPED", np.nan)
        else:
            cats = map_codes(df[p], geomorph_map)

        # keep only mapped classes
        tab = frequency_ratio_table(cats, y, order=pd.Series(list(geomorph_map.values())).unique().tolist())

    # ---- Lithology
    elif p == "Lithology":
        if df[p].dtype == "object":
            cats = df[p].astype(str).replace(["Unknown","unknown","UNKNOWN","nan","None"], np.nan)
            cats = cats.replace("UNMAPPED", np.nan)
        else:
            cats = map_codes(df[p], lithology_map)

        tmp = pd.DataFrame({"cat": cats, "y": y}).dropna()
        top = tmp["cat"].value_counts().head(12).index.tolist()  # top 12 for readability
        tab = frequency_ratio_table(cats, y, order=top)

    # ---- Aspect (fixed bins)
    elif p == "Aspect":
        cats = aspect_to_class(df[p])
        order = [b[2] for b in aspect_bins]
        tab = frequency_ratio_table(cats, y, order=order)

    # ---- Soil Depletion (fixed bins)
    elif p == "Soil Depletion":
        cats = soil_depletion_to_class(df[p])
        order = [b[2] for b in soil_depletion_bins]
        tab = frequency_ratio_table(cats, y, order=order)

    # ---- Other continuous factors -> 5 bins
    else:
        cats, labels = bin_continuous_5(df[p])
        if cats is None:
            continue
        tab = frequency_ratio_table(cats, y, order=labels)

    tab = tab.dropna(subset=["FR"])
    if not tab.empty:
        tables[p] = tab

# -------------------------
# 9) PLOT: ALL PARAMETERS IN ONE FIGURE
# -------------------------
n = len(tables)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(18, 4.1*rows))
axes = np.array(axes).reshape(-1)

fig.suptitle(
    "Relationship of landslide with causative factors (FR) — cleaned (no -9999 / no unknown)\n"
    "LULC FR uses your exact LULC table values",
    fontsize=15, weight="bold", y=0.995
)###what is this code for
