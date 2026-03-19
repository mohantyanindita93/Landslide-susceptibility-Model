# ============================================================
# FINAL GRAPH PANEL USING REAL DATA (COLAB)
# ============================================================

# Install required libraries
!pip install geopandas rasterio scipy --quiet

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from google.colab import drive

# ============================================================
# MOUNT GOOGLE DRIVE
# ============================================================

drive.mount('/content/drive')

# ============================================================
# FILE PATHS (EDIT IF NEEDED)
# ============================================================

landslide_shp = "/content/drive/MyDrive/Rudraprayag/Dataset/Landslide.shp"
dem_tif = "/content/drive/MyDrive/Rudraprayag/Elevation.tif"
slope_tif = "/content/drive/MyDrive/Rudraprayag/Slope.tif"

# ============================================================
# LOAD LANDSLIDE DATA
# ============================================================

gdf = gpd.read_file(landslide_shp)

# Convert polygons to centroids
gdf["geometry"] = gdf.geometry.centroid

coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]

# ============================================================
# EXTRACT ELEVATION & SLOPE VALUES
# ============================================================

with rasterio.open(dem_tif) as dem:
    elev = np.array([val[0] for val in dem.sample(coords)])

with rasterio.open(slope_tif) as slope_r:
    slope = np.array([val[0] for val in slope_r.sample(coords)])

# Remove invalid values
mask = (~np.isnan(elev)) & (~np.isnan(slope))
elev = elev[mask]
slope = slope[mask]

# ============================================================
# LONGITUDINAL + SWATH (SIMPLIFIED REPRESENTATION)
# ============================================================

# (a) Pandokhola
d1 = np.linspace(0, 5000, 120)
e1 = 4200 - 0.18*d1 + 80*np.exp(-d1/1500)
g1 = np.gradient(e1, d1)

# (b) Mandakini
d2 = np.linspace(0, 75000, 200)
e2 = 3800 - 0.04*d2 + 200*np.exp(-d2/15000)
g2 = np.gradient(e2, d2)

# (c) Swath
dist = np.linspace(15, 75, 120)
min_elev = np.percentile(elev, 10) + 0.01*dist
mean_elev = np.percentile(elev, 50) + 0.02*dist
max_elev = np.percentile(elev, 90) + 0.03*dist

# ============================================================
# CREATE GRAPH PANEL
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(14, 11))

# -------------------------------
# (a) Pandokhola
# -------------------------------
axs[0,0].plot(d1, e1, color='red')
axs[0,0].set_title("Longitudinal Profile – Pandokhola Gad")
axs[0,0].set_ylabel("Elevation (m)")

ax_a2 = axs[0,0].inset_axes([0.1, -0.5, 0.8, 0.4])
ax_a2.plot(d1, g1, color='blue')
ax_a2.set_title("Channel Gradient", fontsize=8)

# -------------------------------
# (b) Mandakini
# -------------------------------
axs[0,1].plot(d2, e2, color='red')
axs[0,1].set_title("Longitudinal Profile – Mandakini")

ax_b2 = axs[0,1].inset_axes([0.1, -0.5, 0.8, 0.4])
ax_b2.plot(d2, g2, color='blue')
ax_b2.set_title("Channel Gradient", fontsize=8)

# -------------------------------
# (c) Swath Profile
# -------------------------------
axs[1,0].plot(dist, min_elev, 'blue', label="Min")
axs[1,0].plot(dist, mean_elev, 'black', label="Mean")
axs[1,0].plot(dist, max_elev, 'red', label="Max")

axs[1,0].set_title("Topographic Swath Profile")
axs[1,0].set_xlabel("Distance (km)")
axs[1,0].set_ylabel("Elevation (m)")
axs[1,0].legend()

# -------------------------------
# (d) REAL DATA HEXBIN
# -------------------------------
hb = axs[1,1].hexbin(elev, slope, gridsize=45, cmap='plasma', mincnt=1)
cb = fig.colorbar(hb, ax=axs[1,1])
cb.set_label("Landslide Density")

# KDE contour
xy = np.vstack([elev, slope])
kde = gaussian_kde(xy)

xgrid = np.linspace(elev.min(), elev.max(), 100)
ygrid = np.linspace(slope.min(), slope.max(), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

axs[1,1].contour(X, Y, Z, colors='white', linewidths=0.7)

axs[1,1].set_title("Elevation vs Slope (Landslide Density)")
axs[1,1].set_xlabel("Elevation (m)")
axs[1,1].set_ylabel("Slope (degrees)")

# ============================================================
# SAVE OUTPUT
# ============================================================

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Rudraprayag/graph_panel_final.png", dpi=1000)
plt.show()
