# ============================================================
# GRAPH PANEL CODE – RUDRAPRAYAG
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# 1. LONGITUDINAL PROFILE (PANDOKHOLA GAD)
# ============================================================

distance1 = np.linspace(0, 5000, 100)
elevation1 = 4200 - 0.18*distance1 + 60*np.sin(distance1/600)

gradient1 = np.gradient(elevation1, distance1)

# ============================================================
# 2. LONGITUDINAL PROFILE (MANDAKINI)
# ============================================================

distance2 = np.linspace(0, 75000, 200)
elevation2 = 3800 - 0.04*distance2 + 150*np.exp(-distance2/20000)

gradient2 = np.gradient(elevation2, distance2)

# ============================================================
# 3. TOPOGRAPHIC SWATH PROFILE
# ============================================================

dist_swath = np.linspace(15, 75, 100)

min_elev = 1000 + 25*dist_swath + np.random.rand(100)*150
mean_elev = min_elev + 500
max_elev = mean_elev + 1200

# ============================================================
# 4. ELEVATION vs SLOPE (LANDSLIDE HEXBIN)
# ============================================================

np.random.seed(42)

elev = np.random.normal(1400, 500, 3000)
slope = np.random.normal(30, 10, 3000)

# ============================================================
# CREATE PANEL (2x2)
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# -------------------------------
# (a) Pandokhola Longitudinal Profile
# -------------------------------
axs[0, 0].plot(distance1, elevation1, 'r')
axs[0, 0].set_title("(a) Longitudinal Profile – Pandokhola Gad")
axs[0, 0].set_ylabel("Elevation (m)")
axs[0, 0].set_xlabel("Distance (m)")

# -------------------------------
# (b) Mandakini Longitudinal Profile
# -------------------------------
axs[0, 1].plot(distance2, elevation2, 'r')
axs[0, 1].set_title("(b) Longitudinal Profile – Mandakini")
axs[0, 1].set_ylabel("Elevation (m)")
axs[0, 1].set_xlabel("Distance (m)")

# -------------------------------
# (c) Elevation vs Slope (Hexbin)
# -------------------------------
hb = axs[1, 0].hexbin(elev, slope, gridsize=40, cmap='plasma', mincnt=1)
cb = fig.colorbar(hb, ax=axs[1, 0])
cb.set_label("Landslide Density")

# KDE contour
xy = np.vstack([elev, slope])
kde = gaussian_kde(xy)

xgrid = np.linspace(elev.min(), elev.max(), 100)
ygrid = np.linspace(slope.min(), slope.max(), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

axs[1, 0].contour(X, Y, Z, colors='white', linewidths=0.7)

axs[1, 0].set_title("(c) Elevation vs Slope (Landslide Density)")
axs[1, 0].set_xlabel("Elevation (m)")
axs[1, 0].set_ylabel("Slope (degrees)")

# -------------------------------
# (d) Topographic Swath Profile
# -------------------------------
axs[1, 1].plot(dist_swath, min_elev, 'b', label="Min Elevation")
axs[1, 1].plot(dist_swath, mean_elev, 'k', label="Mean Elevation")
axs[1, 1].plot(dist_swath, max_elev, 'r', label="Max Elevation")

axs[1, 1].set_title("(d) Topographic Swath Profile")
axs[1, 1].set_xlabel("Distance (km)")
axs[1, 1].set_ylabel("Elevation (m)")
axs[1, 1].legend()

# ============================================================
# FINAL LAYOUT
# ============================================================

plt.tight_layout()

# SAVE HIGH RESOLUTION (1000 DPI)
plt.savefig("graph_panel.png", dpi=1000)

plt.show()
