"""
Ordinary Kriging + SGSIM 
========================

This example shows how to run SGSIM ordinary kriging and plot the interpolated field.
"""

from krigekit import Kriging
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

data = pd.read_csv("../test_data/pc2d.csv") 
grid = pd.read_csv("../test_data/grid2d.csv")

# nsim is the number of realizations
k = Kriging(nsim=3, bounds=[0,1]) 
k.set_obs(ivar=1, coord=data[["x", "y"]], value=data["pc"], nmax=100) 
k.set_grid(coord=grid[["x", "y"]])
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=0.12, a_major=5000.0)
k.set_sim()
k.set_search()
k.solve()
df = k.get_result_df()
del k
print(df)

fig = plt.figure(figsize=(11, 5))
axs = ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(1, 3),
                axes_pad=0.1,
                label_mode="L",
                cbar_mode="single", cbar_size="7%", cbar_pad="2%"
                )

for i, ax in enumerate(axs):
    est = df[f"sim_{i+1}"].values.reshape([80, 60])
    im = ax.imshow(est, cmap="turbo", vmin=0, vmax=1)
    ax.set(title=f"Realization {i+1}")
ax.cax.colorbar(im, label="Coarse Fraction")



#%% 
# Change Seed
# -----
#
# Let's change seed number and see how different the results are.

k = Kriging(nsim=3, bounds=[0,1], seed=1000) 
k.set_obs(ivar=1, coord=data[["x", "y"]], value=data["pc"], nmax=100) 
k.set_grid(coord=grid[["x", "y"]])
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=0.12, a_major=5000.0)
k.set_sim()
k.set_search()
k.solve()
df = k.get_result_df()
del k
print(df)

fig = plt.figure(figsize=(11, 5))
axs = ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(1, 3),
                axes_pad=0.1,
                label_mode="L",
                cbar_mode="single", cbar_size="7%", cbar_pad="2%"
                )

for i, ax in enumerate(axs):
    est = df[f"sim_{i+1}"].values.reshape([80, 60])
    im = ax.imshow(est, cmap="turbo", vmin=0, vmax=1)
    ax.set(title=f"Realization {i+1}")
ax.cax.colorbar(im, label="Coarse Fraction")
