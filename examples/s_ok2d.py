"""
Ordinary kriging example
========================

This example shows how to run ordinary kriging and plot the interpolated field.
"""

from krigekit import Kriging
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../test_data/pc2d.csv") 
grid = pd.read_csv("../test_data/grid2d.csv")

# default is 2D univariate ordinary Kriging
k = Kriging()
k.set_obs(ivar=1, coord=data[["x", "y"]], value=data["pc"], nmax=30) 
k.set_grid(coord=grid[["x", "y"]])
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=0.12, a_major=5000.0)
k.set_search()
k.solve()
df = k.get_result_df()

print(df)
#%%
# Plot the interpolated values
#----------------------------------------------------
plt.imshow(df["estimate"].values.reshape([80, 60]), cmap="turbo", vmin=0, vmax=1)
plt.title("Estimate")
plt.colorbar(label="Coarse Fraction", pad=0.01)
plt.show()


#%%
# Plot the Kriging error
#----------------------------------------------------
plt.imshow(df["variance"].values.reshape([80, 60]), cmap="Reds", vmin=0, vmax=0.12)
plt.title("Variance")
plt.colorbar(label="Variance", pad=0.01)
plt.show()