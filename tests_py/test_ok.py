import sys
sys.path.insert(0, "../src/krigekit")

from _kriging import Kriging
import pandas as pd
import numpy as np

print("1. Creating Kriging...")
k = Kriging(ndim=2, nvar=1, verbose=True, bounds=[0.0,1.0])
print(f"   OK handle={k._handle}")

data = pd.read_csv("../test_data/pc2d.csv")
grid = pd.read_csv("../test_data/grid2d.csv")
obs_coord = data[["x","y"]].values
obs_value = data["pc"].values
grid_coord = grid[["x","y"]].values

print("2. set_obs...")
k.set_obs(ivar=1, coord=obs_coord, value=obs_value, nmax=62)
print("   OK")

print("3. set_grid...")
k.set_grid(coord=grid_coord)
print("   OK")

print("4. set_vgm...")
k.set_vgm(ivar=1, jvar=1, vtype="sph", sill=0.12, a_major=5000.0)
print("   OK")

print("5. set_search...")
k.set_search(ivar=1)
print("   OK")
print(k)
print("6. solve...")
k.solve()
print("   OK")

print("7. get_results...")
est, var = k.get_results()
print("Mismatch:", )
print(grid[est.round(3)!=grid["estimate"].values.round(3)])

print("", )
print(f"   est: shape={est.shape}, range=[{est.min():.3f}, {est.max():.3f}]")
print(f"   var: shape={var.shape}, range=[{var.min():.3f}, {var.max():.3f}]")
print("DONE!")