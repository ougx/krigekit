
from _kriging import ordinary_kriging

#%% simple test
if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../../test_data/pc2d.csv")
    grid = pd.read_csv("../../test_data/grid2d.csv")

    est, var = ordinary_kriging(
        data[["x", "y"]].values,
        data["pc"].values,
        grid[["x", "y"]].values,
        variogram_spec="sph 0.0 0.12 5000.0 5000.0 5000.0 0.0 0.0 0.0",
        nmax=62
    )