import sys
sys.path.insert(0, "../src/krigekit")

from krigekit import ordinary_kriging
_VGM   = dict(vtype="sph", nugget=0.0, sill=0.12, a_major=5000.0)
#%% simple test
if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("../test_data/pc2d.csv")
    grid = pd.read_csv("../test_data/grid2d.csv")

    est, var = ordinary_kriging(
        data[["x", "y"]].values,
        data["pc"].values,
        grid[["x", "y"]].values,
        vgm_spec=_VGM,
        nmax=20
    )

    print(pd.DataFrame({"est":est, "ref":grid["estimate"].values}).corr())
