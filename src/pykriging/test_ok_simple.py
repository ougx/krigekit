from scipy.spatial.distance import cdist, pdist
from _kriging import Kriging
_VGM = "sph 0.0 1.0 50.0 50.0 50.0 0.0 0.0 0.0"
#%% simple test
if __name__ == "__main__":
    import numpy as np
    rng   = np.random.default_rng(42)
    coord = rng.uniform(0, 100, (20, 2))
    grid  = rng.uniform(0, 100, (20, 2))
    value = rng.uniform(-5, 5, 20)

    dist = cdist(coord, grid)
    # np.savetxt("obsloc.dat", coord, fmt="%s")
    # np.savetxt("newloc.dat", grid, fmt="%s")
    # np.savetxt("obsval.dat", value[:,None], fmt="%s")
    # print(grid)
    # print(value)

    lower = 0.0
    k = Kriging(ndim=2, nvar=1, bounds=(lower, 10.0), write_mat=True, verbose=True)
    k.set_obs(ivar=1, coord=coord, value=value, nmax=10)
    k.set_vgm(ivar=1, jvar=1, spec=_VGM)
    k.set_grid(coord=grid)
    k.set_search(ivar=1)
    k.solve()
    est, _ = k.get_results()
    print("Done-------")
    print(est)