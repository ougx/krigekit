import numpy as np
from krigekit import Kriging

coord = np.array([[0.3, 1.2],[1.9, 0.6],[1.1, 3.2],[3.3, 4.4],[4.7, 3.8]])
value = np.array([0.47, 0.56, 0.74, 1.47, 1.74])
grid  = np.array([[2., 2.],[3., 4.],[4., 4.]])
vgm   = dict(vtype='sph', nugget=0.01, sill=0.09, a_major=100.0)
NMAX  = 5

# Build k2 fresh without any reference to k1
k2 = Kriging(ndim=2, nvar=1, use_old_weight=True)
k2.set_obs(ivar=1, coord=coord, value=value, nmax=NMAX)
k2.set_grid(coord=grid)
k2.set_vgm(ivar=1, jvar=1, **vgm)
k2.set_search(ivar=1)

# Construct minimal weights manually
nb, ng, nm, nv = 3, 1, 5, 1
nnear  = np.array([[5],[5],[5]], dtype=np.int32)   # (nb, ng) = (3, 1)
inear  = np.ones((nb, ng, nm), dtype=np.int32)     # (3, 1, 5) - all obs 1
weight = np.full((nb, ng, nm), 0.2, dtype=np.float64)  # uniform weights
w = {'nnear': nnear, 'inear': inear, 'weight': weight}

print('set_weights start')
k2.set_weights(w)
print('set_weights done, k2.use_old_weight =', k2.use_old_weight)
print('calling solve...')
k2.solve()
print('solve done')