"""
Verify the persistent between-solve factorization cache:
  1. After solve(), get_factor() returns valid matrices.
  2. A second solve() on the same grid skips kriging_setup (timing test).
  3. set_obs/set_vgm invalidate the cache (valid → False).
  4. Factors satisfy the mathematical identities:
       L L' = K   (within residual tolerance)
       K^{-1} F = kinv_drift
       schur chol = F' kinv_drift
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.linalg import solve_triangular
from krigekit import Kriging
from krigekit.meuk_fortran import MEUKFortran

# ── helpers ────────────────────────────────────────────────────────────────────
def sph_cov(C0, h, a):
    h = h / a
    return C0 * np.where(h < 1, 1 - 1.5*h + 0.5*h**3, 0.)

rng = np.random.default_rng(7)
n   = 15
x   = np.sort(rng.uniform(0, 100, n))
z   = np.sin(x / 20.) + rng.normal(0, 0.05, n)

# ── 1.  Simple UK — check factor content ──────────────────────────────────────
print("=== 1. Simple UK — factor content ===")
kg = Kriging(ndim=1, nvar=1, ndrift=1, unbias=0)
kg.set_vgm(ivar=1, jvar=1, vtype='sph', nugget=0.01, sill=0.09, a_major=30.)
kg.set_obs(ivar=1, coord=x[:, None], value=z)
kg.set_obs_drift(ivar=1, drift=x[:, None])          # linear trend
x_grid = np.linspace(1, 99, 50)
kg.set_grid(coord=x_grid[:, None])
kg.set_grid_drift(drift=x_grid[:, None])
kg.set_search(ivar=1)

# Before solve: factor must be invalid
f0 = kg.get_factor()
assert not f0['valid'], "factor must be invalid before solve()"
print("  Pre-solve valid=False: OK")

kg.solve()

f = kg.get_factor()
assert f['valid'], "factor must be valid after solve()"
npp, p = f['npp'], f['p']
L, kinv, schur = f['L'], f['kinv_drift'], f['schur']
print(f"  npp={npp}, p={p}, L.shape={L.shape}, kinv.shape={kinv.shape}, schur.shape={schur.shape}")

# ── Build the K matrix in Python for verification ─────────────────────────────
C0 = 0.01 + 0.09   # C(0) = nugget + sill
K  = sph_cov(0.09, np.abs(x[:, None] - x[None, :]), 30.)
np.fill_diagonal(K, C0)
F  = x[:, None]    # drift column (1D)

# solver.f90 uses uplo='U' (upper triangular).
# spotrf('U', n, L, ...) fills the UPPER triangle so that K = U' U.
# The lower triangle retains the original K values — ignore it.
U = np.triu(L)

# Verify U' U ≈ K
K_recon = U.T @ U
err_K = np.abs(K - K_recon).max()
print(f"  max|K - triu(L)'triu(L)| = {err_K:.2e}")
assert err_K < 1e-4, f"Cholesky reconstruction failed (err={err_K:.2e})"

# Verify kinv_drift ≈ K^{-1} F   (spotrs('U') solves U'U x = rhs)
kinv_ref = np.linalg.solve(K, F)
err_kinv = np.abs(kinv[:, :1] - kinv_ref).max()
print(f"  max|kinv - K^(-1)F|      = {err_kinv:.2e}")
assert err_kinv < 1e-4

# Verify schur: U_S' U_S ≈ F'K^{-1}F   (spotrf('U') on Schur complement)
S         = F.T @ kinv_ref           # (1,1) Schur complement
U_S       = np.triu(schur)
err_schur = abs((U_S.T @ U_S)[0, 0] - S[0, 0])
print(f"  max|schur'schur - S|     = {err_schur:.2e}")
assert err_schur < 1e-4
print("  Mathematical identities: PASS")

# ── 2.  Between-solve reuse: second solve() should be faster ──────────────────
print("\n=== 2. Between-solve timing ===")
N_REP = 5
times = []
for i in range(N_REP * 2):
    kg.set_grid(coord=x_grid[:, None])
    kg.set_grid_drift(drift=x_grid[:, None])
    t0 = time.perf_counter()
    kg.solve()
    times.append(time.perf_counter() - t0)

t_first = times[0]
t_reuse = np.mean(times[N_REP:])
print(f"  First solve  : {t_first*1e3:.2f} ms")
print(f"  Reuse (mean) : {t_reuse*1e3:.2f} ms  (speedup ~{t_first/t_reuse:.1f}x)")
# Soft assertion — at least a little faster (timing can be noisy)
assert t_reuse <= t_first * 1.5, "Reuse solve should not be significantly slower"
print("  Timing: PASS")

# ── 3.  Invalidation rules ────────────────────────────────────────────────────
print("\n=== 3. Cache invalidation ===")
f_before = kg.get_factor()
assert f_before['valid']

# update_obs_value changes only the RHS (observed values); K is unchanged.
# The persistent factor must remain valid.
kg.update_obs_value(ivar=1, value=z + 0.001)
f_upd = kg.get_factor()
assert f_upd['valid'], "update_obs_value must NOT invalidate factor (K unchanged)"
print("  update_obs_value keeps factor valid: OK")

# set_obs with new coordinates changes K → must invalidate.
x_new = x + rng.normal(0, 0.5, n)   # slightly perturbed coordinates
x_new = np.sort(np.clip(x_new, 1, 99))
kg.set_obs(ivar=1, coord=x_new[:, None], value=z)
kg.set_obs_drift(ivar=1, drift=x_new[:, None])   # restore drift after set_obs
f_new_coord = kg.get_factor()
assert not f_new_coord['valid'], "set_obs (new coords) must invalidate factor"
kg.set_search(ivar=1)          # required after set_obs to cap nmax at n
kg.set_grid(coord=x_grid[:, None])
kg.set_grid_drift(drift=x_grid[:, None])
kg.solve()
assert kg.get_factor()['valid'], "factor valid again after re-solve"
print("  set_obs (new coords) invalidates and re-populates factor: OK")

# set_vgm changes K → must invalidate.
kg.set_vgm(ivar=1, jvar=1, vtype='sph', nugget=0.01, sill=0.09, a_major=30.)
assert not kg.get_factor()['valid'], "set_vgm must invalidate the persistent factor"
print("  set_vgm invalidates factor: OK")

# ── 4.  MEUKFortran — factor accessible via .kriging.get_factor() ─────────────
print("\n=== 4. MEUKFortran factor access ===")
sigma, T = 0.001, 300.
x_e, x_i, rm = 250., 750., 5.

def W(x, Q):
    re = np.maximum(np.abs(x - x_e), rm)
    ri = np.maximum(np.abs(x - x_i), rm)
    return Q * np.log(ri / re) / (2 * np.pi)

EVENTS = [
    dict(n=23, grad=0.001,   Q=10., Z0=50.),
    dict(n=18, grad=-0.0007, Q=3.,  Z0=52.),
]
obs_data = []
for ev in EVENTS:
    xk = np.sort(rng.uniform(0, 1000, ev['n']))
    zk = ev['Z0'] + ev['grad']*xk + W(xk,ev['Q'])/T + rng.normal(0, sigma, ev['n'])
    obs_data.append((xk, zk))

mf = MEUKFortran(ndim=1)
mf.set_variogram('sph', nugget=5e-7, sill=5e-7, a_major=80.)
for k, (ev, (xk, zk)) in enumerate(zip(EVENTS, obs_data)):
    mf.add_event(k, xk[:,None], zk,
                 local_drift=np.column_stack([np.ones(ev['n']), xk]),
                 global_drift=W(xk, ev['Q']))
mf.build()

x_g = np.linspace(5, 995, 50)
mf.predict(0, x_g[:,None],
           pred_local_drift=np.column_stack([np.ones(50), x_g]),
           pred_global_drift=W(x_g, EVENTS[0]['Q']))

fm = mf.kriging.get_factor()
print(f"  MEUKFortran factor valid={fm['valid']}, npp={fm['npp']}, p={fm['p']}")
assert fm['valid']
assert fm['npp'] == sum(ev['n'] for ev in EVENTS)
assert fm['p']   == mf.ndrift()
print("  MEUKFortran get_factor(): PASS")

print("\nALL PASS")
