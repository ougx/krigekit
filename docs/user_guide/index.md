# User guide

Choose the guide that matches your task:

```{toctree}
:maxdepth: 1

ordinary_kriging
cokriging
universal_kriging
multi_event_universl_kriging
```

**Coming in Phase 2:**

- Sequential Gaussian Simulation (SGSIM)
- Space-time kriging
- Spatially Varying Anisotropy
- Kriging weight reuse
- Cross-validation
- OpenMP and reproducibility

---

## Quick-reference: which MEUK backend?

| Situation | Use |
|---|---|
| Production use, large grids, OpenMP | `MEUKFortran` |
| No compiled library available | `MEUK` (pure Python) |
| Debugging / verifying numerics | Both — results should match to ≤ 10⁻¹² |
| Inspecting K's factorisation | `MEUKFortran.kriging.get_factor()` |
