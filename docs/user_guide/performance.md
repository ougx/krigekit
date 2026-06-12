# Performance tuning

## Overview

`solve()` parallelises the block loop with OpenMP and caches Cholesky
factorizations to avoid redundant O(nmax³) linear-algebra work.  Two
parameters on `solve()` control both:

```python
k.solve(nthread=4, ncache=64)
```

Understanding how they interact lets you get the best wall time for your
problem size.

---

## The factor cache

Every kriging system is defined by the exact set of neighbour indices that
fall inside the search radius for a given block.  Once the Cholesky
factorization of that covariance matrix has been computed (O(nmax³)), it can
be reused for any later block that has the **same** neighbour set, reducing
the per-block cost to a triangular solve (O(nmax²)).

Each OpenMP thread keeps its own independent hash cache of up to `ncache`
factorizations (default 64, set at compile time).  There is **no cross-thread
sharing**: two threads that happen to encounter the same neighbourhood pattern
each compute and store the factorization independently.

### Measuring cache effectiveness

After every `solve()` call, `solver_stats` reports the hit/miss counts:

```python
k.solve()
stats = k.solver_stats
hit_rate = stats["chol_reuse"] / (stats["chol_fact"] + stats["chol_reuse"])
print(f"Cache hit rate: {hit_rate:.1%}")
# chol_fact  — fresh O(nmax³) factorizations
# chol_reuse — O(nmax²) solves via a cached factor
```

A hit rate above ~80 % means the cache is working well and tuning can squeeze
out further gains.  A hit rate near zero means every block has a unique
neighbourhood and the cache is not helping.

---

## `nthread` — number of OpenMP threads

```python
k.solve(nthread=0)   # 0 (default): leave OMP_NUM_THREADS unchanged
k.solve(nthread=1)   # force single-threaded
k.solve(nthread=4)   # use exactly 4 threads
```

The block loop is split across threads using OpenMP's default (static
chunked) schedule, so contiguous blocks go to the same thread.  Nearby
blocks share nearby observations and therefore tend to share the same
neighbour sets — this locality means per-thread cache hit rates are already
reasonably high regardless of thread count.

### When to reduce `nthread`

The main reason to reduce threads is **cross-thread redundancy**: because
each thread has its own independent cache, two threads that encounter
identical neighbourhood patterns each factorize independently, doubling the
wasted O(nmax³) work.  This matters when:

- **The observation dataset is small** — with few observations, `nmax` is
  effectively limited by the total observation count, and large portions of
  the grid share *exactly* the same K nearest neighbours.  In the extreme
  case (all blocks pull the same neighbours), 1 thread factorizes once while
  N threads each factorize once — N× more total work, all redundant.
- **`nmax` is large** — each O(nmax³) factorization is expensive, so
  avoiding even a handful of redundant ones can save seconds.
- **The grid is small** — when the total number of blocks is close to the
  thread count, the overhead of spawning threads and the loss of cache warmup
  time outweigh the parallelism benefit.

**Example** — small obs dataset, large grid:

```python
# 30 observations, 50 000 grid blocks, nmax=20
# Most blocks share the same 20 nearest neighbours.
k.solve(nthread=2)          # try fewer threads
stats = k.solver_stats
print(stats["chol_fact"], stats["chol_reuse"])
# Tune nthread until chol_fact is minimised relative to chol_reuse.
```

### When to keep `nthread` high

More threads always helps when:

- The observation dataset is large and varied — each block has a largely
  unique neighbourhood, so the cache barely fires anyway.
- The grid is large and `nmax` is small — per-block cost is low, and
  parallelism savings dominate over any cache benefit.

---

## `ncache` — per-thread cache size

```python
k.solve(ncache=None)   # None (default): use compiled-in value (64)
k.solve(ncache=0)      # disable the multi-slot cache entirely
k.solve(ncache=128)    # give each thread 128 slots
```

`ncache` sets the number of factorization slots in each thread's hash cache.
Increase it when:

- The unique neighbourhood count per thread exceeds the default 64 — visible
  as `chol_fact` ≫ 64 per thread.
- `nmax` is large (large factorizations worth caching) and memory allows.

Memory cost per thread is roughly
`ncache × nmax² × 4 bytes` (single-precision L factor) plus bookkeeping.
With `ncache=64` and `nmax=50`, that is about 640 KB per thread —
negligible.  At `nmax=500` it becomes 32 MB per thread, so reduce `ncache`
if memory is tight.

---

## Quick-reference decision table

| Situation | Recommendation |
|---|---|
| Small obs (< ~100), large grid, high `nmax` | Reduce `nthread`; check `solver_stats` |
| Large obs, large grid | Keep `nthread` high; `ncache` at default |
| Hit rate low despite high `ncache` | Neighbourhoods are all unique; cache cannot help |
| Hit rate high but `chol_fact` still large | Increase `ncache` or reduce `nthread` |
| Memory-constrained with large `nmax` | Reduce `ncache` |
| SGSIM | `nthread` is forced to 1 (sequential simulation); tuning does not apply |
