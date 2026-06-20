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

All OpenMP threads share one bounded, set-associative hash cache containing up
to `ncache` factorizations (default 256 total slots, set at compile time).
Per-bucket locks let one thread reuse a factorization produced by another
without serializing accesses to unrelated neighbourhoods.

### How the four-way shared cache works

The shared pool is divided into buckets, with exactly four slots (or **ways**)
in each bucket.  For example, `ncache=256` creates 64 buckets × 4 ways.
Each neighbourhood system follows this path:

1. The solver first checks the worker's single-entry `ctx%cache`.  This is the
   cheapest path and catches immediately repeated systems without touching a
   shared lock.
2. On a local miss, the neighbour indices, neighbour counts, system size, and
   local range/nugget modifiers are mixed into a hash.
3. The hash selects exactly one bucket.  The solver locks that bucket and
   examines its four ways.
4. A matching hash is only a candidate: the complete cache key is compared
   before reuse, so a hash collision cannot select the wrong factorization.
5. On a hit, the prepared factors are copied into the worker's `ctx%cache`,
   the bucket's local recency counter is updated, and the lock is released.

After a cache miss, the worker factorizes the system normally.  It then tries
to insert the result into the selected bucket.  An empty way is used first; if
all four ways are occupied, the least-recently-used way **within that bucket**
is replaced.  Eviction is local rather than global, so insertion and lookup
scan at most four entries.

Lookups wait briefly for their selected bucket because an existing factor can
avoid an expensive O(nmax³) factorization.  Inserts use a non-blocking lock:
if another worker currently owns that bucket, the new factor is simply not
inserted.  Skipping a contended insert affects only future hit rate, never the
current result or numerical correctness.  Different buckets have independent
locks and remain fully concurrent.

Because the cache is set-associative, capacity and collision pressure are
related but distinct.  A bucket can retain at most four keys that map to it,
even when other buckets are empty.  Increasing `ncache` adds more buckets,
which both increases total capacity and spreads hashes across more lock and
eviction domains.

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

The block loop uses dynamic scheduling.  The shared cache preserves
factorization reuse even when matching neighbourhoods are processed by
different workers.

### When to reduce `nthread`

The main reasons to reduce threads are scheduling overhead and memory-bandwidth
contention.  This matters when:

- **`nmax` is large** — factorization and solve traffic can saturate available
  memory bandwidth.
- **The grid is small** — when the total number of blocks is close to the
  thread count, the overhead of spawning threads can outweigh the parallelism
  benefit.

**Example** — small obs dataset, large grid:

```python
# 30 observations, 50 000 grid blocks, nmax=20
# Most blocks share the same 20 nearest neighbours.
k.solve(nthread=2)          # try fewer threads
stats = k.solver_stats
print(stats["chol_fact"], stats["chol_reuse"])
# Tune nthread for elapsed time; the shared cache avoids cross-thread duplicates.
```

### When to keep `nthread` high

More threads usually help when:

- The observation dataset is large and varied — each block has a largely
  unique neighbourhood, so the cache barely fires anyway.
- The grid is large and `nmax` is small — per-block cost is low, and
  parallelism savings dominate over any cache benefit.

---

## `ncache` — total shared cache size

```python
k.solve(ncache=None)   # None (default): use compiled-in value (256)
k.solve(ncache=0)      # disable all factorization reuse
k.solve(ncache=4)      # smallest pool: one bucket with four ways
k.solve(ncache=128)    # 128 slots shared by all threads
```

`ncache` sets the total number of factorization slots in the shared hash cache.
Positive values from 1 through 3 are promoted to 4 because a bucket has four
ways.  Other values are rounded down to a complete bucket, so `ncache=5`,
`6`, or `7` currently allocates the same four-slot pool as `ncache=4`.
Increase it when:

- The working set of recurring neighbourhoods exceeds the default 256 slots.
- `nmax` is large (large factorizations worth caching) and memory allows.

The pool has a 64 MB total cap.  Slot sizing includes the normal Cholesky
storage, neighbour keys, and the worst-case lazy SSYTRF fallback factors, so
the runtime may allocate fewer slots than requested when `nmax` is large.  The
final allocation is also rounded down to a complete four-way bucket.  The pool
exists only for the duration of one `solve()` call; the optional persistent
factor cache is a separate one-entry cache used across solves.

### Automatic shrinking

`solve()` shrinks the requested `ncache` whenever the problem structure proves
the extra slots cannot produce a hit, so you rarely need to tune it down by
hand:

- **Fixed neighbour set** — when every variable fits within `nmax`
  (`need_search` is false, i.e. no subset search) and the local range/nugget
  modifiers are uniform across blocks, every block assembles the *same* system.
  The single always-on slot covers it, so `ncache` collapses to **1**.
- **Locally-varying variogram** (`varying_vgm=True`) — caching is disabled
  entirely, so `ncache` collapses to **0**.

The shrink only ever *lowers* the value you asked for; it never reduces the
achievable hit rate.  Cross-validation and SGSIM keep the full requested cache
because their neighbour sets change block to block.

Likewise, `nthread` is capped at the number of blocks — there is no point
spawning more workers than there is work.

---

## Quick-reference decision table

| Situation | Recommendation |
|---|---|
| Small obs (< ~100), large grid, high `nmax` | Benchmark `nthread`; shared-cache reuse is preserved |
| Large obs, large grid | Keep `nthread` high; `ncache` at default |
| All obs fit `nmax`, uniform variogram | `ncache` auto-shrinks to 1 (one system reused) |
| Hit rate low despite high `ncache` | Neighbourhoods are all unique; cache cannot help |
| Hit rate high but `chol_fact` still large | Increase `ncache` |
| Memory-constrained with large `nmax` | Reduce `ncache` |
| SGSIM | `nthread` is forced to 1 (sequential simulation); tuning does not apply |
