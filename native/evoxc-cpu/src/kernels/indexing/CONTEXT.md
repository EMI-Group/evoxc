# indexing — `slice` / `gather` / `scatter` / `reshape` / `concatenate` kernels of `evoxc-cpu`

## Intent

This node holds the data-movement kernels: `reshape`, `transpose`, `slice`, `gather`, `scatter`, `concatenate`.
`strided_copy.rs` is the ONE two-sided run copier they share: an innermost-first run plan (`CopyPlan`/`AxisCopy`/`RunCopy`, collapsed by `collapse_steps` where strides line up) that is a run map, never a per-element odometer.
`copy_view` is the ndarray-`assign` fallback the copier declines into; every fast path here is a documented specialization whose ndarray expression is the reference in `strided_copy_tests.rs`.
`mod.rs` owns the shared vocabulary: `resolve_index_fast` (i64 fast path, else `resolve_index` over `i128`), `reconcile` (declared-vs-derived shape), the block builders, and the two destinations `pooled_total` (zero-filled on a pool miss, exact-size stock otherwise) and `pooled_append` (stock with its length reset or a fresh reservation, never filled).

## Fast paths

- `slice` (`slice.rs:151`, `slice_typed` `:172`): ndarray `array_view` + `slice_axis_inplace` per axis, then `pooled_total` and, in order, `view.as_slice()` → one `copy_from_slice` (the whole-array / step-1 contiguous case), `strided_copy::copy_view`, else `assign`.
- An all-extent-1 last-axis narrowing (`(q,m,4) → (q,m,1)`) collapses to ONE strided `Step` and rides `fill_strided` (`LANES = 4`), the same lane-grouped loop the pre-ndarray copier used — measured 0.2–0.33 ns/element, at parity.
- `gather` (`gather.rs:57`): `gather_runs` (`:101`) first — the index is resolved inline inside `append_runs`/`place_runs` with no intermediate `Vec<usize>` and takes `pooled_append` when the destination positions continue in order; only a layout `plan_axis_runs` declines goes to `resolve_indices` (`:151`) + `gather_typed` (`:178`, a per-index `AxisCopy` replay with an `assign` fallback).
- `scatter` (`scatter.rs:99`): `scatter_runs` writes one `copy_from_slice` per run; the general `scatter_typed` (`:354`) walks an ndarray broadcast index expression per element over a `pooled_total` copy of the source.
- `reshape` (`reshape.rs:43`) and an identity `transpose` (`:126`) are `block_storage_filled` (one bulk memcpy, unchanged); a non-identity `transpose` builds a `permuted_axes` view and hands it to the copier.
- `concatenate` (`concatenate.rs:129`) allocates ONE pooled destination and lays parts down through `assemble` (`:163`) with `BULK_RUN_ELEMENTS = 8`, a whole-row `copy_from_slice` per part, and a block-at-a-time `interleave` for the one-element-run family.

## Constraints

- Depends on `evoxc-core` plus `crate::buffer`/`crate::kernels::util`; no Python, no PyO3, no device APIs.
- Every destination is fully written: `pooled_total` zero-fills on a pool miss and `pooled_append` only hands out a stock whose length it reset, so no stale element can reach a result (pinned).
- Never a second layout engine: new addressing is an ndarray view/`Zip`/`indices` expression or a whole-run map, never a per-element stride odometer; negative strides are the copier's documented decline into `assign`.
- No `unsafe`; user errors are `Result` values (`ShapeError`/`BackendError`), never panics, and an index outside the gathered extent is an explicit error.

## Known Issues / Notes for Agents

- **`RUN_MEMCPY_MIN = 32` penalizes the mid-length run families.** `fill_run` (`strided_copy.rs:576`) uses `copy_from_slice` only at runs of ≥32 elements and an element loop below it, and `append_runs` (`:205`) likewise uses `extend`/push below it.
  The destination-contiguous 4-of-N slice (`(q,m,5) → (q,m,4)`, 25k four-element runs) measures **1.10 ns/element** in `bench_slice` while the same family reaches **0.38 ns/element** with one `copy_from_slice` per run in the pre-ndarray block copier (evoxc-cpu profile of `xor_p500` at `e07bac8`: `slice f32 (500,50,4)`, 40 calls, 1.532 ms / 100k elements).
  The gate is set for the 1–8 element gather runs it was tuned on; it is likely wrong for 4–31 element slice runs — re-measure before trusting the constant.
- The flat-table gather with a small run (`A = 2..8`, e.g. `(500,100,2)`) measures 1.08–1.10 ns/element in `bench_gather` against 0.94 ns/element for the same family in the `e07bac8` profile — the same short-run gate plus a per-index `extend`/push.
- The one-element-per-index gather family (`(500,100)`, `bool (50000)`) still "spreads between runs" in `bench_gather` and is not pinned; it is the largest gather family in the downstream cartpole graph (`gather (5000,) with a (100,50,4)` index), so its per-call/per-element cost is the least-characterized hot number here.
- Destinations below `POOL_MIN_BYTES` (4 KiB) are never retained by the pool, so every small result (`(q,1)` slices, `(100,4)` gathers) mallocs; `take` also scans up to `POOL_MAX_ENTRIES` (64) entries before reporting a miss. Pre-existing, not introduced by the copier.
- A rank-0 `gather`/`scatter` drops the axis and a rank-0 `slice` result goes through `pooled_total(1)` rather than a direct one-element `vec!`; the extra cost is the view/plan setup, tens of ns per call, multiplied by every executed small op.
- `scatter` computes `selection_shape` twice per call (`scatter.rs:109` and `:170`) and clones the source shape (`:117`, `:137`); one call per generation in every known downstream graph, so it is not hot.
- Test gaps: `strided_copy_tests.rs` pins both short- and long-run arms for correctness but nothing pins the `RUN_MEMCPY_MIN` threshold's performance; there is no bench case for a 100-element (`(100,4) → (100,1)`) slice or for the `(100,50,4) → (100,50,1)` element count that the downstream cartpole graph executes thousands of times per run.

## Routing Table

| Path | Area |
|---|---|
| `./mod.rs` | Registry, `resolve_index_fast`/`resolve_index`, `reconcile`, `pooled_total`/`pooled_append`. |
| `./slice.rs` | `slice_ranges`/`DimSlice`, `slice`, `slice_typed` (view + `copy_from_slice` / copier / `assign`). |
| `./strided_copy.rs` | `CopyPlan`/`AxisCopy`/`RunCopy`, `plan_axis_copy`, `plan_axis_runs`, `copy_at`, `append_runs`, `place_runs`, `copy_view`, `fill_run`, `fill_strided`, `collapse_steps`. |
| `./strided_copy_tests.rs` | `#[path]` child test module pinning the copier against ndarray `assign` and a per-element reference. |
| `./gather.rs` | `gather`, `gather_runs`, `resolve_indices`, `gather_typed`. |
| `./scatter.rs` | `scatter`, `scatter_selection`, `scatter_into`, `scatter_runs`, `scatter_typed`. |
| `./reshape.rs` | `reshape` (bulk memcpy), `transpose` (identity memcpy / `permuted_axes` view + copier). |
| `./concatenate.rs` | `concatenate` (one pooled destination, `assemble`, `BULK_RUN_ELEMENTS`). |
| `./bench_{gather,slice,concatenate}.rs` | `#[ignore]`d per-element instruments; `bench_gather` also times `scatter`. |
| `./test_support.rs` | Shared `#[cfg(test)]` builders. |
