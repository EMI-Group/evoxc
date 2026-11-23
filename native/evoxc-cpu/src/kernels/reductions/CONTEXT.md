# reductions — reduce_* / argmax / argmin / cumsum / cumprod kernels

## Intent

Implements the `reduce_*`, `argmax`/`argmin` and `cumsum`/`cumprod` kernels for the native CPU engine, with semantics mirroring `evoxc/backends/numpy/kernels/reductions.py`.
Result shapes are always the DECLARED graph shapes (`util::result_shape`); a kernel never re-infers one.
The shared ndarray substrate, the reference-vs-fast-path rule and the crate-wide cost model live in `../../../CONTEXT.md` (the `evoxc-cpu` contract); this file records only the reduction-specific contract.

## API Surface

- `register` (mod.rs) maps `reduce_sum/max/min/mean/prod` → `reduce_kernel`, `argmax`/`argmin` → `arg_kernel`, `cumsum`/`cumprod` → `cumulative_kernel`.
- `fold_block` (flat.rs) is the contiguous-block SPECIALIZATION; `fold_all`/`fold_first` + `for_each_lane` (layout.rs) are the ndarray lane-walk reference for every other layout.
- `FoldSelect` (layout.rs) is the NaN-propagating max/min comparator shared by the flat fold and the lane walk.

## Fast-path gates (current)

`reduce_*`: `Block::new(in_shape, reduced).filter(|b| b.inner > 0 && b.red > 0)` selects `fold_block` (mod.rs).
`Block` = one contiguous run of reduced axes; `inner == 1` folds each group's contiguous run with `fold_run`, `inner > 1` accumulates `inner` lanes row by row through one reused `acc` buffer.
Full reductions (all axes, incl. scalar output), rank-1 reductions, trailing-axis reductions and contiguous multi-axis blocks all satisfy the gate and fold flat; only a gapped reduced-axis pair, an empty reduced axis, a zero trailing extent, or a rank-0 operand (no reduced position) fall to the lane walk.
`argmax`/`argmin` use the same `Block` gate (`red > 0 && inner > 0`); `inner == 1` scans each run with `arg_run`, `inner > 1` scans per lane through reused `best`/`indices` buffers.
`ARG_BRANCHY_MIN_RUN` (1024) picks `arg_run_branchy` at or above that run length and `arg_run_branchless` below it; both return identical indices.

## Fold vocabulary

`fold_run` uses `LANE_COUNT = 8` independent accumulators for integer/bool/f32/f64 and a single sequential lane for f16/complex.
A run shorter than `LANE_COUNT` folds sequentially, so small reductions keep the interpreter's exact arithmetic.
`divide_by` turns a sum into `reduce_mean` (a zero count yields NaN, numpy's rule).

## Constraints / gotchas (current cost state)

- The flat block path allocates exactly one output `Vec` (plus, for `inner > 1`, one reusable `acc`); it is the hot path for real graphs and must stay allocation-stable.
- The general lane walk allocates one `axis_permutation` `Vec` and builds zero-copy views (`array_view` uses an inline `IxDyn` for rank ≤ 4, `permuted_axes` is a view), then recurses `axis_iter`; a strided leaf folds through `fold_iter`, which pays an `index % LANES` per element.
- That general path costs materially more than the flat fold on small per-lane runs; it is a reference/fallback, not a hot path — keep large real reductions on `Block`.
- `cumsum`/`cumprod` build `array_view` + `array_view_mut` and walk `lanes(Axis)`/`lanes_mut(Axis)`; `values.to_vec()` is the (pre-existing) full copy, and the two views add a small per-call setup cost over the old hand-rolled stride loop.
- No reduction materializes a permuted or broadcast copy; the only temporaries are the pre-existing `input.cast(acc_dtype)` and `cumulative`'s `to_vec()`.
- A dropped `sparse_*` op never reaches here (the plan rejects it); complex ordering for max/min/arg is an explicit `BackendError` from `util::reject_complex`.
