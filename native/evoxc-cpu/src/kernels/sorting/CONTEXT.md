# sorting — the `sort` / `argsort` kernels of evoxc-cpu

## Intent

`sort` and `argsort` follow `evoxc/backends/numpy/kernels/sorting.py`: `axis` (default `-1`) selects the sorted axis, `descending` is executed as the write-side flip of the ascending result, NaN sorts last ascending, and a complex operand is an explicit `BackendError` (numpy would order it lexicographically).
One code path serves both numpy kinds: every segment is ordered by a `(rank_key, original index)` total order, which IS the stable order and a valid result for numpy's unspecified quicksort ties, so `stable` is validated and ignored.
The result shapes always come from the graph (`util::result_shape`); the engine re-validates every buffer against the declared types afterwards.

## API Surface (crate-internal)

- `mod.rs::register` — inserts `"sort"` (`sort_kernel`) and `"argsort"` (`argsort_kernel`); both share `run`.
- `mod.rs::run(spec, input, out_shape, index_dtype)` — attribute validation, axis normalization, `permutation`, then `permuted_storage` (`sort`) or `index_storage` (`argsort`).
- `mod.rs::complex_unordered` — the complex rejection other modules also call.
- `permutation.rs::permutation` — dtype dispatch into `sorted_positions<T: RowKey>`, the one monomorphized per-segment core.
- `permutation.rs::permuted_storage` / `gather_segments` — the `sort` output; `permutation.rs::index_storage` — the `argsort` output cast into the declared dtype.
- `network.rs::order_keys` — the packed-key ordering entry point; `network_width` (2/4/8/16 pad or `0`), `NETWORK_MAX = 16`, `PAD_KEY = u64::MAX`.
- `keys.rs::RowKey` — the monotone key per dtype; `NARROW` (key < 2^32, so `(key, slot)` packs into one `u64`) holds for all dtypes except `i64`/`u64`/`f64`; `STACK_KEYS = 64`.
- `test_support.rs` — the independent `sort_by` reference oracle plus the direct kernel drivers.

## Segment ordering

`RowKey::NARROW` dtypes: one `u64` per slot = `(rank_key << 32) | slot`, ordered by `order_keys` (the extent's own 2/4/8/16-key Batcher network for extents 2..=16, `sort_unstable` above), and the low 32 bits are read back as the stable position.
Wide dtypes (`i64`/`u64`/`f64`): a `u64` key plus a `u32` index vector sorted by `sort_unstable_by` on `(key, index)` — the same total order with a comparator indirection.
The key built is one branch-free bit transform; `float_key32`/`float_key64` fold `-0.0` onto `+0.0` and map NaN to the maximum key, so NaN sorts last and `sort` gathers the operand's own bit patterns (a NaN payload survives).

## Layout walk

`inner == 1` (contiguous axis) reads each segment as `&all[base..base + extent]` and writes `perm[base..base + extent]` with one `copy_from_slice` per segment.
`inner > 1` gathers each segment into one reused `scratch: Vec<T>` (allocated on the first segment), and `gather_segments` copies the whole operand (`values.to_vec()`) and assigns every strided slot in place.
The walk is deliberately hand-written flat base/stride arithmetic: an ndarray `ArrayViewD`/`lanes(Axis)` expression measured ~37 ns/segment against a ~7 ns core (262144×2 7.1 → 44.8 ns/segment, 131072×8 22.3 → 59.7, strided 131072×4×2 17.2 → 65.4), so ndarray is not used here.

## Per-call cost model

Heap allocations per `sort`/`argsort` call, all plain `Vec`s (never the recycled storage pool, which only intercepts `Storage` drops):
- two rank-sized `Vec`s for the result shape (`exec.rs::eval_shape` `Vec<Option<usize>>` + `kernels/util.rs` `Vec<usize>`), plus `out_shape.to_vec()` in `run` — three rank-sized allocations;
- `permutation.rs:60` `vec![0u32; all.len()]` — INPUT-proportional (4 bytes per element), zero-filled although every slot is overwritten whenever extent ≥ 2 and both outer/inner are non-zero;
- `permutation.rs:88-90` three heap resizes (`keys` `u64`, `order`/`positions` `u32`) — SEGMENT-extent-proportional, only when `extent > STACK_KEYS`;
- `permutation.rs:93` `scratch` — SEGMENT-extent-proportional, allocated once on the first `inner > 1` segment;
- `permutation.rs:233` `Vec::with_capacity(perm.len())` (`sort`) — INPUT-proportional;
- `permutation.rs:247` `values.to_vec()` (`sort`, `inner > 1` only) — INPUT-proportional;
- `permutation.rs:264-289` the `index_storage` `.collect()` (`argsort`) — INPUT-proportional, 8 bytes per element for the frontend's `int64` result;
- `mod.rs:117-121` `out_shape.to_vec()` and `single`'s `vec![Rc::new(..)]` + one `Rc` allocation.
The three stack scratch arrays (`[0u64; 64]`, `[0u32; 64]`, `[0u32; 64]`, ~1 KiB) are zero-initialised on every call whatever the extent.

Elements-independent per-call overhead: the three rank-sized shape allocations, three `Attrs` string lookups (`axis`, `descending`, `stable` — a linear `Vec<(String, AttrValue)>` scan), `reject_complex`, the `Buffer::new` dtype/product check, and the `single` result vector (roughly 150–300 ns; every kernel pays the `util::result_shape` part, so `step.fast`'s precomputed static shape is not used here).

Segment counts: `(100,4)` = 100 segments of extent 4; `(100,50,4)` along the last axis = 5000 segments of extent 4; `(4096,64)` along the last axis = 4096 segments of extent 64 (extent 64 is the last stack-resident size); `(5000,)` = one segment of extent 5000 (heap scratch, `sort_unstable`).

## Constraints

- Depends on `evoxc-core`, `crate::buffer`, `crate::exec`, `crate::kernels::util`, `crate::plan` — never Python or a compiler backend.
- No `unsafe` anywhere: the packed-key trick is what buys an integer sort without intrinsics.
- Result types/shapes come from the graph; never re-infer. Axis extent above `u32::MAX` is an explicit `ShapeError` (positions are `u32`).
- The ordering is a total order, so the algorithm choice cannot change values: a swap between the network and `sort_unstable`, or between scratch layouts, keeps ties in original-index order.
- Never reintroduce an ndarray per-lane view in the segment walk (measured slower); ndarray stays the layout engine for the broadcast/reduction paths.

## Known Issues

- `permutation.rs:233` allocates `Vec::with_capacity(perm.len())` and `permutation.rs:247` immediately shadows it with `values.to_vec()` on the `inner > 1` path: one input-proportional allocation per `sort` call is allocated and dropped unused.
- Sorting scratch (`perm`, `keys`, `order`, `positions`, `scratch`) is plain `Vec`, so it never reaches the thread-local storage pool and is malloc/freed on every call; a per-run scratch buffer (or pooled storage for the output vectors) would remove it without changing values, since the main path fully overwrites `perm`.
- A graph that needs both `sort` and `argsort` of the same operand orders it twice: each op call rebuilds the keys, runs the ordering and allocates its own `perm`; nothing is shared between the two calls.
- `sort` materializes the full input-sized `perm` (`u32` per element) and then gathers through it; sorting the values directly per segment (with the same `(key, slot)` tie-break) would avoid that buffer and one full pass, at the cost of a per-segment value gather.
- The wide-key arm (`i64`/`u64`/`f64`) sorts through a comparator closure over an index vector; it is the arm most TensorNEAT graphs hit (int64 identity tables) and it has no recorded measurement in this module.
- `gather_segments` on `inner > 1` walks the operand twice beyond the sort (per-segment gather into `scratch`, then a whole-array `to_vec`, then the strided assignment).
- The per-op overhead inherited from the engine is not amortised here: `util::result_shape` re-evaluates and re-allocates the declared shape even though `plan::Step::fast` already holds the static `(dtype, shape)`.

## Routing Table

Leaf module — no child directories; `test_support.rs` is the shared test-only oracle, `network.rs`/`keys.rs`/`permutation.rs` are private submodules of `mod.rs`.
