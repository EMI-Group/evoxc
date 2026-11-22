# buffer — the engine's value representation and its ndarray layout substrate

## Intent

`buffer` owns the execution engine's materialized value ([`Buffer`]) and every N-dimensional addressing rule the kernels use.
A `Buffer` is a dtype, a concrete shape and exactly that dtype's typed `Vec` — always C-contiguous, never a view, so a kernel is a flat loop over plain slices.
`ndarray` is the ONLY N-dimensional layout engine in this module: there is no hand-written stride odometer anywhere under `src/buffer/`.
The kernels in `../kernels/` keep their hand-written contiguous typed-slice fast paths; everything that has to address a broadcast or strided layout goes through the drivers here.
The reference behaviour is the numpy interpreter in `../../../evoxc/backends/numpy/`; this module must reproduce it bit for bit where the contract says so.

## API surface

`Buffer { pub dtype: DType, pub shape: Vec<usize>, pub storage: Storage }` is the public representation and must stay exactly those three fields (the PyO3 node builds and reads them).
- `Buffer::new` validates dtype-vs-storage and element count; `zeros`, `from_payload` (little-endian envelope constants), `cast` (numpy `astype`), `with_shape` (relabel only), `numel`, `rank`, `is_flat_for(shape)` (the flat-fast-path gate).
- `Storage` is one `Vec<T>` variant per one of the 14 dtypes (`bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`).
- Shape algebra: `broadcast_shape(shapes) -> Result<Vec<usize>>` (numpy right-aligned rule, message `"operand shapes cannot broadcast: (3) vs (2)"`) and `broadcast_strides(shape, out_shape) -> Vec<usize>` (0 on broadcast axes). Both are pure arithmetic, deliberately not ndarray.
- `broadcast_to(&Buffer, shape) -> Result<Buffer>`: equal shape returns a clone; otherwise it errors unless `broadcast_shape` of the pair is exactly `shape` (message `"operand shape (1, 3) cannot be broadcast to target shape (3)"`), then materializes per storage variant through `fill_broadcast`.
- Layout drivers, all `crate::buffer::…` paths: `walk(rank, out_shape, strides, body)`, `apply1/apply2/apply3(out, out_shape, operands + their shapes, f)`, `fill_broadcast<T: Scalar>(values, shape, out_shape) -> Vec<T>`, plus the compatibility aliases `apply2_runs`/`apply3_runs`.
- Flat typed-slice fast paths (measured specializations, not layout engines): `zip1/zip2/zip3` (write into caller memory), `zip2_accumulate` (in-place), `map1/map2/map3` (allocate or pool a fresh result), `map2_runs`/`map3_runs` (broadcast operands, pooled result through the repeating-run driver or `apply2`/`apply3`).
- The repeating-run driver (`runs2`/`runs3`/`copy_runs`, with `repeat_view`/`run_block`/`run_source`/`walk_blocks`/`run_chunks` behind them): the ONE measured specialization beside the view drivers, entered from `zipmap`'s `map2_runs`/`map3_runs` and from `fill_broadcast`. Its knobs are `RUN_TILE_MAX` (256), `RUN_BLOCK_TARGET` (64), `RUN_MIN_ELEMS` (64) and `RUN_SPLIT_MIN_ELEMS` (8 × `PARALLEL_MIN_ELEMS` = 2^21); `#[cfg(test)]` `RUNS_DISABLED` is the A/B switch `bench_broadcast_zip` times it through.
- Block/repeat copy builders: `block_storage`, `block_storage_filled`, `copy_blocks`, `repeat_storage`.
- Thread split (module-private): `parallel_threads`, `split_parallel`, `window`, with the measured thresholds `PARALLEL_MIN_ELEMS` (2^18), `PARALLEL_MIN_PER_THREAD` (2^17), `PARALLEL_MAX_THREADS` (4), plus `rowsplit`'s `RowSplit`/`Rider`/`split_rows` for the pooled broadcast builders.
- Storage pool (module-private, crate-visible `take_pooled_vec`): a thread-local free list with `POOL_MIN_BYTES` (4 KiB) admission, `POOL_MAX_ENTRIES` (64), `POOL_MAX_BYTES` (64 MiB) and `POOL_MAX_PER_SIZE` (8) per exact size class; `Drop for Storage` is the admission hook.
- ndarray seam: `trait NdScalar` (14 impls) and `Buffer::{as_array, as_array_mut, as_array2, from_array, into_array}`, plus the free `array_view`/`array_view_mut(values, shape)` for a kernel that already holds a dispatched typed slice.
- Dispatch macros: `with_type!`, `with_same_type!`, `with_same_type_mut!` (they match the storage enum once and keep a kernel body generic over `T: Num`).

## The ndarray view seam (`ndarray_views.rs`)

`Storage` stays owned and `Vec`-backed, so a view is a transient borrow: `as_array`/`as_array_mut`/`as_array2` borrow the buffer, `from_array` standardizes a non-standard layout (`.as_standard_layout().into_owned()`) and takes ownership, `into_array` MOVES the `Vec` into `ArrayD::from_shape_vec` (no copy for a C-contiguous buffer).
`NdScalar` is the per-dtype bridge the seam needs: `DTYPE`, `into_storage`, `take_storage` (by-value extraction — `Storage` implements `Drop`, so a `Vec` is taken with `std::mem::take`, still a move), `slice_of`, `slice_of_mut`.
A dtype mismatch is always `Error::dtype` naming both dtypes; a shape/length mismatch is `Error::shape`; none of these functions panics.
This seam exists for the PyO3 FFI handoff and for kernels that hold a typed slice and want ndarray semantics; nothing in the engine's hot path depends on it.

## Which path is which

ndarray-reference paths (the layout authority):
- `walk` iterates `ndarray::indices(shape)` and reads each position via `Dimension::slice`, mapping the strides to one offset per operand — the single fallback and the reference for strided addressing.
- `apply1`/`apply2`/`apply3` are one `ndarray::Zip` pass over static-dimension broadcast views writing an `ArrayViewMut`; `a[i]` reads element `i` of each broadcast operand, so the caller's closure owns every semantic (NaN propagation, signed zeros, wrapping).
- `fill_broadcast` and `broadcast_vec` build one broadcast view and materialize it with `ArrayViewMut::assign` / an iterator collect; `broadcast_to` dispatches them per storage variant.
- `map2_runs`/`map3_runs` are `apply2`/`apply3` with a pooled result vector of the exact length (never zero-filled), and a result of at least 2^18 elements is split over scoped threads on its OUTERMOST axis (row-aligned chunks, the leading chunk on the calling thread; an operand that cannot ride a row chunk — a higher rank, a disagreeing outer extent, a slice whose length is not its element count — sends the whole result back to the single-threaded driver).

Measured typed-slice specializations, deliberately kept:
- The repeating-run driver (`broadcast.rs::runs2`/`runs3`, with `copy_runs` behind `fill_broadcast`): an operand whose broadcast view reduces to a contiguous run repeating across the result — result element `i` reads operand element `i % period`, which is the `(n,)` bias, the `(1, …, 1, n)` operand and a whole-storage repeat — is written as whole blocks out of one tile instead of a per-element broadcast `Zip`. `repeat_view` proves the layout EXACTLY from the shapes and broadcast strides (`stride_k == cstride_k % period` for every axis, so the offsets telescope to `i % period` with no approximation and no case table); the block is the lcm of the tiled periods scaled up to `RUN_BLOCK_TARGET` (64) and capped at `RUN_TILE_MAX` (256) elements; a period covering the whole result is read straight out of the operand (`RunBlock::Run`) instead of a tile, a period of 1 declines (a scalar operand keeps the `Zip`'s captured-scalar arm, which is faster) and so does a period above the tile cap. A layout it declines keeps the ndarray `Zip`/`assign` path, which stays the REFERENCE.
- The driver splits over block-aligned chunks only above `RUN_SPLIT_MIN_ELEMS` (8 × `PARALLEL_MIN_ELEMS`, 2^21 elements): a chunk start is a multiple of the block and therefore of every period, so each chunk reads its tile from the first element exactly as the serial walk does, and every chunk owns a disjoint output range. Below that gate the calling thread writes the whole result — the driver is an order of magnitude cheaper per element than the `Zip`, so the `Zip`'s own 2^18 threshold would spend ~10 µs of thread setup to save less than that.
- `map2_runs`/`map3_runs` therefore try the repeating-run driver BEFORE the row split: a layout it takes must not also be cut into `RowSplit` chunks (that measured ~0.17 ns/element against the driver's 0.12 on the `(4,)` case at 2^18 elements).
- `zip1..3` / `zip2_accumulate` / `map1..3` are monomorphized indexed or iterator loops over contiguous slices, with the closure (and therefore the op kind) dispatched once per call; a map result comes from the pool when an exact-size stock exists.
- A flat map over ≥ 2^18 elements is split over scoped threads (one per 2^17 elements, at most four); below that the calling thread runs it.
- `rowsplit.rs` splits a pooled broadcast result the same way but row-aligned: `RowSplit` (plane = the product of the trailing axes, rows = the outermost extent, at most `PARALLEL_MAX_THREADS` chunks) plus a `Rider` per operand — `Whole` for an operand that broadcasts over the outer axis (lower rank, or an outer extent of 1) and `Rows` for one with the result's outer extent, whose chunk is a contiguous `values[r0*plane .. +n*plane]` — and `split_rows` (`chunks_mut`, leading chunk on the calling thread) driving one `apply2`/`apply3` per chunk. A result that is not row-splittable keeps the single `apply2`/`apply3` call.
- `blocks.rs` copies whole contiguous runs (`copy_blocks` walks the destination with `chunks_exact_mut(inner)`), and `repeat_storage` fills a whole-source `tile` one run at a time.
- `pool.rs` recycles large result storages per thread with an LRU eviction policy; a pooled vector is only handed out for a full overwrite.
- `apply*` dispatch on the result rank to `Ix0..Ix6`, and a size-1 operand (rank 0 or an all-ones shape) is captured as a scalar and dropped from the `Zip`, leaving a single pass over the destination; `IxDyn` survives only in the one-per-call operand base view and the rank > 6 fallback.
- Any layout a view cannot express (a slice whose length disagrees with its shape, an operand of higher rank than the result, a non-broadcastable shape, an `out` slice of the wrong length) falls back to the `walk` loop, which keeps the pre-existing accept/reject/panic behaviour byte-identical.

## Measured cost

Protocol: `cargo test --release -p evoxc-cpu --lib bench_ -- --ignored --nocapture` (3 warmups, 15 iterations, median, 400 000 elements, shared host — expect run-to-run spread).
`bench_broadcast_zip` times `map2_runs` for out `(2000, 50, 4)` f32 (the `xor_p500` population shape) with an `add` closure; the driver's cost is closure-independent.
`bench_broadcast_zip_threads` A/Bs the identical code with the row split forced off and on (`rowsplit`'s `#[cfg(test)] SPLIT_DISABLED`) in one process, so the thread effect is separated from host drift.

| right operand | single result (split off) | default (row split, 3 chunks) | old hand-written run driver |
|---|---|---|---|
| `(4,)` bias | 1.11 ns/element, 0.444 ms | **0.455 ns/element, 0.182 ms** | 0.098 |
| `(50, 1)` per-row scalar | 1.14 ns/element, 0.455 ms | **0.439 ns/element, 0.176 ms** | 1.104 |
| `()` scalar | 0.088 ns/element, 0.035 ms | 0.096 ns/element, 0.039 ms | 0.093 |

`bench_walk` times the fallback/reference offset driver: rank 1 `[1000000]` contiguous 0.92, rank 2 `[1000,1000]` contiguous 1.20, rank 2 column-strided 1.02, rank 3 `[100,100,100]` contiguous 1.50 ns/element (the deleted odometer was 1.06–1.60).
Reading: the row split buys 2.4–2.6x on the two CPU-bound broadcast cases, which puts the per-row shape ahead of the deleted driver (0.44 vs 1.10) and leaves only the `(4,)` bias ~4.6x behind its memory-bandwidth tiling (ndarray's generic `Zip` cannot express that tiling); the scalar case is memory-bound and pays ~8 % (~4 µs) of thread setup for nothing, which is small against the 0.26 ms the same change buys. `walk` and the scalar case are at or better than the deleted driver.

## Constraints

- Depends on `evoxc-core`, `ndarray`, `half` and `num-complex`; no Python, no `unsafe`, no SIMD intrinsics.
- Never a second layout engine: new N-dimensional addressing is expressed with ndarray views/`Zip`/`indices`, never with a hand-written increment/decrement odometer.
- `Buffer`/`Storage` keep their owned representation and public fields; result buffers are always fresh C-contiguous `Vec`s (a shape-only op copies — that is the documented cost of having no views in `Storage`).
- Errors are explicit `Result` values with the crate's message conventions; no `unwrap`/`expect` on user input, no silent truncation of a short operand.
- Threads are scoped inside one call only (never a pool, never `rayon`), gated by the measured thresholds (element count for a flat map, row count for the pooled broadcast builders), and each thread owns a disjoint output range so results stay bit-identical.
- Pooled results are never zero-filled on the hot paths; a pooled stock is only taken when it is fully overwritten.
- One concern per file, target <~1000 lines including tests: the broadcast driver's independent reference tests live in `broadcast/tests.rs` as a sibling file.

## Test strategy

- `cargo test -p evoxc-cpu --lib` → 319 passed, 4 ignored (`bench_thread_crossover`, `bench_broadcast_zip`, `bench_broadcast_zip_threads`, `bench_walk`).
- The row split is pinned in `zipmap.rs`: `the_row_split_agrees_with_the_per_element_reference` (out `[4096,64]`, `[401,977]`, `[3,100000]`, `[1000,1000]`, `[262144]` against a leading-ones, trailing-bias, per-row-scalar, scalar and flat operand, compared element for element with an independent index-arithmetic reference), `the_row_split_agrees_with_the_per_element_reference_for_select` (`map3_runs` with a broadcast predicate), `the_row_split_covers_whole_rows_or_stays_whole` (the `RowSplit` geometry and which operands ride), `an_operand_that_cannot_ride_the_split_keeps_the_single_threaded_behaviour` (a rank-equal operand with a disagreeing outer extent behaves exactly as with the split off) and `a_split_result_is_fully_overwritten_from_the_pool` (a parked garbage stock of a 2^18-element result).
- `broadcast/tests.rs`: every driver is pinned against an INDEPENDENT naive per-element reference (`reference_slot`, plain index arithmetic) — never against another driver — over i32 and f32 and a rank/shape matrix (`[2,3]`, `[2,2,2]`, `[4,3,5]`, `[4096,64]`, `[1000,1000]`, `[3]`, `[7,6]`, `[5]`, rank 0, zero-length axes); ordering pins that make a transposed or shifted read visible; NaN/signed-zero/i32-wrapping pass-through; the size-1 arms for `apply2`, `apply3` and `fill_broadcast`; a one-element operand whose shape claims two elements must still fall back (its pre-existing panic is pinned); a pooled garbage stock of the exact result size is fully overwritten by `map2_runs`; and the view path agrees with the flat fast path (`map2`/`apply2`, `map3`/`apply3`).
- `ndarray_views.rs` tests: all 14 dtypes round-trip through `from_array`/`into_array`/`as_array`; a rank-0 buffer; a zero-element shape; a dtype mismatch naming both dtypes; a non-standard-layout `from_array`; and `into_array` moving the allocation (pointer identity).
- `pool.rs`, `blocks.rs`, `zipmap.rs`, `parallel.rs` and `storage.rs` keep their own pins (retention/eviction, in-place accumulation vs the allocating map, wide-vs-narrow bit identity, thread windows, LE decode).

## Known Issues

- The `(4,)` bias broadcast elementwise case is still slower than the deleted hand-written tiled driver (0.455 vs 0.098 ns/element); recovering it needs a specialized tiling driver, which the ndarray-only mandate excludes, so the trade is documented rather than fixed. The per-row-scalar case is now ahead of it (0.439 vs 1.104).
- A size-1 operand is only captured when its shape's element count is 1 AND its slice length is 1; a one-element slice with a larger declared shape is a malformed operand and keeps the `walk` fallback (which panics exactly as before).
- Result rank > 6 dispatches to the `IxDyn` arm: correct but slow, and no real graph reaches it.
- The broadcast builders' row split is capped at `PARALLEL_MAX_THREADS` (4) chunks regardless of result size, and it costs ~4 µs of thread setup on a memory-bound result (the `()` scalar case, −8 %); the flat `map1..3`/`zip*` paths split over `split_parallel` on element counts, the pooled broadcast builders over row-aligned chunks.
- `walk`'s index iteration costs 0.9–1.5 ns/element, which is what a non-identity `transpose` (in `../kernels/indexing/reshape.rs`) now pays per element.
- Shape-only ops copy because `Storage` owns plain `Vec`s (`reshape`, identity `transpose`, same-shape `broadcast`, a `constant` step's payload clone).
- The stride-copy machinery for `slice`/`gather` lives in `../kernels/indexing/stride_copy.rs` and is not part of this module.
- Stale doc links in `../kernels/elementwise/{binary,compare}.rs` still say the broadcast path is `buffer::stride`; that module no longer exists (kernels are out of this node's write scope).

## Routing Table

| Path | Area |
|---|---|
| `./mod.rs` | `Buffer` + inherent impl, `shape_text`, the three dispatch macros, the module wiring and every `crate::buffer::…` re-export. |
| `./storage.rs` | `Storage` over the 14 dtypes, little-endian payload decode, `cast_storage`/`convert_generic`. |
| `./broadcast.rs` | `broadcast_shape`/`broadcast_strides`, the static-dimension rank dispatch, `walk`, `apply1..3`, `broadcast_to`, `broadcast_vec`, `fill_broadcast`. |
| `./broadcast/tests.rs` | The independent-reference pins for every driver in `broadcast.rs`. |
| `./ndarray_views.rs` | `NdScalar` and the zero-copy `Buffer`↔`ndarray` seam (`as_array*`, `from_array`, `into_array`, `array_view[_mut]`). |
| `./zipmap.rs` | The flat `zip1..3`/`zip2_accumulate`/`map1..3` fast paths and the pooled broadcast builders `map2_runs`/`map3_runs` (with their per-element-reference and pool pins). |
| `./rowsplit.rs` | `RowSplit`/`Rider`/`split_rows`/`chunk_shape`: the row-aligned outermost-axis split the pooled broadcast builders use, plus the `#[cfg(test)]` `SPLIT_DISABLED` switch the A/B bench uses. |
| `./parallel.rs` | `PARALLEL_*`, `parallel_threads`, `split_parallel`, `window` and their crossover pins. |
| `./pool.rs` | The thread-local recycled-storage pool, its caps/LRU policy and `take_pooled_vec`. |
| `./blocks.rs` | Contiguous block copies (`copy_blocks`, `block_storage`, `block_storage_filled`, `repeat_storage`) behind the structure/indexing kernels. |
| `./bench_broadcast.rs` | The `#[ignore]`d `bench_broadcast_zip` (three operand shapes), `bench_broadcast_zip_threads` (the same cases with the row split forced off and on) and `bench_walk` measurement instruments (evidence, never asserts speed). |
| `./test_support.rs` | The shared `#[cfg(test)]` garbage-stock fixture. |
