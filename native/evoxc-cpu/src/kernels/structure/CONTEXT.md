# structure — `tile` / `flip` / `roll` / `pad` / `diag` kernels

## Intent

Materialize the five position/shape structure ops of the dense native CPU engine from the operand into a fresh C-contiguous `Storage`.
Every op is an `ndarray` expression over a view of the operand, with ONE measured exception: `pad`'s row-wise pooled copy (`pad.rs`), which beats the equivalent ndarray slab-fill-plus-`assign` expression and is pinned against it in `pad`'s tests.
`mod.rs` owns the registry, the single `structure_kernel` entry point, the attribute readers and the two module-wide helpers (`pooled`, `storage_of`); each op lives in its own file.

## API Surface

- `register(map)` — installs `"tile"`, `"flip"`, `"roll"`, `"pad"`, `"diag"` → `structure_kernel`.
- `structure_kernel(ctx, spec, operands)` — resolves the declared result shape (`util::result_shape`) then calls `structural`.
- `structural(spec, input, out_shape)` — the `spec.name`-selected materialization, returned as one `Value` through `exec::single`.
- `mod.rs` helpers: `pooled::<T>(total)` (pooled result vector, zero-filled on a pool miss), `storage_of::<T>(ArrayD)` (standardize + move an ndarray allocation into `Storage`), `as_json`/`int_list`/`int_of` (attribute readers over the typed `AttrValue`), `identity_shape`/`same_shape`.
- `tile.rs`: `reps`, `tile_whole_repeat` (leading-axis-only repeat → one `repeat_storage`), `tile_storage` (the reference broadcast expression), `tile_promoted` (shape validation).
- `flip_roll.rs`: `flip_axes`, `flip_storage` (`slice_each_axis` with a `-1` step), `roll_storage` (two-slice assignment for ≤1 rolled axis, `concatenate` per axis beyond that), `roll_amounts`.
- `pad.rs`: `padding`, `pad_value`, `pad_result`, `pad_storage` (row copy; `ndarray_pad_storage` is the test-only reference).
- `diag.rs`: `diag_storage` (`diag`/`diag_mut` views; rank 1 zero-fills then assigns the diagonal).

## Constraints

- Values and error wording are the engine contract: every fast path is pinned against an independent per-element reference and/or the ndarray expression in the same file's `#[cfg(test)]` tests; error kinds and message text must not change.
- Result buffers are always fresh C-contiguous `Vec`s (`Storage` has no views), so each op materializes; a rank-1 `diag` must zero-fill before assigning the diagonal.
- `pad` is the only hand-written row loop; it validates the config against the declared result shape before allocating, and its two degenerate configurations are flat copies (all-zero config = one bulk copy, rank-0 operand = one element).
- Attribute schemas (generated table): `tile.reps` = `ints`, `roll.shift` = `ints`, `pad.padding_config` = `nested_ints`, `pad.value` = `float`, `flip.axes` / `roll.axis` = `any` (nullable), `diag` has none.
- Kernel signature is `(ctx, &OpSpec, &operands)`; the plan's precomputed `Step.fast` static extents (`plan.rs`) are not passed to kernels, so the declared result shape is re-evaluated per call.

## Known Issues (per-call fixed cost — measured/estimated, current state)

- `tile.rs:163-173` — the tile result is one `dst.assign(&broadcast)` over a stride-0 view: a per-element assign (~0.5–1.5 ns/element) rather than the run-wise block copy still available as `buffer::block_storage_filled`; a tile on a `(100,50,4)` f32 operand pays ~10–30 µs of kernel time where run copies would be ~1–2 µs.
- `flip_roll.rs:36-51` — `dst.assign(&flipped)` walks reversed-axis views per element; a flip of an outer axis could be one memcpy per outer row (~0.06–0.1 ns/element) instead of ~1 ns/element.
- `flip_roll.rs:129-139` — roll on ≥2 non-zero axes does `to_owned()` + one `concatenate` per axis + `storage_of`'s possible `as_standard_layout` copy: up to four full-array allocations/copies per call.
- `mod.rs:95` → `util::result_shape` → `dims::eval_shape` + `util::result_shape` — two `Vec` allocations per call re-derive a shape the plan already holds statically (`Step.fast`); ~40–80 ns/call (same pattern in the indexing/reductions/linalg/sorting families).
- `mod.rs:199-235` (`as_json` + `int_list`) — attributes are re-rendered through `serde_json` per call (tile 3, flip 3, roll 5–7, pad 5 allocations for a rank-3 operand) although they are immutable per plan; typed accessors (`AttrValue::as_ints`) and direct `NestedInts` matching would allocate nothing, and decoding once into `Step` (like `Step.constant`) removes the cost entirely.
- `mod.rs:157-159` (`pooled`) — `vec![T::zero(); total]` zero-fills a destination that tile/flip/roll/pad fully overwrite, and the storage pool refuses stocks under `POOL_MIN_BYTES` (4 KiB), so the `(100,1)` / `(100,4)` shape families allocate and free fresh every call.
- `pad.rs:196` (`util::c_strides`) and `pad.rs:201` (`vec![0; outer.len()]`) — rank-sized `Vec`s per call where a `[usize; 8]` stack array suffices.
- `tile.rs:141-161` — up to five rank-sized `Vec`s per tile call (`source_shape`, `interleaved`, `repeated_slots`, plus `reps`/`tile_promoted`), all stack-array-able.
- No `#[inline]` appears in this node and `with_type!` expands each storage helper 14× (one per dtype), so the family costs code size / icache on a many-small-op graph with no runtime allocation.
- Downstream frequency: tensorneat's NEAT evaluation graph emits no structure op (`evoxc.flip` only in a docstring, `evoxc.tile` only in FuncFit, `evoxc.pad` only in the mutation-plan helper), so this node is not on the cartpole `Pipeline.step` hot path.

## Routing Table

| Path | Area |
|---|---|
| `./mod.rs` | Registry, `structure_kernel`/`structural` dispatch, `pooled`/`storage_of`, the attribute readers (`as_json`/`int_list`/`int_of`), `identity_shape`/`same_shape`. |
| `./tile.rs` | `tile`: promotion/validation, whole-source `repeat_storage` fast path, broadcast reference expression. |
| `./flip_roll.rs` | `flip` (reversed `slice_each_axis` view) and `roll` (flattened, single-axis two-slice, multi-axis `concatenate`). |
| `./pad.rs` | `pad`: config/value reading, validation, the row-wise pooled copy. |
| `./diag.rs` | `diag`: rank-1 matrix build and rank-2 diagonal extraction. |
| `./test_support.rs` | Shared `#[cfg(test)]` fixtures (`op`, `ints`, `buffer`, `run`, `message`). |
