# kernels/elementwise — the pointwise kernel family of the native CPU engine

## Intent

Owns every pointwise op of `evoxc-cpu`: the arithmetic/bitwise/logical binary family and its in-place accumulation path (`binary.rs`), the comparisons and `select` (`compare.rs`), the unary family (`unary.rs`), `erf`/`gelu`/`nan_to_num` (`transcendental.rs`), and the ops belonging to no family — `register`, `stop_gradient`, `cast`, `broadcast`, the shared `bool_results` builder (`mod.rs`).
All arithmetic runs in the op's DECLARED result dtype; an operand is converted only when its dtype differs from it (the frontend already inserted compensation casts).
A comparison computes in the promoted INPUT dtype and produces `bool`; `abs` of a complex value produces a real result.
Layout probing, the broadcast drivers, the storage pool and the `map*`/`zip*` builders live one level up in `buffer/` (`../../buffer/CONTEXT.md`); this node only picks the arm.

## API surface (crate-internal)

- `register(map)` (`mod.rs:77`) — the single registration point; `binary_kernel` serves both the arithmetic and the comparison names.
- `mod.rs`: `bool_results` (`:58`, the one-byte result builder), `BOOL_RESULT_THREAD_MIN` (`:46`, 2^19), `stop_gradient_kernel` (`:132`), `cast_kernel` (`:140`), `broadcast_kernel` (`:153`).
- `binary.rs`: `binary` (`:233`), `binary_kernel` (`:90`), `in_place_kernel` (`:113`) with `in_place_left`/`in_place_right` (`:125`/`:136`) → `in_place_binary` (`:153`) → `accumulate_typed` (`:200`), `BinKind` (`:23`) + `bin_kind` (`:45`), `Layout`/`layout` probes (`:424`/`:430`).
- `compare.rs`: `compare` (`:38`), `compare_typed` (`:110`), `compare_op` (`:135`), `select_kernel` (`:225`), `select_numeric` (`:257`), `select_apply` (`:298`).
- `unary.rs`: `unary` (`:67`), `apply_un` (`:106`), `dense_unary` (`:157`), `abs_complex` (`:196`).
- `transcendental.rs`: `erf_kernel` (`:12`), `gelu_kernel` (`:37`), `nan_to_num_kernel` (`:100`).

## Dispatch map (arm actually taken)

Binary arithmetic (`add`/`subtract`/`multiply`/`divide`/`negate`-free family): `binary_kernel` → `binary` → `bin_kind` → `broadcast_shape` → (`with_same_type!`) → `typed_binary` (`binary.rs:388`) → `layout` × 2 (`:398`) → arm.
- (Flat, Flat) → `dense_binary` (`:451`) → `buffer::map2` → `zip2_slice` (indexed loop).
- (Flat, Scalar) / (Scalar, Flat) → `dense_binary_right_scalar` (`:480`) / `_left_scalar` (`:467`) → `buffer::map1` → `zip1_slice` (iterator form).
- anything else → `broadcast_binary` (`:500`) → `buffer::map2_runs`.
- `Layout::Scalar` means an operand of RANK 0 with one element (`binary.rs:436`); a size-1 operand of shape `(1,)`/`(1,1)` is classified `Broadcast`, unlike `buffer::broadcast::size_one_value` which accepts any product-1 shape.

Comparison: `binary` → `compare` (`compare.rs:38`) → promote/cast → `compare_typed` (`:110`) → `layout` × 2 → `dense_compare` (`:156`), which uses `bool_results` below `BOOL_RESULT_THREAD_MIN` and `map2` at/above it, `dense_compare_{left,right}_scalar` (`:174`/`:186`) for a rank-0 operand, and `broadcast_compare` (`:205`) → `map2_runs(DType::Bool, ..)` otherwise.
`logical_and`/`logical_or` are `BinKind::Mul`/`Add` on bool (`binary.rs:60-61`) and take a dedicated fast path (`:292-307`) to `bool_results` when both operands are flat bools and the result is below the thread crossover, so a bool-logical op allocates like a comparison, not like arithmetic.

`select`: `select_kernel` → dtype/predicate casts → `broadcast_shape` → `select_numeric` storage dispatch → `select_apply`, which takes `map3` only when predicate AND both branches are `is_flat_for(out_shape)` (`compare.rs:311`); EVERY other combination (a 0-d or `(1,..)` branch, a broadcast predicate) goes to `map3_runs` — there is no size-1 capture arm before the broadcast builders.

Unary: `unary` → `un_kind` → optional cast → `with_type!` → `typed_unary` → `dense_unary` (`unary.rs:157`), one `map1` arm per kind with the kind a literal, so the closure folds to one machine op.
`logical_not` leaves through `typed_unary`'s bool arm (`:145`) and is the one unary op that pays `to_scalar().truthy()` per element.

## Per-call cost inventory (fixed cost dominates small arrays)

Every op pays, per call: the name→kind string match; `broadcast_shape` (one `Vec`); `layout`/`is_flat_for` probes (product over the shape per operand); the `with_same_type!` 14-arm dispatch; `take_pooled_vec` (TLS + `RefCell` + a linear scan of up to 64 pool entries); `out_shape.to_vec()` (one `Vec`); `Buffer::new` validation; `single` (one `Vec<Value>` + one `Rc`); and `Drop for Storage` → the pool's `push` scan.
A broadcast op additionally pays the classification chain in `buffer::map2_runs`/`map3_runs`: `broadcast::runs2` first (the Repeat driver — for a `(…,1)` chunky operand it always declines, after allocating one `broadcast_strides` `Vec` per operand), then `periodic2`/`periodic3`, which allocates another stride `Vec` per operand and re-classifies from scratch; 4 stride allocations for a binary op and 6 for `select`.
On a pool miss `map2_runs`/`map3_runs` also zero-fill the result (`R::zeros(count)`) before overwriting all of it; `map1`/`map2`/`map3`/`bool_results` instead build from an exact-size iterator on a miss (no memset).
`transcendental.rs` (`erf`/`gelu`/`nan_to_num`) uses plain `iter().map().collect()` — an exact-size allocation, never the storage pool.

## Per-element characteristics

`apply_bin` (`binary.rs:364`), `compare_op` (`compare.rs:135`) and `apply_un` (`unary.rs:106`) each contain a `match kind`, but every driver arm passes a literal `BinKind`/`UnKind`, so the match folds away once inlined; none of the three is annotated `#[inline]`, so the fold depends on LLVM inlining them (a non-folded dispatch measured `ew_chain_1e6` 1.99 → 1.50 ms, i.e. ~0.5 ns/element).
No hot loop has a per-element bounds check (every driver pre-slices or iterator-zips) and none pushes/extends per element.
`exp` on f32 is the scalar libm call (`scalar.rs:440` → `f32::exp` → `expf`), one call per element, with no vectorized or polynomial arm anywhere: the polynomial arm was measured 1.64 ns/element against glibc `expf` at ~1.0 and removed; LLVM does not vectorize `expf` calls (no libmvec declaration), so a 20k-element `exp` is 20k scalar calls.
`Num::max`/`min` for the float dtypes are hand-written NaN-propagating branches, not `f32::max`, so `maximum`/`minimum` do not vectorize.
`convert_generic` (`buffer/storage.rs:239`) round-trips every element through `ScalarValue` and is what a dtype mismatch costs; a same-dtype `cast` short-circuits in `cast_kernel` (`mod.rs:147`).

## Constraints

Never a second layout engine: addressing is `layout`/`is_flat_for` probes plus the `buffer` drivers; a new arm must reuse `map*`/`map*_runs` rather than walk strides.
Dtype conversion is never silent at the semantic level: the cast the node performs is the one the declared result type demands.
Errors are the numpy backend's, verbatim (`validate_binary`/`validate_unary`/`compare`'s complex-ordering rejection/`select`'s incompatible-branch message).
`is_ordering`, `BinKind`, `Layout` and `layout` are re-used by `compare.rs` and `crate::kernels::pow`; keep them `pub(super)`-visible from `binary.rs` rather than duplicating the probes.

## Known issues / cost notes

- A `(…,1)` broadcast operand pays two full classification passes and 4-6 `Vec<usize>` allocations per call before the periodic driver writes it, because `broadcast::runs2` (Repeat) can never accept a stride-0 innermost axis.
- `select` has no size-1 capture arm: `select(mask, x, scalar)` and any select with a rank-0/`(1,..)` predicate goes to `map3_runs` (classification + possible `R::zeros`) where `apply3_static` would have captured the constant.
- `layout`'s `Scalar` arm requires rank 0, so a one-element operand of shape `(1,)` takes the whole broadcast path instead of `map1`.
- `sigmoid` is registered as a fused unary op (`unary.rs:126`) but graphs written as `1/(1+exp(-x))` pay four ops (negate, exp, add, divide) with four allocation/classification rounds.
- A mixed-dtype comparison promotes and casts BOTH operands (f32 vs int constant → f64): two full `convert_generic` passes over the array.
