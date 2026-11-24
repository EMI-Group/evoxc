# tests/backends — backend-layer test suite

## Intent
pytest suites validating `evoxc.backends` (sibling `../../evoxc/backends` — read its CONTEXT.md and `../../evoxc/CONTEXT.md` before changing expectations): the backend interface/registry/program contracts, the numpy interpreter (the numerical reference), the StableHLO exporter, the native CPU engine backend, and the iree / xla / tvm adapters.
Tests are the executable spec of the backend contract: a graph run by the numpy backend must match a direct numpy reference computed with the SAME formulas as `evoxc/backends/numpy/kernels/`, and every other backend must match that numpy result.
This file owns the test-side contracts, the skip/host-guard policy and the recorded counts; source behaviour, per-option tables and adapter Known Issues belong to `../../evoxc/backends/CONTEXT.md` and `../../evoxc/backends/adapters/CONTEXT.md`.

## Binding contracts pinned here

### Numerical reference and interpreter seams
- `test_numpy_interpreter.py` is the numerical oracle for every other backend: per-op output, dtype and error kind against hand-written numpy references, never against evoxc kernels.
- The interpreter's execution-plan contract is pinned through its seams: one plan per module block (entry + cond + body for a `while_loop`), each value released at its last use, the static-result `(dtype, shape)` fast path recorded on the plan, and the unchanged `ShapeError`/`BackendError` wording on both the static and the symbolic path.
- `test_numpy_single_decode.py` pins SINGLE-DECODE staging through `decode_stats()`/`reset_decode_stats()`: in-process `trace → lower → compile → load`, `build` and `evaluate` decode zero times, each disk load decodes exactly once, save→load→save container bytes are identical, and a saved artifact runs in a fresh subprocess at its own decode count.
- `test_numpy_random_scalar_shapes.py` pins the rank-0 (`shape=()`) sampling contract: `random_uniform` / `random_normal` (both branches) / `random_randint` return a genuine 0-d `core.Tensor` on `Device('cpu', 0)`, scalar equals `(1,)`[0] from the same key, deterministic, bit-exact vs the native backend (the f32 `normal` ⊕ `philox4x32_10` leg takes the documented `rtol=1e-6` budget), and `permutation(0|1)` yields real `(0,)` / `[0]` int32 arrays.
- `test_backend_interface.py` + `test_program.py` pin the interface/registry/program surface: `Capabilities` fields and numpy flags, `Backend` ABC abstractness, registry semantics, `Signature` fields (including `output_static_values`), structured inputs, and `LoweredProgram`/`CompiledArtifact` save-load.
- Plan-text/stats seams are asserted as NAMED keys and plan markers, never as prose: `test_native_backend.py` checks `summary()`/`plan_text()`/`stats()`, and `test_native_backend_perf_shape.py` checks the flat-block `static float32[…]` markers.

### Native-vs-numpy parity
- `test_native_backend_parity_{core,structure,random}.py` compare the native engine against the numpy oracle (`_native_utils.evaluate_pair`, or `build_native` + `run_native` when one executable serves several input sets): exact for bool/integer, `nu.dtype_tolerance` per float width, and `RANDOM_NORMAL_TOL` for f32 `random_normal`.
- Parity expectations are never a bare hard-coded value: an independent `np.*` or hand-written reference backs every row where the formula is unambiguous.
- The extension is OPTIONAL: every native file except `test_native_backend_missing_engine.py` carries `pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)`, and `nu.without_engine()` (bogus `EVOXC_NATIVE_LIB` + loader-cache reset and restore) is the ONLY supported way to exercise the missing-extension path.
- `test_native_backend_missing_engine.py` runs in BOTH configurations and pins the actionable `core.BackendError` (naming `scripts/build-native.sh`, `EVOXC_NATIVE_LIB`, `native_compile`) with no silent numpy fallback and no `SystemExit`/`ImportError`.
- `test_native_backend_features.py` pins the feature integrations (control flow, pytrees, symbolic dims, transforms, `external_call`, `block_call`, collectives, the sparse deferral) and `test_native_backend_perf_shape.py` pins fast-path SHAPE under generous ceilings, so a performance assertion can only fire on a regime change.
- `test_native_numpy_interop.py` pins the bidirectional HOST-DATA boundary through the PUBLIC API only (`evoxc.from_numpy` → `evoxc.build`/`evoxc.run`/`evoxc.bind`, `backend="native"`): every declared dtype round-trips against both an explicit `np.*` reference and the numpy interpreter; native outputs are fresh, writable, C-contiguous arrays that never alias the input; Fortran-order, transposed-view and negative-stride inputs equal their `np.ascontiguousarray` copies and never mutate the caller's buffer; read-only inputs run and stay read-only; 0-d, 1-D-empty and rank-2 empty (`(0, 3)`) arrays round-trip; consecutive runs are identical; and a two-call chain (`from_numpy` / the output `core.Tensor` / `bind`) equals one numpy pass of the equivalent graph.
- Native outputs cross the FFI as TYPED numpy arrays: the binding downcasts inputs to typed views (never a byte view) and HANDS each engine `Vec<T>` to numpy, so numpy records the transferred Rust buffer as the array's base object and `array.flags.owndata` is legitimately `False` while the memory is exclusively owned by that array — 13 of the 14 dtypes take that zero-copy path, and `bool` is copied because a `Vec<bool>` is bit-packed.
- The interop suite therefore asserts freshness as ndarray-ness + C-contiguity + writeability + `np.shares_memory(...) is False` against the caller's array, never `owndata`.

### Exporter golden ownership
- `test_stablehlo.py` owns the general export goldens: mnemonics and the module shell, dtype text, comparison attrs, program order, determinism, symbolic dims → `?`, decompositions, the deferred-op list and the dynamic-dim rejections, dot_general batch ranks, mixed-dtype equalization, the SET-form `rng_bit_generator` goldens, the `eigh` compositions (line counts, early-exit, the invalid-option message), the byte-identity of the emission fast paths, and the single verification per export.
- `test_split_reduction.py` owns the LARGE-shape reduce text: `count('"stablehlo.reduce"') == 2` plus the one `// evoxc split reduction:` marker on split cases over `(10001,)`, `(10000,100)`, `(10007,)`, `(1024,)`, `(1023,)`, the `edge_padding_high` geometry and the stage-reshape comment.
- `test_stablehlo.py` pins reduce text only on TINY shapes, and its only exact reduce COUNT is the `eigh_early_exit` sweep check.
- Presence-only reduce asserts live in `test_count_argsort.py` and `test_while_init_rewrite.py`; reduce INIT constants, reducer-region mnemonics and argmax/argmin comparator text are deliberately NOT pinned anywhere in this node.
- `test_stablehlo_remainder.py` OWNS the `remainder` text (ordered mnemonic sequence and counts, the unsigned no-correction shortcut, the float zero constant, the bool/complex dtype deferrals).
- `test_iree_emitters_parity.py` section 8 pins only the `remainder` NUMBERS (no MLIR text in that file), and the `stablehlo.remainder` sites inside the `eigh` compositions stay direct fmod, so split-reduction and stablehlo goldens never pin `remainder` text.
- Mnemonic quoting is part of the pin: functional ops such as `stablehlo.remainder` are counted UNQUOTED, generic ops (`"stablehlo.while"`, `"stablehlo.reduce"`, `"stablehlo.transpose"`, `"stablehlo.if"`, `"stablehlo.slice"`) in the quoted custom-op form.
- Option VALUES are validated by the Writer (loud `BackendError` naming the accepted set), option KEYS by `options.validate_options` — the goldens assert both sides.

### Option plumbing and adapter-fake isolation
- `test_options_override.py` is the backend half of `evoxc/backends/options.py` (env half = `tests/test_pipeline_options.py`): the `validate_options` union, per-adapter `KNOWN_OPTIONS`, the iree compile-flag policy, per-adapter overrides, `normalize_bool_option`, and the rocm default compile args.
- `test_opt_level.py`, `test_vulkan_target.py` and `test_dispatch_formation.py` are the device-LESS option suites: derivation matrix, value validation, `KNOWN_OPTIONS` stage placement, env rows and target/label/load-gate consequences, driven by a COMPLETE fake `iree` package.
- Compiler-free rule (binding): a helper-level adapter test must PASS with the compiler absent, so a fake must be installed as a complete top-level PACKAGE module (`types.ModuleType("<dep>")` with `__path__ = []`) plus its submodules, via `monkeypatch.setitem`/`setattr` — never `pytest.importorskip` on such a test, and never clobber an already-imported real package (patch its attribute instead).
- Only a leg that needs a REAL compiler may skip, and only on the genuine absence signature (its own `importorskip`, `_rocm_guard`, or the adapter's `pip install evoxc[` / no-plugin `core.BackendError`); any other error must FAIL.
- Reference implementation for the fake-package convention: `test_options_override.py::_install_fake_iree_runtime`.

### Explicit device placement, guards and skips
- A run input must already live on the executable's device: place it with `Tensor.to(device)` before `evoxc.run`, since host ndarrays and cpu:0 tensors are rejected at the run boundary with `core.DeviceError` naming the input pytree path.
- Outputs are materialized through the explicit `out.to(Device('cpu', 0)).numpy()` hop, because `.numpy()` on a non-cpu-kind payload raises `core.DeviceError`.
- llvm-cpu executables keep the UNCHANGED host-input path (raw ndarray / cpu:0 `core.Tensor` in via `asdevicearray`, lazy per-call `.numpy()`), pinned by `test_iree_llvm_cpu_host_inputs.py`.
- CUDA legs guard on `nvidia-smi` (most-free scan) or a hardcoded `Device("cuda", 5)` smoke convention; ROCm legs guard through `_rocm_guard.require_rocm()` and NEVER use `nvidia-smi`.
- The ROCm index mapping is `iree hip device_id - 1`, and `"hip"` is an accepted alias whose `Device` label is preserved.
- GPU legs are correctness-only (no timing — the box may be shared), with ONE compile per `(fn, specs, device, target_backends)` reused across value cases.
- Vulkan device-guarded tests use an `importorskip` of iree plus a CHILD-PROCESS probe of `get_driver("vulkan").query_available_devices()` under a timeout, because a wedged vendor ICD makes the loader HANG instead of erroring.
- `test_iree_vulkan_device_resident.py` therefore runs only where the HAL enumerates a device, and every host readback there goes through the adapter's explicit staged copy.
- A skip reason names the guard that fired; a missing device, dependency or plugin never turns into a failure, and a code bug is never turned into a skip.

### External kernels and host IO
- `test_external_call_iree.py` pins the iree split contract: a `stablehlo-segments` payload plus per-segment vmfbs, HOST-mode dispatch bit-exact vs the pure-graph numpy run, per-backend slot resolution, the once-per-executable staging `UserWarning`, the explicit errors (unregistered kernel, call inside a control-flow body, non-static HOST result dims), and DEVICE-RESIDENT operands/results with METADATA-ONLY validation.
- `test_external_batched_iree.py` pins the batched-variant sugar through `vmap`/`vectorize`: one invocation with the FULL batched stack, zero staging warnings, zero host round-trips on the vmapped boundary, and the explicit HOST-mode static-dims deferral.
- `test_external_portable_fallback.py` pins the portable-decomposition fallback on a backend without host dispatch: the portable body inlined into ordinary StableHLO, one user-visible warning per kernel name, and the loud `BackendError` when no portable exists.
- `test_iree_device_dlpack.py` pins device-payload DLPack export: verbatim delegation of `__dlpack__`/`__dlpack_device__`, non-CPU payloads only, the `torch.from_dlpack` zero-copy round trip on the GPU legs, and the explicit `DeviceError`/`BackendError` refusals.
- `test_iree_host_io_parity.py` pins `evoxc.io` parity: `prefetch` equals an explicit `Tensor.to` upload, `sink` round-trips to an ndarray-backed cpu:0 host tensor, schedule-time errors surface verbatim, and runtime errors re-raise at every `wait()`/`result()` instead of being swallowed.

### xla plugin and PJRT driver contracts
- `test_xla_plugin_discovery.py` pins the four-step resolution order (`plugin_path` > `EVOXC_PJRT_PLUGIN` > fixed well-known paths > the `jax_plugins/*/*.so` scan), candidate and `sys.path` realpath dedupe, per-realpath probe caching, the `EVOXC_PJRT_PLUGIN_DISCOVERY` truth table and opt-out, no `jax`/`jaxlib` import, and the reported-platform→device-kind check.
- Ambient-box isolation is binding for every xla test: a test that needs "no plugin anywhere" deletes both env vars, empties `xla_util._DEFAULT_PLUGIN_PATHS`, opts out of (or resets) the scan cache, and never assumes a bare box; resolution tests REPLACE `sys.path` because candidates sort by path string.
- `test_xla_autodiscovery.py` pins the pipeline consequences: `xla.register()` is unconditional while `check_available()`/`build(..., backend="xla")` raise the same four-step `BackendError`, `plugin_path` alone builds and runs, and `build`/`build_cached` forward the SAME resolved options to `lower`, `compile` and `load`.
- `test_pjrt_ctypes_plugin.py` pins the ctypes driver contract through the gcc-built fake plugin: the ABI/version gate, shared-client refcounting, lower/compile/load/run and artifact round-trips, multi-dim/multi-output graphs, and env-injected failures surfacing the plugin message.
- `test_xla_device_resident.py` pins placement semantics (device-resident run-outs, same-device pass-through with zero `BufferFromHostBuffer`, host/foreign inputs illegal at the boundary, load-device validation, payload/client lifecycle, the per-backend transfer-provider slot) and the client-lock discipline via a subprocess pin.
- `test_xla_synchronize.py` pins `XlaExecutable.synchronize`: completion handles accumulate across runs, one call awaits then destroys each pending handle in order, a second call performs zero FFI, await/destroy failures re-raise the FIRST error and stay retryable, a missing handle is an explicit `BackendError`, and the CPU staging path requests no event.

### Memoized build
- `test_build_cache_iree.py` pins `build_cached`/`CompileCache` on real iree compiles: is-identity hits with `misses==1`/`hits==1`, `opt_level` participating in the key, semantic device-spelling equality (default cpu:0 == explicit `Device("cpu", 0)` == `"cpu"`, while the bogus `"cpu:0"` kind is rejected at load and never cached), and a cached same-device inner loop.

## Routing table

| File | What it pins | Items |
|---|---|---|
| `test_backend_interface.py` | `Capabilities`/`Backend`/`Executable` protocol, numpy capability flags, registry semantics, program save/load smoke | 63 |
| `test_program.py` | numpy `LoweredProgram.text()`/`.payload`, `Signature` fields, structured inputs, round-trips | 13 |
| `test_numpy_interpreter.py` | the numerical correctness suite plus the execution-plan/liveness/static-fast-path/error-wording seams | 315 |
| `test_symbolic_dims.py` | one-build/multi-size symbolic dims, `DimExpr` reshapes, `None` dims | 8 |
| `test_control_flow.py` | `cond`/`while_loop`/`scan` semantics and their nesting under symbolic dims | 11 |
| `test_runtime_call.py` | `runtime_call` sync execution (called once, 0-d scalar operands) and count/dtype/shape errors | 10 |
| `test_collectives.py` | single- and multi-rank collectives, `rank`/`world_size`, executor hook and rank-context save/restore | 22 |
| `test_stablehlo.py` | the general exporter goldens, dynamic-dim rejections, batch-rank dots, RNG set form, `eigh` compositions, fast-path byte identity, verify-once | 163 |
| `test_stablehlo_remainder.py` | the `remainder` floormod text golden, the unsigned shortcut and the bool/complex deferrals | 9 |
| `test_stablehlo_custom_ops.py` | the `custom_call` / `inline_stablehlo` exporter goldens: result arities, target quoting/escaping, `backend_config`/`api_version`/`has_side_effect`, `%in`/`%out` substitution, multi-line/nested-body preservation, writer token validation, program order, `status()` v1 vs deferred | 19 |
| `test_split_reduction.py` | the two-stage split-reduction emission goldens (exactness classes, gates, geometry, `split_reduction=False` byte identity) | 67 |
| `test_iree_eigh_diag_parity.py` | llvm-cpu `eigh`/`diag` parity on scale-normalized invariants, plus `eigh` inside `cond`/`while` regions | 24 |
| `test_mixed_dtype.py` | mixed-dtype compile+run parity on every available adapter via `_xla_plugin_probe`/`importorskip` gates | 18 |
| `test_options_override.py` | the backend half of the options contract: validation, `KNOWN_OPTIONS`, flag policy, overrides, rocm defaults, the split-reduction sub-option plumbing | 97 |
| `test_opt_level.py` | `normalize_opt_level`, the derived iree flag, the xla wire-format editor, env precedence, end-to-end smokes | 53 |
| `test_vulkan_target.py` | device-less `vulkan_target` derivation, validation, `KNOWN_OPTIONS`, target labels and the load gate on both load paths | 73 |
| `test_dispatch_formation.py` | `dispatch_formation` derivation and validation, and result parity of a merged population with the option on/off | 28 |
| `test_while_init_rewrite.py` | the while-init rewrite (MLIR position, repeated compiles, bit-exact run) and its opt-out | 6 |
| `test_count_argsort.py` | count-mode argsort emission, the `sort_emission` selection rule and stable-reference parity | 9 |
| `test_nsga2_xla_route.py` | the xla route for large-axis argsort (no sort bufferization bug) | 2 |
| `test_deferred_errors.py` | numpy lower-time rejections, compile backend mismatch, load device gate, stablehlo not a backend | 8 |
| `test_artifact_persistence.py` | program/artifact/executable round-trips and their corruption, mismatch and tamper errors | 11 |
| `test_compiler_framework.py` | the shared `CompilerBackend` contract through a stub, and the verification-pass COUNT (one walk, two after a portable splice) | 17 |
| `test_adapters_optional.py` | lazy adapter import, the missing-dependency message path, and the default backend | 7 |
| `test_adapter_iree.py` | the iree adapter contract plus the HIP/CUDA 16-binding cap rewrite (`TestIreeBindingLimitError`) | 27 |
| `test_adapter_xla.py` | the same contract against the auto-discovered PJRT plugin (module-level skip without one) | 10 |
| `test_adapter_tvm.py` | the same contract against TVM (module-level skip without tvm/jaxlib) | 13 |
| `test_xla_plugin_discovery.py` | the four-step PJRT plugin resolution order, candidate hygiene, probe caching, the discovery opt-out and the platform→kind check | 45 |
| `test_xla_autodiscovery.py` | the pipeline consequences: unconditional registration, the actionable no-plugin error, `plugin_path` forwarding to every stage | 7 |
| `test_pjrt_ctypes_plugin.py` | the PJRT ctypes driver contract through the gcc-built fake plugin (ABI gate, client refcounts, plumbing) | 10 |
| `test_xla_device_resident.py` | xla explicit-placement semantics, payload/client lifecycle and the client-lock subprocess pin | 10 |
| `test_xla_synchronize.py` | the explicit `synchronize()` surface against the fake plugin's completion events | 10 |
| `test_iree_device_resident.py` | iree residency semantics on llvm-cpu and cuda: payload-backed outputs, `.numpy()` kinds, boundary enforcement, zero `asdevicearray` | 8 |
| `test_iree_same_device_loop.py` | the cuda same-device loop pattern: place once, feed outputs back, zero host round-trips, relative perf pins | 4 |
| `test_iree_rocm_device_resident.py` | the ROCm residency and boundary contract, the `"hip"` alias, and the guard's own detection | 7 |
| `test_iree_rocm_while_scan.py` | ROCm `while_loop`/`scan` shapes device-resident and bit-exact, with the unchanged placement boundary | 6 |
| `test_iree_vulkan_acquisition.py` | device-less vulkan HAL acquisition (queried-hash index, per-index cache, error wrapping), the staging tier and provider registration | 22 |
| `test_iree_vulkan_device_resident.py` | the vulkan explicit-placement contract: parity, residency, readback, host-input rejection, warm loop, out-of-range index | 6 |
| `test_iree_emitters_parity.py` | llvm-cpu parity for the v1 emitters (gather/sort/argsort/tile/scatter/scan/shifts/random) and the `remainder` numbers | 59 |
| `test_iree_split_reduction_parity.py` | numeric parity and reassociation budgets for the split reduction, with the dtype/NaN accumulator pins | 127 |
| `test_iree_llvm_cpu_host_inputs.py` | the unchanged llvm-cpu host-input path and per-call lazy `.numpy()` | 6 |
| `test_iree_random_algorithms_parity.py` | algorithm-aware RNG parity (native threefry vs inline expansions) and the per-call `rng_bit_generator` override | 29 |
| `test_iree_hotpath_validation.py` | payload metadata caching and the static-spec validation fast path, observably equal to the walker on accept and reject | 10 |
| `test_iree_normal_f32_fastpath_parity.py` | f32 `random_normal` fast-path parity across all three algorithms on the adapter default lowering | 12 |
| `test_external_call_iree.py` | iree host- and device-resident `external_call` dispatch, split artifact, the error set and metadata-only validation | 39 |
| `test_iree_device_dlpack.py` | device-payload DLPack delegation, the torch zero-copy round trip, and the placement refusals | 12 |
| `test_external_batched_iree.py` | batched-variant / batch-invariant sugar through `vmap` on a device boundary | 5 |
| `test_external_portable_fallback.py` | portable-decomposition inlining with one warning per kernel, and the no-portable deferral (tvm-gated) | 9 |
| `test_iree_host_io_parity.py` | `evoxc.io` prefetch/sink parity, verbatim schedule errors and runtime failures surfacing at wait/result | 11 |
| `test_build_cache_iree.py` | `build_cached`/`CompileCache` keys and is-identity hits on real iree compiles | 4 |
| `test_custom_ops_iree.py` | `inline_stablehlo` compiles/runs on iree with numpy parity (host-resident); `custom_call` reaches IREE and fails the legalization diagnostic (naming the op/target, never a parse error) | 3 |
| `test_custom_ops_backend_rejection.py` | numpy + native `lower()` reject `custom_call` / `inline_stablehlo` with a naming `BackendError` (native leg engine-guarded) | 6 |
| `test_block_portable_compiler.py` | portable block inlining on a compiler backend (numpy, a `CompilerBackend` stub, iree) and the explicit rejection of a backend-only block | 7 |
| `test_registry_extensibility.py` | `register(replace=True)`, `unregister`, and entry-point discovery (module/`register()`, callable, `Backend` instance), single-scan-per-process, failing entry point → `BackendError` with `__cause__` | 11 |
| `test_iree_module_option.py` | `iree_compiler_module` / `iree_runtime_module`: option-surface validation, a delegating fake routing the compile/load hop, bogus dotted path → `BackendError` naming option+value | 7 |
| `test_native_backend.py` | native staging/errors/diagnostics/persistence, `summary()`/`plan_text()`/`stats()` keys | 32 |
| `test_native_backend_missing_engine.py` | the missing-extension contract in both configurations | 10 |
| `test_native_backend_parity_core.py` | native-vs-numpy parity for elementwise/comparison/reduction ops across every dtype | 155 |
| `test_native_backend_parity_structure.py` | native-vs-numpy parity for structure/indexing/sorting/linalg | 122 |
| `test_native_backend_parity_random.py` | native-vs-numpy RNG parity for all three algorithms and the subkey trees | 70 |
| `test_native_backend_features.py` | native feature integrations: control flow, pytrees, symbolic dims, transforms, external/block calls, collectives | 33 |
| `test_native_numpy_interop.py` | the bidirectional NumPy host-data contract through the public API: every dtype round-trips, layouts are never misread, read-only/0-d/empty inputs, no output aliasing, chained runs | 41 |
| `test_numpy_single_decode.py` | the single-decode staging property through `decode_stats()` | 15 |
| `test_numpy_random_scalar_shapes.py` | the rank-0 sampling contract across algorithms/dtypes, plus the empty and single-element permutations | 83 |
| `test_native_backend_perf_shape.py` | native fast-path SHAPE pins (strided slice, broadcast add, typed random, bench workloads) | 5 |
| `_native_utils.py` | shared native helper: engine gate, tolerances, `evaluate_pair`, `without_engine` — NOT collected | n/a |
| `_adapter_utils.py` | shared adapter graphs, fixed-seed inputs, staging/parity helpers — NOT collected | n/a |
| `_rocm_guard.py` | the ROCm/HIP device guard and index mapping — NOT collected | n/a |
| `_xla_plugin_probe.py` | the shared xla run-device probe, placement/readback/parity helpers — NOT collected | n/a |
| `_fake_pjrt_plugin.c` | the test-only fake PJRT C API plugin source, built with gcc at test time, with its ABI and failure knobs — NOT collected | n/a |
| `__init__.py` | package marker, so modules import as `tests.backends.<module>` | n/a |

## Conventions (binding for this directory)
- Run from repo root: `python3 -m pytest -q tests/backends` (the root `conftest.py` handles sys.path).
- Pure-lower/golden files are numpy-only, CPU-only, small shapes, under 2 s; the adapter/parity/plugin files run real compilers or a gcc build, so the 2 s rule does not apply there.
- Numerical references re-implement the kernel formulas (never import evoxc kernels): `math.erf` for erf, the erf-based gelu formula, manual dilate→pad→cross-correlate loops for conv.
- Tolerances: exact for int/bool, fp32 rtol=atol=1e-6, fp64 rtol=atol=1e-12 in the numpy-reference suites, and 1e-5 for CROSS-COMPILER fp32 parity.
- Execute graphs through `evoxc.evaluate(...)`, `evoxc.build` + `evoxc.run`, or the explicit pipeline; results are `evoxc.Tensor`, unwrapped with `.numpy()`.
- Documented evoxc quirks encoded as tests (do NOT "fix" evoxc here): `evoxc.slice` lengths are spans, `solve` needs an `(..., n, k)` rhs for batched `a`, getitem supports only static int/slice, `dot` requires rank ≥ 2, and `evoxc.scan` desugars to while+gather+scatter at trace time.
- Shared state hygiene: the collective-executor hook and rank context are process/thread global, so tests touching them save and restore through fixtures.
- Stage asymmetry to keep in mind in the explicit pipeline: `target_backends` is a COMPILE-stage option (accepted and ignored at lower/load), so a ROCm build spells `lower(..., target_backends=["rocm"])` → `compile(..., target_backends=["rocm"])` → `load(..., device=rocm)`.
- `BUG(evoxc)` policy: a test that exposes a real evoxc bug keeps its assertions intact and is reported upward with a minimal repro; never fix evoxc and never weaken the test.
- The green sibling suites get `@pytest.mark.xfail(strict=True, reason="BUG(evoxc): …")` with a `# BUG(evoxc):` comment, so the pin XPASSes — failing loudly — the day evoxc is fixed.
- Current pins in this node: none.

## Recorded counts
Measured at this HEAD with `iree-base` 3.13.0 installed, the native extension BUILT, no NVIDIA GPU, no usable ROCm/HIP device, no PJRT plugin, no tvm and no torch, and one enumerated Vulkan device (so the device-guarded Vulkan suite RUNS).
`pytest tests/backends -o addopts='' -q` → **2060 passed, 73 skipped, 0 xfailed, 0 failed**; `--collect-only -q` → 2130 collected plus the 3 module-level collection skips (`test_adapter_tvm.py` 13, `test_adapter_xla.py` 10, `test_external_portable_fallback.py` 9 items when their dependencies are present) = 2133 reported items.
The 73 skips are host/dependency guards only: 22 CUDA (`nvidia-smi` absent, incl. the `Device("cuda", 5)` smokes), 22 ROCm (the `hip` HAL driver reports no devices), 12 tvm-absent, 8 PJRT-plugin-absent xla items, 5 tvm-adapter-unavailable, 1 ROCm-runtime probe and the 3 module-level collection skips.
The node has no xfails.
Item counts per file are in the routing table; the three dependency-gated modules report 1 item each in this environment.

## Known Issues
- v1 collective-protocol limitations (asserted as-is): `reduce_scatter`'s `reduce_op` is not forwarded to the executor, `all_to_all` forwards only `split_axis`, and `dist.broadcast` records the default `src_rank=0` in IR even when the user passes another value.
- `split_reduction_threshold` / `split_reduction_non_exact` are accepted lower options on every compiler adapter (iree/xla/tvm): `CompilerBackend._exporter_options` forwards them (bool coerced / int passed through) and the Writer validates their values. The numpy backend documents-ignores any lower option (it has no `KNOWN_OPTIONS`), the native backend rejects them, and neither has an env row.
- Default split policy leaves FLOAT rank-0 reductions single-stage (reassociation): a full f32 `sum` of 16M elements on iree/vulkan measured ~648 ms while the same graph with `split_reduction_non_exact=True` ran in ~11 ms, so the opt-in is the workaround until the default policy lands.
- The `eigh_algorithm` / `split_reduction` env rows live outside this node: three `(backend, "lower")` rows in `evoxc.pipeline_options.ENV_OPTION_TABLE`, whose env-half tests are in the sibling `tests/test_pipeline_options.py`, and the two sub-knobs above have NO env row.
- `test_iree_vulkan_device_resident.py` needs a Vulkan device IREE enumerates; where nothing is enumerated the suite skips, and every result it produces on this host is software-device correctness, never a GPU performance number.
