# evoxc_hf — HuggingFace checkpoints as one evoxc graph

## Intent

Load a HuggingFace checkpoint (`config.json` **and** its weights from disk) and run the **whole model forward pass as ONE evoxc graph** — trace → lower → compile → load → run — with evoxc's explicit semantics applied to a whole model: no hidden tracing, no hidden weight embedding, no hidden device movement.
`evoxc_hf` is a **separate, independently publishable distribution** — `evoxc-hf` on PyPI, whose import package stays `evoxc_hf` — and never part of the `evoxc` distribution: evoxc core's non-goals forbid NN modules / loaders / model zoos, so model-level code lives here in a **src layout** (`./src/evoxc_hf/`) and only *depends on* the public evoxc API.
The API is **functional** and split along evoxc's own value model into three layers that never blur:

| Layer | Module | Question it answers | Surface |
|---|---|---|---|
| **loading** | `load.py` | *what is this checkpoint?* | `load_config`, `load_params`, `load_hf` → `LoadedModel` (plain immutable data), `validate_params` |
| **the compute function** | `program.py` | *what does it compute?* | `forward_fn` (the memoised `@evoxc.defn` forward, capturing **no** parameters), `param_specs`, `reference_logits`, `resolve_seq_len`, `DEFAULT_SEQ_LEN` |
| **staging** | `pipeline.py` | *where and how does it run?* | `build`, `run`, `predict`, `place_params`, `place_ids`, `resolve_build_options`, the `DEFAULT_*` option constants |

Nothing here is a class with behaviour: `LoadedModel` is a `typing.NamedTuple` of field access only, and every operation is a module-level function taking its inputs as arguments.
Two model families are implemented — **`llama`** (HF `llama`/`mistral`/`qwen2`/`qwen3`, the Qwen3 case adding per-head QK-Norm via the static `qk_norm` config field) and **`gpt2`** (HF `gpt2`, e.g. `gpt2`/`distilgpt2`) — each as one `@evoxc.defn` forward whose weights are ordinary graph inputs.
Parity is pinned against an **independent pure-numpy oracle** (`reference.py`, no `evoxc` import) and, when `torch`+`transformers` are installed, against the real HuggingFace implementation (`tests/test_torch_oracle.py`).
Whole-graph StableHLO export is pinned without any compiler (`tests/test_export.py`); the compiled `iree` paths are pinned when `iree` is installed (`tests/test_iree_llvm_cpu.py`, `tests/test_iree_rocm.py`), and the compiled `xla` path is validated by the benchmark harness (`../bench_logs/hf_bench.py` → `../bench_logs/hf_bench_results.md`), not by a gated test.

## Package boundary & distribution status

- **Two names, one package**: the PyPI distribution is `evoxc-hf` (PyPI normalises it to `evoxc_hf`) while the **import name is `evoxc_hf`** everywhere in code, docs and tests — `evoxc-hf` is never a Python path.
- **Independent, self-contained distribution**: `evoxc-hf` publishes on its own (`uv build --project evoxc_hf`) and is never bundled into the `evoxc` wheel or sdist (pinned by the root `../tests/test_packaging.py`); it depends on `evoxc` like any third-party package, and `evoxc` never imports it.
- **src layout**: `./src/evoxc_hf/` holds all 13 modules plus `models/`, while `./tests/` and this `./CONTEXT.md` stay at the top level of this directory — `uv_build` rejects `module-root = ".."`, so a subproject whose repository root is the `evoxc` project cannot use a flat layout.
- **Build backend**: `./pyproject.toml` is a `uv_build` project — `module-root = "src"`, `module-name = "evoxc_hf"`, static `version = "0.1.0"`, `requires-python = ">=3.11"`, PEP 639 `license = "Apache-2.0"` with `license-files = ["LICENSE"]`.
- **Hard dependencies: `evoxc>=0.1.0` + `numpy>=1.24`, nothing else.** The readers are `numpy` + `struct` + `json` + `zipfile`/`pickle`, and `torch`/`transformers`/`safetensors`/`iree` are **never** imported at module scope (pinned in a subprocess by `tests/test_public_api.py`).
- **The extras live here**: `hf = ["torch>=2.0", "transformers>=4.40"]` and `dev = ["pytest>=7"]`, where `hf` is **reference oracles only, never runtime requirements** — the real HuggingFace implementation this one is validated and benchmarked against, and `import evoxc_hf` works without either.
- **The compiled path is an evoxc extra, not this package's**: `pip install "evoxc[iree]"` (or `evoxc[tvm]`, or a user-provided PJRT plugin `.so` for `xla`) enables it; `pip install evoxc-hf` alone gives the numpy interpreter.
- **Workspace member**: the root `../pyproject.toml` — the `evoxc` project, owned outside this node — declares `[tool.uv.workspace] members = ["evoxc_hf"]` and carries no `hf` extra, and this project's `[tool.uv.sources] evoxc = { workspace = true }` resolves `evoxc` from that root project rather than PyPI; `uv build` does not resolve `[tool.uv.sources]`, so a missing workspace root blocks `uv lock`/`uv sync` but never a build.
- **`./LICENSE` is a tracked relative symlink** (git mode `120000`) to `../LICENSE`, so the Apache-2.0 text is declared once for the repository while `license-files` still ships it — the wheel carries `evoxc_hf-0.1.0.dist-info/licenses/LICENSE`.
- **Version is single-sourced**: `./pyproject.toml`'s `project.version` is the ONE place to bump; `src/evoxc_hf/__init__.py` reads `importlib.metadata.version("evoxc-hf")` and falls back to `"0.0.0+unknown"` for a bare source checkout (mirrors `evoxc/__init__.py`).
- **Measured wheel** (`uv build --project evoxc_hf` → `evoxc_hf-0.1.0-py3-none-any.whl`): exactly `evoxc_hf/**` (13 modules + `models/` incl. its `CONTEXT.md`) plus `evoxc_hf-0.1.0.dist-info/{METADATA,WHEEL,RECORD,licenses/LICENSE}`, and **nothing from `evoxc/`**; METADATA reads `Name: evoxc-hf`, `Version: 0.1.0`, `License-Expression: Apache-2.0`, `License-File: LICENSE`, `Requires-Dist: evoxc>=0.1.0`, `numpy>=1.24`, extras `hf`/`dev`.
- **Tests are not part of the package**: `./tests/` sits outside the src tree and is never shipped, and its modules import their siblings (`hf_download`, `test_export`) through the explicit `sys.path` entry each of them installs — never `from evoxc_hf.tests import ...`.
- `evoxc_hf` imports only public evoxc entry points (`evoxc.trace`, `evoxc.build_cached`, `evoxc.run`, `evoxc.TensorSpec`, `evoxc.core`, `evoxc.backends.stablehlo` in tests), so nothing here changes evoxc's public API or import DAG.
- **Import layering inside the package**: `errors`/`_validate` bottom, then `config` → `weights` → `ops`/`reference` → `models/*` → `families` → `pack` → `load` → `program` → `pipeline` → `__init__`; `models/__init__.py` re-exports nothing (each family module is discovered by `families`), and `models/**` never imports `pipeline`/`load`.

## API Surface

`src/evoxc_hf/__init__.py` re-exports exactly: `LoadedModel`, `ModelConfig`, `load_config`, `load_hf`, `load_params`, `validate_params`, `forward_fn`, `param_specs`, `reference_logits`, `build`, `place_ids`, `place_params`, `predict`, `run`, `PackPlan`, `pack_params`, `pack_plan`, `FAMILIES`, `FamilySpec`, `family_for_model_type`, `get_family`, `HFConfigError`, `HFError`, `HFUnsupportedError`, `HFWeightError`, the submodules `config`/`families`/`load`/`ops`/`pack`/`pipeline`/`program`/`reference`/`weights`, and `__version__` (the installed `evoxc-hf` distribution's version, `"0.0.0+unknown"` in a bare source checkout).
`evoxc_hf.build` is the **function** `build` — there is no `build` submodule and no `HFModel`; `evoxc_hf.models` is reachable as a subpackage but is deliberately not in `__all__`.

### `src/evoxc_hf/load.py` — checkpoint → static config + host params (the data half)

```python
class LoadedModel(NamedTuple):      # immutable, field access only — NO methods, NO seq_len/batch_size
    config: ModelConfig
    params: dict[str, np.ndarray]   # canonical (in_features, out_features) tree, exactly param_specs' keys
    family: str                     # "llama" | "gpt2"
    model_dir: Path | None          # where it came from, or None

def load_config(model_dir, *, family: str | None = None) -> ModelConfig
def load_params(model_dir, config: ModelConfig, *, family: str | None = None) -> dict[str, np.ndarray]
def load_hf(model_dir, *, family: str | None = None) -> LoadedModel      # == load_config + load_params
def validate_params(config, params: Mapping[str, np.ndarray], *, family=None) -> dict[str, np.ndarray]
```

- `load_hf` is the documented one-call composition of the other two and **nothing more**: no compilation, no `seq_len`/`batch_size` (those are *build*-time specialisation values owned by `forward_fn`), no graph caching.
- `load_config` requires a **directory**: it reads `<dir>/config.json` before any weight is touched, so a single checkpoint *file* path fails with `HFConfigError` naming the missing `config.json` (file-level loading is `weights.load_checkpoint`'s job).
- `load_params` reads the checkpoint (`weights.load_checkpoint`), canonicalises it for the family (orientation, fused-QKV split, tied head), `del`s the raw HF state dict as soon as the map is built, and validates the result with `validate_params` — a mis-shaped canonicalisation raises here, not deep inside a run.
- `validate_params` is the documented pre-flight check for a params mapping that did **not** come from `load_params` (hand-edited tree, converted checkpoint); keys and shapes must match `param_specs` exactly.
- `LoadedModel` is the **only** thing "a loaded model" means — config plus params — and every call that needs a graph takes `loaded.config` (and `loaded.params`) explicitly.

### `src/evoxc_hf/program.py` — the compute function (specs, forward, oracle)

```python
DEFAULT_SEQ_LEN: int = 32
def resolve_seq_len(config: ModelConfig, seq_len: int | None = None) -> int      # min(max_position_embeddings, 32)
def forward_fn(config, *, seq_len=None, batch_size=None, family=None,
               pack_weights=None) -> Callable                                  # the memoised @evoxc.defn
def param_specs(config, *, family=None, pack_weights=None) -> dict[str, evoxc.TensorSpec]
def reference_logits(config, params, input_ids, *, family=None) -> np.ndarray
```

- `forward_fn` returns the **whole model pass** — `(params, input_ids) -> logits`, every weight an ordinary graph **input**, all shapes static. It captures **no parameters** and knows nothing about backends, devices or compilation: build it here, stage it with `build`.
- **Memoised** on `(family, config.digest, seq_len, batch_size, pack_weights)` in a bounded 64-entry LRU (built outside the lock, first-insert-wins): identical calls return the **same object**, so `id(fwd)` — the identity component of `evoxc.build_cached`'s key — is stable, and two identical `build` calls return the identical `Executable`.
- The returned object carries its specialisation as `fwd._evoxc_hf_specialisation = {"family", "digest", "seq_len", "batch_size", "pack_weights"}`; `build`/`predict` read their `seq_len`/`batch_size`/`pack_weights` defaults from it and reject an explicit value that disagrees (`HFConfigError` naming `evoxc_hf.forward_fn`) rather than tracing to a mismatched ids spec.
- `param_specs` is the specs tree `build` traces with and `run` validates against; `pack_weights=True` returns the packed inputs (`PackPlan.specs`), default `False`/`None` the canonical per-weight tree.
- `reference_logits` is the parity oracle — same numbers from a *deliberately different* formulation, both inputs explicit (nothing defaults to a checkpoint's own weights).
- `seq_len=None` everywhere means `resolve_seq_len(config, None)` = `min(config.max_position_embeddings, DEFAULT_SEQ_LEN)`; `batch_size=None` means 1.

### `src/evoxc_hf/pipeline.py` — staging (build / run / predict / placement)

```python
DEFAULT_LLVM_CPU_COMPILE_ARGS: tuple[str, ...] = ("--iree-llvmcpu-target-cpu=host", "--iree-opt-data-tiling")
DEFAULT_OPT_LEVEL: str = "O3"
DEFAULT_ROCM_OPT_LEVEL: str = "O1"
def resolve_build_options(backend, device, options: Mapping) -> tuple[str, dict]
def build(fwd, config, *, seq_len=None, batch_size=None, backend=None, device=None, cache=None,
          pack_weights=None, family=None, **options) -> evoxc.Executable
def run(executable, params: Mapping[str, np.ndarray], input_ids) -> evoxc.Tensor
def place_params(params: Mapping[str, np.ndarray], device: evoxc.core.Device) -> dict[str, evoxc.Tensor]
def place_ids(input_ids, device: evoxc.core.Device) -> evoxc.Tensor
def predict(loaded, input_ids=None, *, params=None, seq_len=None, batch_size=None, backend=None,
            device=None, cache=None, pack_weights=None, family=None, **options) -> np.ndarray
```

- **`build`** is the explicit build stage: `evoxc.build_cached(forward_fn(...), param_specs(config), TensorSpec((batch_size, seq_len), int32), backend=..., device=..., cache=..., **resolved_options)` — trace → lower → compile → load, memoised so an identical specialisation returns the **same** `Executable`. Weight identity is deliberately **not** in the key.
- `build` validates in this order: config / positive `seq_len` / `batch_size` / `seq_len <= max_position_embeddings` → family resolution → `resolve_build_options` → **activation gate before any tracing** → the memoised forward → the cache call. A `pack_weights`/`seq_len` disagreement with `fwd` is an `HFConfigError`.
- **`run`** is validation + `evoxc.run`: `input_ids` may be a host integral array (2-D, matching the executable's ids spec; the **one** documented host-side coercion is the cast to `int32`, non-integral → `evoxc.DTypeError`) or an `evoxc.Tensor` validated from **metadata only** and forwarded **verbatim**, and `params` passes through **untouched** except for a packed-vs-canonical mix (`HFWeightError` naming `pack_weights` and the `pack_params` remedy). Outputs are `evoxc.Tensor`s on the executable's device — never copied to the host.
- **`predict`** is the ONE host-facing shorthand: `build(forward_fn(loaded.config, ...), loaded.config, ...)` → `run` → one explicit `Tensor.to(Device("cpu", 0))` when the output is non-cpu → `np.asarray(..., np.float32)`; non-cpu `params` are rejected up front with `evoxc.DeviceError` naming the device-resident recipe, `params=None` means `loaded.params`, `input_ids=None` all-zero ids of the static shape, and a canonical tree under `pack_weights=True` is packed once on the host.
- **`place_params`/`place_ids`** are the explicit placement primitives: one `Tensor.to(device)` per leaf (host ids cast to `int32`; a `Tensor` is never re-typed, only moved). Place **once** and keep the placed tree; the device-resident recipe is `build(..., device=d)` + `place_params` + `place_ids` + `run`, which stages nothing per call.
- **`resolve_build_options`** is the ONE implementation of the default matrix; precedence is binding: **explicit argument > environment > evoxc_hf default**.

| Situation | Resolution |
|---|---|
| `backend` explicit | used verbatim (a name, or an evoxc backend instance with `.name`) |
| `backend=None`, `EVOXC_BACKEND` set | that value (honoured, not fought — evoxc itself would apply it) |
| `backend=None`, device `None`/cpu | `"numpy"` (the interpreter) |
| `backend=None`, device cuda/rocm/vulkan/other | `"iree"` (the compiled path) |
| any non-numpy backend whose effective IREE targets are **all** `rocm`/`hip` | `opt_level="O1"` injected unless passed (`DEFAULT_ROCM_OPT_LEVEL`) — O2/O3 hit an upstream `iree-compile` `'func.func' op failed to distribute` on gpt2-class LayerNorm-reduce shapes, while O0/O1/default emit a byte-identical vmfb |
| any other non-numpy backend | `opt_level="O3"` injected unless passed (`DEFAULT_OPT_LEVEL`) |
| `opt_level` passed explicitly, or `EVOXC_OPT_LEVEL` set | nothing injected |
| `"iree"` whose effective targets are exactly `["llvm-cpu"]` | `iree_compile_args=list(DEFAULT_LLVM_CPU_COMPILE_ARGS)` injected unless passed |
| `"numpy"` | options returned as an unchanged copy — nothing injected |

- Effective IREE targets: `options["target_backends"]`, else `EVOXC_TARGET_BACKENDS`, else derived from the device (`None`/cpu → `["llvm-cpu"]`, cuda → `["cuda"]`, rocm/hip → `["rocm"]`, **vulkan → `["vulkan-spirv"]`** — a device *kind* is not a target-backend name, so the bare `vulkan` would be an unknown-target error at compile time — otherwise `[device.kind]`); the llvm-cpu flags are injected only for that exact target.
- `iree_compile_args` is a **compile-stage** option: `build` forwards options to both `lower` and `compile` through `evoxc.build_cached`, so the flags reach `evoxc.compile` structurally (passing them to `evoxc.lower` alone is silently ineffective).
- **`vulkan` is an accepted device kind for `build(...)`**: `build(fwd, config, device=Device("vulkan", 0))` resolves to backend `"iree"` with the `vulkan-spirv` compile target, the `O3` `opt_level` default (not an all-ROCm target set) and **no** injected `iree_compile_args`; placement and the device-resident loop are the ordinary recipe and `predict` still makes the one explicit `Tensor.to(Device("cpu", 0))` (real-size models need an explicit `opt_level="O1"` — see Known Issues).

### `src/evoxc_hf/config.py` — HF `config.json` → frozen static `ModelConfig`

```python
FAMILY_BY_MODEL_TYPE: dict[str, str] = {"llama": "llama", "mistral": "llama", "qwen2": "llama", "qwen3": "llama", "gpt2": "gpt2"}
@dataclass(frozen=True)
class ModelConfig:
    model_type, family, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
    num_key_value_heads, intermediate_size, hidden_act, layer_norm_eps, max_position_embeddings,
    tie_word_embeddings, rope_theta, head_dim, qk_norm, raw              # `raw`: parsed JSON, diagnostics only
    @staticmethod
    def from_json(path) -> "ModelConfig"
    @staticmethod
    def from_dir(model_dir) -> "ModelConfig"
    @classmethod
    def from_dict(cls, raw, *, source: str = "<dict>") -> "ModelConfig"
    @property
    def digest(self) -> str          # 16-hex sha256 over every graph-specialising field, `raw` excluded
```

Llama reads `rms_norm_eps` (1e-6), `hidden_act` (`"silu"`), `rope_theta` (10000.0), `num_key_value_heads` (= heads), `tie_word_embeddings` (False) and an explicit `head_dim` override else `hidden_size // num_attention_heads`; `qk_norm` is derived from `model_type` alone (`"qwen3"` → `True`) — a static specialisation value, never inferred from checkpoint contents, and covered by `digest` like every other non-`raw` field.
GPT-2 normalises its compact keys (`n_embd`/`n_head`/`n_layer`/`n_vocab`/`n_positions`/`n_inner`/`activation_function`/`layer_norm_epsilon`, with canonical fallbacks), defaults `n_inner` to `4 × hidden_size` (HF writes `null`), activation to `"gelu_new"`, eps to 1e-5, ties embeddings, and pins `rope_theta`/`head_dim` to `None` and `qk_norm` to `False`.
Validation raises `HFConfigError` for head-count divisibility, non-positive/non-int sizes and unknown `model_type` (listing the supported set), and `HFUnsupportedError` for non-empty `rope_scaling` or any Qwen3 sliding-window request — truthy `use_sliding_window`, a `layer_types` entry other than `"full_attention"`, or a positive `sliding_window` without `use_sliding_window: false`.
`ModelConfig` values are **static graph specialisation values**: one config = one graph, and `digest` is directly usable as a cache key (configs differing only in harmless extra HF fields digest equal).

### `src/evoxc_hf/weights.py` — checkpoint readers (numpy + stdlib only, no key mapping)

```python
read_safetensors(path) / read_torch_bin(path) / load_checkpoint(model_dir_or_file) -> dict[str, np.ndarray]
widen_to_float32(arr) -> np.ndarray
state_dict_parameter_count(state_dict) -> int
```

- **safetensors**: 8-byte LE u64 header length + JSON header + raw buffer; `__metadata__` skipped; dtype/shape/offsets validated against the buffer; sharded checkpoints merged through the index `weight_map`.
- **torch**: zip format (`*/data.pkl` + `*/data/<storage-key>`) vs legacy format (magic → protocol 1001 → `sys_info` → pickled state dict → storage-key list → per key an **int64 element count followed by raw bytes**), chosen by `zipfile.is_zipfile`, never by name; views/offsets/strides (negative included) resolved; `{"state_dict": {...}}` unwrapped.
- **Security boundary**: `_RestrictedUnpickler` allows only `collections.OrderedDict`, the three `torch._utils` rebuild helpers, storage classes from its dtype table, inert placeholders and `copyreg` reconstructors — any other global raises `HFWeightError` instead of executing code (no `allow_pickle`, no torch import).
- **Dtype policy**: float32 preserved, float16/bfloat16 widened to float32 (bfloat16 via `u16 << 16`), float64/integer/bool kept; every array is writable and C-contiguous.
- **Directory candidate order**: `model.safetensors` → `pytorch_model.bin` → `model.safetensors.index.json` → `pytorch_model.bin.index.json` (an absent checkpoint lists every candidate); a file argument is dispatched by suffix, else by content sniffing.
- Returned names are the **on-disk** keys — orientation and canonical naming happen in the family canonicalisers, never here.

### `src/evoxc_hf/ops.py` — the transformer idioms `evoxc` does not ship (graph→graph compositions)

```python
softmax, layer_norm, rms_norm, silu, gelu_tanh, gelu_erf, resolve_activation, activation_is_export_safe,
ACTIVATIONS, EXPORT_UNSAFE_ACTIVATIONS, MASK_NEG,
causal_mask(seq_len), rope_cos_sin(seq_len, head_dim, theta), rope_tables(...), apply_rope(x, cos, sin),
split_heads(x, n_heads), merge_heads(x), repeat_kv(x, n_rep)
```

Every helper is a composition of public evoxc ops on `SymbolicTensor`s (no new IR op, no eager path — a concrete `Tensor` surfaces the underlying `evoxc.TraceError`), **static-shape only** (`_static_dim` is the single enforcement point; a symbolic `Dim`/`DimExpr` raises `HFUnsupportedError` naming the axis), and never casts implicitly.
`EXPORT_UNSAFE_ACTIVATIONS`/`activation_is_export_safe` exist so the pack/build layer can refuse the compiled path up front for erf-based GELU; `rope_tables` is host numpy for the oracle only and must never feed the graph.

### `src/evoxc_hf/reference.py` — the independent pure-numpy oracle

`softmax`, `layer_norm`, `rms_norm`, `silu`, `gelu_tanh`, `gelu_erf`, `relu`, `resolve_activation`, `rope_tables`, `apply_rope`, `additive_causal_mask`, `attention`, `param_count`, `MASK_NEG`, `ACTIVATIONS`.
Deliberately written in a *different* formulation (per-head Python loops in `attention`, hand-written mean/variance, `math.erf`), so a parity failure points at exactly one side; hard rules: numpy + stdlib only (the single `evoxc_hf` import is `HFUnsupportedError`), every numeric input/return normalised to `float32`, explicit `ValueError`/`TypeError` on malformed shapes/dtypes.

### `src/evoxc_hf/families.py` — the family registry (the one place families are enumerated)

```python
@dataclass(frozen=True, slots=True)
class FamilySpec:
    name: str                      # "llama" | "gpt2"
    model_types: tuple[str, ...]   # every HF model_type it covers, sorted
    canonicalize_weights: Callable # (cfg, raw) -> dict[str, np.ndarray]
    param_specs: Callable          # (cfg) -> dict[str, evoxc.TensorSpec]
    graph: Callable                # (cfg, seq_len, batch_size=1, *, pack_plan=None) -> @evoxc.defn forward
    reference: Callable            # (cfg, params, input_ids) -> np.ndarray
FAMILIES: MappingProxyType[str, FamilySpec]
get_family(name) -> FamilySpec                       # HFUnsupportedError otherwise
family_for_model_type(model_type) -> FamilySpec
resolve_family(config, family=None) -> FamilySpec    # the ONE "config + optional override" resolver
```

`resolve_family` is the single place the disagreement rule lives — `load`, `program` and `pipeline` all go through it, so it can never drift between the loading half and the compute half.
Two invariants are checked at **import** time (`RuntimeError`): every `FamilySpec.name` equals its `FAMILIES` key and its module's `FAMILY`, and a module-declared `MODEL_TYPES` must agree with the config map.
Adding a family is one module plus one `FAMILIES` entry — `load`/`program`/`pipeline` never hard-code a family.

### `src/evoxc_hf/models/_common.py`, `llama.py`, `gpt2.py` — one family per module

```python
# _common.py (shared verbatim by both families)
linear(x, weight, bias=None), attention(q, k, v, mask, scale),
gated_mlp(x, w_gate, w_up, w_down, act), two_layer_mlp(x, w1, b1, w2, b2, act)
# per family
FAMILY: str; MODEL_TYPES: tuple[str, ...]
canonicalize_weights(cfg, raw), param_specs(cfg), graph(cfg, seq_len, batch_size=1, *, pack_plan=None), reference(cfg, params, ids)
```

Graph contract (both families): `params` is the flat canonical dict (exactly the `param_specs` keys), `input_ids` is `(batch_size, seq_len)` `int32`, `logits` is `(batch_size, seq_len, vocab_size)` float32; all shapes static; every weight a graph **input**, never `evoxc.constant`.
**Orientation rule — `(in_features, out_features)`** for every weight fed to `evoxc.dot(x, W)`: llama linears are `(out, in)` in HF and transposed at load; GPT-2 `Conv1D` weights are already `(in, out)` and are **not** transposed; `lm_head.weight` is `(hidden, vocab)` in both (HF stores `(vocab, hidden)`).
**Tied embeddings — one graph input, never two**: `tie_word_embeddings=True` omits `lm_head.weight` from both `param_specs` and `canonicalize_weights`, and the graph feeds `params["embed.weight"]` to the token gather and to the final projection — as `evoxc.transpose(embed)` on the canonical (unpacked) path, or as the head key the **packed** path supplies pre-transposed (`params["lm_head.weight"]`, a static slice of its own packed entry, when `pack_plan.has("lm_head.weight")`), so a packed tied graph contains no weight-sized transpose at all.
A tied tree carries one key fewer, which is exactly the model's true parameter count.
**Canonical key tables (llama `2 + 9·L` keys, gpt2 `16·L + 4`, plus the untied head; Qwen3 adds `2·L` QK-Norm keys) live in `./src/evoxc_hf/models/CONTEXT.md`** — the per-key HF-source, shape and orientation reference, driven by one `_layer_records(cfg)` list per family so specs/weights/oracle cannot drift.
**QK-Norm placement**: in both the graph and the numpy oracle, `rms_norm(q, attn.q_norm.weight, eps)` / `rms_norm(k, attn.k_norm.weight, eps)` run **after `split_heads` and before `apply_rope`**, over the last axis `head_dim` (the HF `Qwen3Attention` order); a Qwen3 config missing those keys is a `HFWeightError` naming QK-Norm, as is a non-Qwen3 config carrying them.
Llama attention dims derive from `cfg.head_dim`; GPT-2 always uses `hidden_size // num_attention_heads`.
Both canonicalisers resolve tied heads explicitly: a tied config shipping `lm_head.weight` is verified (`np.allclose`), consumed and dropped; a disagreeing tied head is `HFWeightError`, and an untied config without it is `HFWeightError` listing the remedy.
Unexpected/duplicate keys are errors, never silently dropped; each family has one documented ignore list — llama `model.rotary_emb.inv_freq`, GPT-2 `h.{i}.attn.bias` / `h.{i}.attn.masked_bias`.
Llama rejects **any** `*.bias` tensor (`HFUnsupportedError`, v1 builds are bias-free); GPT-2 tolerates a `transformer.` prefix in any mix (duplicates rejected).
Both graphs compose the causal mask and (llama) the RoPE tables **in-graph** and read every weight from `params` by its flat canonical key.

### `src/evoxc_hf/pack.py` — opt-in packed weights (few graph inputs, static in-graph slices)

```python
PACKED_PREFIX: str = "packed."
@dataclass(frozen=True)
class PackEntry:
    input: str; start: tuple[int, ...]; shape: tuple[int, ...]   # shape = the STORED (unpacked) shape
    source: str | None = None     # None → a canonical key; else the canonical key this entry derives from
    transposed: bool = False      # stored data is params[source].T
    @property
    def canonical(self) -> bool   # == (source is None)
@dataclass(frozen=True)
class PackPlan:
    specs: dict[str, evoxc.TensorSpec]   # the PACKED params inputs (4-5 for these families)
    entries: dict[str, PackEntry]      # canonical + derived key -> entry
    canonical_keys / derived_keys / unpacked_keys   # introspection (sorted tuples)
    def has(self, name) -> bool
    def pack(self, params) -> dict[str, np.ndarray]
    def unpack(self, packed) -> dict[str, SymbolicTensor]   # canonical keys + derived keys
def pack_plan(specs: Mapping[str, evoxc.TensorSpec], *, prefix=PACKED_PREFIX,
              transposed_aliases: Mapping[str, str] | None = None) -> PackPlan
def pack_params(plan: PackPlan, params) -> dict[str, np.ndarray]
def is_packed_tree(tree) -> bool
```

**Derived (pre-transposed) entries** — a weight's orientation is decided at `param_specs` time, not per call: `pack_plan` resolves the declaration from the spec tree itself (`evoxc_hf._specs.transposed_aliases(specs)`: an `embed.weight` with no `lm_head.weight` is a tied tree, so it declares `{"lm_head.weight": "embed.weight"}`), an explicit `transposed_aliases=` mapping overrides it and `{}` forces none.
The declaration rides on the **plan**, never on the returned mapping — a `dict` subclass would break evoxc's typed pytree structure matching at the run boundary, so `pipeline.build` hands `evoxc.build_cached` a plain `dict` and a plain-dict conversion is a safe fallback to the canonical in-graph `transpose`.
The derived key gets its own bucket keyed by its **stored** (transposed) shape, `pack()` writes `np.ascontiguousarray(params[source].T)` exactly once, and `unpack()` returns it as a STATIC `evoxc.slice` of that bucket — so the graph never transposes a weight-sized operand.

Generic over any `param_specs` tree — **no family-specific code**: buckets are `(dtype, rank, shape[1:])` groups concatenated along axis 0, ordered by their minimum canonical key with keys sorted inside a bucket and offsets as prefix sums, so a plan is deterministic for equal spec trees; rank-0 scalars get their own bucket and unpack through `reshape(slice(...), ())`.
Packed input counts are shape-driven, not role-driven — read them from `pack_plan(...).specs`, never assume a number (llama 5 untied and tied, gpt2 4 untied and tied, so the real 124M `gpt2` declares 5 inputs with `input_ids` against 197 unpacked); the element count equals the canonical total **plus exactly one transposed copy of the tied head**, and everything else is lossless and bit-identical.
Families opt in with one line: `graph(cfg, seq_len, batch_size, *, pack_plan=None)` and, as the forward's **first** statement, `params = pack_plan.unpack(params)`.
Pure graph→graph composition: **no new IR ops** (one `evoxc.slice` per canonical key) and never an `evoxc.constant` of weights; symbolic/dynamic dims → `HFUnsupportedError`, missing/extra keys or shape/dtype mismatches → `HFWeightError` naming the key.
Packing costs the compiled program exactly **1× the weight bytes per call** — every bucket is an axis-0 concatenation of same-trailing-shape matrices in their canonical `(in, out)` orientation and every slice has exactly one consumer, so the graph holds **no weight-sized layout op at all** (every `dot` RHS on the real 124M `gpt2` traces to a `slice` of a packed input, and the only remaining transposes are activation head reshuffles).

### `src/evoxc_hf/_specs.py` — the packing-orientation rule

```python
EMBED_WEIGHT = "embed.weight"; HEAD_WEIGHT = "lm_head.weight"
TRANSPOSED_ALIASES: Mapping[str, str] = {HEAD_WEIGHT: EMBED_WEIGHT}
def transposed_aliases(specs: Mapping[str, object]) -> dict[str, str]
```

Stdlib-only, no carrier class: it owns the ONE static rule "a spec tree with an embedding and no head of its own is a tied tree, so its head is the transposed embedding", returning `{}` otherwise. Deterministic on the tree, so equal trees always produce equal plans.

### `src/evoxc_hf/errors.py` and `src/evoxc_hf/_validate.py`

```python
class HFError(evoxc.core.EVOXCError)      # catch-all for the whole integration
class HFConfigError(HFError)          # malformed/unsupported config.json, bad seq_len/batch_size, family override disagreement, forward specialisation mismatch
class HFWeightError(HFError)          # checkpoint reading, key mapping, tensor shapes, tied-head disagreement, packed-vs-canonical mix
class HFUnsupportedError(HFError)     # outside the v1 matrix: unknown family, unknown activation, erf-GELU on a compiled backend, llama biases, rope_scaling, symbolic dims
```

Pure host-side argument errors (a non-positive `n_heads` to `ops.split_heads`, a non-numeric `input_ids` in `reference.py`) raise `ValueError`/`TypeError`, never an evoxc error class.
`_validate.py` holds the internal argument guards (`positive_int`, `require_config`, `require_bool`, `require_device`) shared by `load`/`program`/`pipeline`, so the same message shape is used everywhere.

## Design Decisions

- **Functional, not OOP.** Behaviour lives in module functions taking their inputs as arguments and `LoadedModel` is a `NamedTuple` of field access only, so no call can silently reuse a stale `seq_len`, backend or placed tree — every piece is passed explicitly, exactly like `evoxc`'s pipeline-object model.
- **The compute function captures no parameters.** `forward_fn(config, seq_len=..., batch_size=..., pack_weights=...)` is pure in its static specialisation and weights are graph inputs (`param_specs`) — which is why the build cache can exclude weight identity, why one graph serves every checkpoint with the same config, and why a test can assert no ndarray appears in any closure or default of the returned forward.
- **`seq_len`/`batch_size`/`pack_weights` are *build*-time specialisation values**: `LoadedModel` carries no `seq_len`, and `forward_fn` stamps its choice on the returned object (`_evoxc_hf_specialisation`) so `build`/`predict` derive the ids spec from the graph that actually exists — an explicit disagreement is an `HFConfigError`, never a silently mismatched spec.
- **Memoisation on both layers.** The 64-entry forward LRU (on `(family, digest, seq_len, batch_size, pack_weights)`) keeps `id(fwd)` stable so `evoxc.build_cached` returns the **same** `Executable` for an identical `build` call; weights are excluded because they cannot change the traced program.
- **Static shapes are baked in**, not symbolic: dynamic-dim reshape/reduce paths are the exporter's per-op limitation, and "one graph per `(config, seq_len, batch_size)`" is the documented, memoised contract — a mismatched length is an explicit error, never a recompile surprise.
- **In-graph RoPE and mask, not host tables.** `ops.rope_cos_sin` builds the tables from `arange` + `power` + `cos`/`sin` (all StableHLO-exportable), keeping the signature at `(params, input_ids)`, the emitted IR op count independent of `seq_len` (pinned by `tests/test_export.py`) and the MLIR free of inlined payload bytes; `ops.rope_tables` is oracle-only.
- **The activation export gate runs before any tracing** (`_check_activation`): an erf-GELU config on a compiled backend fails immediately with `HFUnsupportedError` naming the activation, the backend and both remedies (`backend="numpy"`, or a `gelu_new` config), instead of failing deep inside lowering; the numpy interpreter accepts every supported activation.
- **`run`'s id-dtype coercion is the only host-side type coercion**: integral token ids are cast to `int32` (evoxc's run boundary is dtype-strict), params are never filtered, renamed or cast, and an `evoxc.Tensor` of ids is forwarded verbatim — which is what makes the device-resident recipe reachable.
- **`predict` is the only place a host transfer happens**, one documented `Tensor.to(Device("cpu", 0))` on the *output* of a non-cpu executable, and it refuses non-cpu params with the device-resident recipe rather than copying.
- **Device-resident is explicit on both sides**: `place_params` + `place_ids` then `run`; the run boundary rejects host inputs for a non-cpu executable by design.
- **Packed weights are an explicit, opt-in composition, never a silent default.** Packing removes IREE's Python-runtime ~161-*consumed*-input cliff on llvm-cpu, bucketing by `(dtype, rank, shape[1:])` beats a hand-written per-role scheme and keeps every weight a graph input; unpacked stays default because packing is not free (see Known Issues).
- **A weight's orientation is a pack-time decision, not a per-call op**: the tied head needs `(hidden, vocab)` while the token gather needs `(vocab, hidden)`, and `evoxc.dot` has no `b_transposed`, so the pack stores the transposed copy once (one extra weight-sized host+device buffer) instead of the graph transposing a weight-sized table on EVERY call, and the decision is made at `param_specs` time from static metadata (`tie_word_embeddings` + shapes).
  The alternative (an axis-1 token gather over the single untransposed table) would save the duplicate but puts a gather — not a static slice, not demoed through every adapter — on the hot path, so the copy is the cheaper trade.
- **Pack once on the host, then place — never per call**: `pack_params` runs at build/params time and `place_params` then stages 3-5 leaves instead of ~200, so the same-device loop keeps its zero-host-round-trip property.
- **`opt_level` is per-target** — `"O3"` generally, `"O1"` when every effective IREE target is ROCm — because O2/O3 trip an upstream `iree-compile` failure on these LayerNorm-reduce shapes; encoding it as a default rather than a workaround is what makes real-size ROCm builds work, and an explicit `opt_level`/`EVOXC_OPT_LEVEL` still wins.
- **Legacy pickles are a security boundary** (`weights._RestrictedUnpickler`): an explicit allowlist, every rejection naming the offending `module.name`.
- **HF layout knowledge lives only in the family modules** (`canonicalize_weights`): readers stay format-only, so a mapping bug is testable one layer at a time and a new family is one module plus one registry entry.
- **Two-formulation parity**: the graphs use evoxc ops and the oracle plain numpy in different shapes of code, so a shared bug would have to be written twice to pass.

## Constraints

- **No new hard dependency**: `torch`/`transformers` appear only behind availability guards as golden oracles, `iree` only behind `skipif` guards.
- **Never modify `../evoxc/`** (sibling node — read-only; escalate core gaps to the parent agent). Gaps are composed around here and documented: `rsqrt` → `1/sqrt`, `erf` → numpy-backend-only GELU.
- **Explicit errors only**: everything derives from `evoxc.core.EVOXCError` via `HFError` (or `ValueError`/`TypeError` for pure host-side argument errors), messages name the file/field/tensor/key, and there is never a silent fallback, dtype swap or positional guess.
- **Static-shape-first**: dynamic dims are not a goal; a mismatched `seq_len`/`batch_size` is an explicit `HFConfigError`/`evoxc.ShapeError`.
- **Files stay small** (~1000 lines baseline, one concern per module); no duplicate family logic — shared attention/norm/MLP pieces live in `ops.py` / `models/_common.py`. `pipeline.py` sits at ~1110 lines: one concern (the verbatim option matrix plus `build`/`run`/`predict`/placement), each function documented in place.
- **Packaging**: the version lives only in `evoxc_hf/pyproject.toml`; the package reads it back from the installed distribution metadata.
- **Status (measured)**: **727 collected — 694 passed / 33 skipped** with `PYTHONPATH=evoxc_hf/src uv run --with pytest python -m pytest evoxc_hf/tests` in the base env (no workspace member installed), zero failures and zero errors; every skip is a host/dependency guard (torch/transformers absent, `EVOXC_HF_BIG` unset, `iree` absent).
  The real `Qwen/Qwen3-0.6B` legs run explicitly with `EVOXC_HF_BIG=1` + `EVOXC_HF_MODELS_DIR=…` under a torch/transformers venv.
- **Status (capability)**: both families trace, verify, export to StableHLO and run on the numpy interpreter, and the same graph compiles and runs on **every validated compiler adapter** — one IREE program on `llvm-cpu` and — packed where needed — device-resident on ROCm, plus device-resident **XLA/ROCm** via a user-provided PJRT plugin.
  Adapter coverage for the whole forward: `numpy` (host), `iree` (llvm-cpu + ROCm), `xla` (ROCm — measured by the benchmark harness, not a gated test); **`tvm` is UNVALIDATED for `evoxc_hf`** (its option handling is unit-tested only).
  On llvm-cpu the compiled graph is comparable to the interpreter (whose OpenBLAS matmuls win on CPU); on ROCm it is ~5× (tiny) to ~13× (packed 124M `gpt2`) faster. Numbers: `../bench_logs/hf_bench_results.md`.
- **Status (parity)**: against the real HuggingFace implementation (`tests/test_torch_oracle.py`) `Qwen/Qwen3-0.6B` agrees at ~6e-5 max-abs (≈3e-6 of the ~22.8 absmax) at `seq_len` 8 batch 1, batch 2 and `seq_len` 4, with argmax agreement 1.0 at every position and bit-identical repeats, and scaling `layers.0.attn.q_norm.weight` by 1.5 moves the logits by the same 2.0067 in HF, the oracle and the graph.
  `tiny_llama`, `tiny_mistral` and `sshleifer/tiny-gpt2` agree to ≤1.8e-7 max-abs, and the torch-free goldens over `../bench_logs/hf_goldens/` range from bit-exact (`tiny_gpt2`) to 1.5e-4 (`gpt2`).
- **Status (vs PyTorch, `../bench_logs/hf_bench_results.md`)**: one whole-forward graph in the device-resident same-device loop (`asdevicearray` = 0) over five checkpoints at `seq_len` 32/128 — **`xla`/rocm beats torch-rocm in 6/6 measured cells (7.2×–10.0×)** and **`iree`/rocm in 5/6 (0.73×–9.1×)**, the single loss being `gpt2` at `seq_len=128`.
  XLA's default fp32 `dot` is TF32-class, so the opt-in `dot_precision="HIGHEST"` leg is the bit-accurate one (rel-to-absmax ~2e-6, argmax 1.0, 3%–29% more latency).
  Legs that cannot run fail explicitly, never silently: `tiny_mistral` on `iree`/`xla` (its `gelu` needs `evoxc.erf`) and `distilgpt2`/`gpt2` at `seq_len=128` on iree-llvm-cpu (upstream `iree-compile` stack-allocation limit); the xla legs are ROCm-only (no XLA CPU plugin on that host).

## Known Issues

- **Exact-erf GELU is numpy-backend-only**: `hidden_act: "gelu"`/`"erf"` builds `ops.gelu_erf` (which uses `evoxc.erf`, a v1 StableHLO export deferral), so the graph runs on the numpy interpreter and raises an explicit `evoxc.BackendError` on any compiler backend — after the even earlier `HFUnsupportedError` from the activation gate, and never as a silent tanh substitute.
- **Static shapes only** — one graph per `(config, seq_len, batch_size)`; `seq_len > max_position_embeddings` is `HFConfigError` and a mismatched `input_ids` shape is `evoxc.ShapeError` telling you to build another length rather than reshape.
- **The llama family is bias-free**: any `*.bias` key raises `HFUnsupportedError` (v1 covers `attention_bias=False`/`mlp_bias=False` models only).
- **Qwen3 is full-attention only**: `model_type: "qwen3"` is supported (Llama architecture + per-head QK-Norm, validated on the real `Qwen/Qwen3-0.6B` on the numpy interpreter plus whole-graph StableHLO export; no compiler-adapter run is recorded for a Qwen3 checkpoint yet), but any sliding-window request raises `HFUnsupportedError` from the config guard — a truthy `use_sliding_window`, a `layer_types` entry other than `"full_attention"`, or a positive `sliding_window` without `use_sliding_window: false`.
  QK-Norm is declared by the config and never inferred from checkpoint keys; bf16/fp16 checkpoints load normally (widened to float32), fp8 safetensors tags (`F8_E4M3`/`F8_E5M2`) raise `HFWeightError`, and a scaled-RoPE config hits the global `HFUnsupportedError`.
- **Per-family key allowlists**: llama ignores `model.rotary_emb.inv_freq`; GPT-2 ignores `h.{i}.attn.bias` / `h.{i}.attn.masked_bias`. Everything else must map onto the canonical scheme or it is an `HFWeightError` listing the offending keys.
- **`rope_scaling` is unsupported** (`HFUnsupportedError` unless `None`/`{}`/`[]`).
- **No KV cache, no incremental decode, no sampling, no training**: every call re-runs the whole graph over `seq_len` tokens.
- **`load_config`/`load_hf` require a directory**: `<dir>/config.json` is read before any weights, so a single checkpoint *file* path fails with `HFConfigError` naming the missing `config.json` (single-file reading is `weights.load_checkpoint`'s job).
- **numpy-on-CPU vs the compiled graph**: on CPU the interpreter can beat the compiled graph for large matmuls (OpenBLAS) while the compiled path's value is GPU — report both, never conflate them (numbers in `../bench_logs/hf_bench_results.md`).
- **Device-resident run costs**: the FIRST `place_params` leaf in a process pays a one-time HIP driver init, re-placing per call instead of keeping the placed tree costs about an order of magnitude more than the same-device loop, and the first `.to(Device("cpu", 0))` readback is the slowest one (numbers: `../bench_logs/hf_bench_results.md`).
- **Packing is not free, just cheap**: the packed 124M `gpt2` tree is fresh concatenated host arrays plus the pre-transposed tied head (peak RSS ~1.7 GB packed vs ~1.1 GB unpacked-numpy, since canonical and packed trees briefly coexist) and the interpreter pays ~0-2% extra latency; its payoff is the input count (197 → 5) and the staging leaf count (196 → 4).
- **ROCm compiles gpt2-class graphs only at `opt_level` O0/O1** (O2/O3 hit an upstream `iree-compile` `'func.func' op failed to distribute` on the LayerNorm mean reduce; O0/O1/default emit a byte-identical vmfb), which is why the all-ROCm default is `"O1"` — an explicit `opt_level`/`EVOXC_OPT_LEVEL` overrides it and owns the risk.
  ROCm fp32 `dot` is true fp32, so `dot_precision="HIGHEST"` is a no-op there (pinned by bit-identical logits) and ROCm parity is fp32 accumulation-order noise — real-size comparisons must be relative-to-absmax, never element-wise.
- **The IREE Python-runtime ~161-consumed-input `_invoke` cliff exists on llvm-cpu only — and `pack_weights=True` removes it**: a graph that **consumes** more than ~161 distinct inputs compiles, loads and then SIGSEGVs at the first run (exit 139, no traceback, `iree/runtime/function.py` `_invoke`), the trigger being input *consumption* rather than declared-argument count, op count or tensor size — no compile flag avoids it.
  The 124M `gpt2` (197 inputs) is in that class and runs packed at a few hundred ms/call on llvm-cpu with bit-identical repeats, while `distilgpt2` (101 inputs) is unaffected; **ROCm has no such cliff**, so packing is optional there. Measured breakdown: `../bench_logs/hf_bench_data/`.
- **A tied head is one array, so a tied tree is the model's true parameter count**: a tied checkpoint that ships its `lm_head.weight` copy is verified against the embedding and dropped, never kept as a second graph input; the *peak* RSS exceeds the retained canonical tree only because the raw safetensors dict briefly coexists (`load_params` drops it afterwards).
- **A tied head costs no in-graph transpose when PACKED, but an UNPACKED tied graph still transposes the whole embedding per call**: the canonical path builds `linear(x, evoxc.transpose(embed))` (154 MB per forward for `gpt2`, ~1.2 GB for a 1.7B-class model) while the packed path slices a once-stored pre-transposed head — an embedding-sized device copy per call saved for one extra weight-sized host+device buffer.
  Pinned by `tests/test_pack.py` (`test_tied_head_has_no_weight_sized_transpose`, `test_untied_path_is_unchanged`); `evoxc.dot` has no `b_transposed` parameter, which is why the orientation is encoded at pack time.
- **IREE/ROCm trails XLA/ROCm by ~5–12× on this same single-graph forward, and it is a backend-throughput gap, not weight duplication**: the committed `../bench_logs/hf_bench_data/` records show iree/rocm achieving a small fraction of xla/rocm's GEMM throughput and degrading faster with `seq_len`, with the latency ratio smallest exactly where the tied transpose exists — so the transpose does not explain it; load-time facts belong to the iree ROCm adapter.
- **Vulkan (`device=Device("vulkan", 0)` → `iree` + `vulkan-spirv`) is compile-limited on real-size models**: IREE's default Vulkan target profile `vp_android_baseline_2022` allows only **16384 bytes** of workgroup shared memory, so 124M-class graphs do not compile at the default `opt_level="O3"`; the measured remedy is an explicit `opt_level="O1"`, and a larger-profile `--iree-vulkan-target=<profile>` lifts the shared-memory limit but not the later vector-lowering failure.
  Tiny checkpoints compile with the defaults, and both knobs are explicit caller options — evoxc_hf injects neither. Host/driver reality and the measured software-Vulkan (lavapipe) matrix: [`../bench_logs/vulkan_findings.md`](../bench_logs/vulkan_findings.md).
- **`evoxc` ships no `rsqrt`, top-level `where`, `softmax`, `layer_norm`, `rms_norm`, `silu` or `iota`**, and its `slice` takes static start/limit tuples (no data-dependent slice): all composed in `ops.py`.
- **Legacy `pytorch_model.bin` layout**: the legacy format writes an **int64 element count before each storage's bytes** (after the pickled state dict and the storage-key list) and stores storages in trailer key order, so `weights._read_torch_legacy` must read the count first — consuming `numel × itemsize` bytes **without** that prefix shifts every storage.
  A shifted parse still yields correctly-shaped, plausible tensors and graph-vs-`reference` parity cannot detect it (both sides read the same arrays) — validate a legacy `.bin` against an external oracle or assert the reader consumes the file exactly; the `~1e-44` "degenerate checkpoint" figure in `../bench_logs/hf_feasibility_findings.md` describes a shifted parse, not `sshleifer/tiny-gpt2`.

## Routing Table

| Path / module | Area |
|---|---|
| `./src/evoxc_hf/load.py` | Loading half: `LoadedModel`, `load_config`, `load_params`, `load_hf`, `validate_params`. |
| `./src/evoxc_hf/program.py` | Compute half: `forward_fn` (+ memoisation/specialisation stamp), `param_specs`, `reference_logits`, `resolve_seq_len`, `DEFAULT_SEQ_LEN`. |
| `./src/evoxc_hf/pipeline.py` | Staging: `build`, `run`, `predict`, `place_params`, `place_ids`, `resolve_build_options`, the `DEFAULT_*` option constants and the default backend/option matrix. |
| `./src/evoxc_hf/config.py` | HF `config.json` → `ModelConfig`, `digest`, `FAMILY_BY_MODEL_TYPE`. |
| `./src/evoxc_hf/weights.py` | Checkpoint reading: safetensors, torch zip/legacy, sharded indices, dtype policy, restricted unpickler (no key mapping). |
| `./src/evoxc_hf/ops.py` | Graph idioms (norms/activations/softmax/mask/RoPE/head layout) + `ACTIVATIONS`/`EXPORT_UNSAFE_ACTIVATIONS`. |
| `./src/evoxc_hf/reference.py` | Independent pure-numpy oracle primitives (no `evoxc` import) — the thing graphs are validated against. |
| `./src/evoxc_hf/families.py` | `FamilySpec`, the `FAMILIES` registry, `get_family`, `family_for_model_type`, `resolve_family`. |
| `./src/evoxc_hf/models/` | Per-family graph builders (`llama.py` — llama/mistral/qwen2/qwen3, the Qwen3 case adding per-head QK-Norm — and `gpt2.py`) plus the shared block assembly `_common.py`; `models/__init__.py` re-exports nothing (import acyclicity). |
| `./src/evoxc_hf/pack.py` | Opt-in packed weights: `PackPlan`/`PackEntry`/`pack_plan`/`pack_params`/`is_packed_tree` — bucket the canonical `param_specs` by `(dtype, rank, shape[1:])`, concatenate along axis 0 on the host, unpack in-graph with static `evoxc.slice`; derived entries store a pre-transposed copy so the graph never transposes a weight. |
| `./src/evoxc_hf/_specs.py` | The packing-orientation rule: `transposed_aliases(specs)` → `{derived_key: source_key}` (a tied tree declares its pre-transposed head; nothing else does). |
| `./src/evoxc_hf/errors.py` | The `HFError` hierarchy (rooted at `evoxc.core.EVOXCError`). |
| `./src/evoxc_hf/_validate.py` | Internal argument guards shared by `load`/`program`/`pipeline`. |
| `./src/CONTEXT.md` | src-layout stub: the package is `./src/evoxc_hf/`, the tests stay at `../tests/`, the metadata at `../pyproject.toml`, nothing else under `src/`. |
| `./pyproject.toml` | Distribution metadata: name `evoxc-hf`, static version, the `hf`/`dev` extras, `uv_build` with `module-root = "src"` / `module-name = "evoxc_hf"`, `[tool.uv.sources] evoxc = { workspace = true }`. |
| `./README.md` | The PyPI-facing readme (`readme` in `./pyproject.toml`), rendered as the project page on PyPI. |
| `./LICENSE` | Tracked relative symlink (mode `120000`) to `../LICENSE`, shipped through `license-files`. |
| `./tests/` | pytest suite, outside the src tree and never shipped (parity, export, iree/llvm-cpu + iree/ROCm device-resident gates, packed-weight pins, real-checkpoint gates, the functional-API design pins in `test_public_api.py`); `tests/hf_download.py` is the guarded checkpoint downloader for the network-dependent tests. |

Cross-references (read-only): `../evoxc/` → the EvoX Compiler package (`evoxc`) this distribution builds on, whose root `pyproject.toml` is the workspace root project (public API contract in `../evoxc/CONTEXT.md`); `../bench_logs/` → the HF feasibility spike, probe scripts and measured results — notably the HF-vs-PyTorch harness `../bench_logs/hf_bench.py` (+ `hf_bench_env.py`, records in `../bench_logs/hf_bench_data/`) and its report `../bench_logs/hf_bench_results.md`, the older `../bench_logs/hf_perf_results.md`, and `../bench_logs/hf_goldens/`, the torch-golden fixtures read by `tests/test_torch_oracle.py`.
Root-level packaging pins live in the root files (`../tests/test_packaging.py`, the workspace stanza in `../pyproject.toml`), not here.

## Notes for Agents

- **The functional surface is the contract.** No re-introduction of a loaded-model class with methods, no `HFModel`, no `build_model`/`run_model`, and no `seq_len` on the loaded record: config data (`load_*`), compute function (`forward_fn`), and staging (`build`/`run`/`predict`) stay separate layers.
- The StableHLO exporter defers `erf`, `gelu`, `tril`, `triu`, `cumsum`, `cumprod`, `flip`, `roll` (all verified to raise `BackendError`) — do not put them in a compiled graph; compose from comparisons/`select`/arithmetic instead, as `ops.py` does.
- `evoxc.dot` does not auto-cast mixed dtypes (unlike elementwise ops): pass weights of the same dtype as the activations (every `evoxc_hf` spec is float32).
- Never `evoxc.constant` a weight — it inlines the payload into the emitted MLIR text; weights are graph inputs (`param_specs`) by contract.
- Export needs an explicit trace first: `fwd = forward_fn(config, seq_len=s)` → `evoxc.trace(fwd, param_specs(config), evoxc.TensorSpec((b, s), evoxc.int32))` → `graph.verify()` → `stablehlo.export(graph)` (`stablehlo.export` on a `Defn` raises `TypeError`).
- `param_specs` is a **function** (`param_specs(config)`), never a mapping.
- **Dev loop**: at the repo root create the env with `uv sync --all-packages` — a plain `uv sync` uninstalls the workspace member and leaves an importable-looking namespace directory at `evoxc_hf/`, so imports would silently resolve to the wrong thing.
  Without installing the member, prefix test runs with `PYTHONPATH=evoxc_hf/src` (`uv run --with pytest python -m pytest evoxc_hf/tests`), and use `uv run --all-packages --extra hf …` (or `--extra iree`) when the torch/transformers oracles or the compiled path are wanted.
  `pytest` is not in the base uv env, the `iree` adapter's nanobind teardown can swallow the `-q` summary line (count with `--junit-xml`), and `uv run --extra iree` mutates the shared venv (`uv sync` restores the documented base env).
- **Version**: bump `evoxc_hf/pyproject.toml` only — `__version__` comes from the installed distribution metadata and is `0.0.0+unknown` in a bare checkout, so tests accept either value.
- **Tests are not part of the package**: import `hf_download` / `test_export` as sibling top-level modules through the `sys.path` entry each test module installs, never `evoxc_hf.tests.*`.
- **Build/verify**: `uv build --project evoxc_hf`, then inspect the wheel — it must contain only `evoxc_hf/**` plus the dist-info directory (including `licenses/LICENSE`).
- **Root-level follow-ups are outside this node** (the root `pyproject.toml` workspace stanza and removal of its `hf` extra, root `tests/test_packaging.py`, `../CONTEXT.md`, `../README.md`, `../CHANGELOG.md`, `../RELEASING.md`, `../.github/workflows/**`): escalate, never edit.
- Use `$TMPDIR` for checkpoints, never `/tmp` (`hf_download` defaults to `$TMPDIR/evoxc_hf_models`, overridable with `$EVOXC_HF_MODELS_DIR`); real-size checkpoints are opt-in with `EVOXC_HF_BIG=1`, which also satisfies `require_model` for locally cached checkpoints offline.
- Packed weights are opt-in and generic: `build(forward_fn(cfg, pack_weights=True), cfg, pack_weights=True)` + `pack_params(pack_plan(param_specs(cfg)), params)` (or `predict(..., pack_weights=True)`, which packs a canonical tree once on the host); pack on the **host first**, then `place_params`, then `run` — never re-concatenate per call, and never assume a fixed packed count (read `pack_plan(...).specs`). A tied config's pre-transposed head entry is declared automatically; pass `transposed_aliases={}` to `pack_plan` for a plan without it (which keeps the in-graph `transpose`).
- GPU work needs `uv run --isolated --extra iree …` (the base env has no iree, so `evoxc.devices()` shows only cpu); the ROCm device is `Device("rocm", 0)` (HAL `device_id` is 1-based), the adapter auto-detects the `gfxNNN` target, and the ROCm cases are GPU-guarded so the suite stays green on a CPU-only host (run that module with `PYTHONUNBUFFERED=1`).
- Real-size (`gpt2`/`distilgpt2`) work is memory-hungry (packed 124M ≈ 1.7 GB peak, canonical tree ≈ 475 MB): guard such cases on available memory and keep GPU runs short (this host's GPU is often shared).
