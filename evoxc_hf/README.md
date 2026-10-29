**`evoxc_hf`** — load a HuggingFace checkpoint and run its whole forward pass as **ONE compiled evoxc graph**: `trace → lower → compile → load → run`, with every weight an ordinary graph input and nothing hidden.

`evoxc_hf` sits **above** [`evoxc`](https://github.com/EMI-Group/evoxc): evoxc's non-goals exclude NN layers, loaders and model zoos, so model-level code lives here and only *depends on* evoxc's public API. It is a separate, independently publishable distribution (`evoxc-hf`, imported as `evoxc_hf`).

```python
import evoxc_hf

loaded = evoxc_hf.load_hf("path/to/checkpoint")          # config + host params
logits = evoxc_hf.predict(loaded, input_ids, seq_len=32)  # numpy backend, one call
oracle = evoxc_hf.reference_logits(loaded.config, loaded.params, input_ids)
```

---

## Install

```bash
pip install evoxc-hf          # numpy + evoxc only
pip install evoxc-hf[hf]      # + torch/transformers (reference oracles, never at runtime)
pip install evoxc[iree]       # compiled path (LLVM-CPU / CUDA / ROCm / Vulkan)
```

Nothing in this package imports `torch`, `transformers`, `safetensors` or `iree` at module scope: the checkpoint readers (`config.json` + `safetensors` / `pytorch_model.bin`) are pure `numpy` + `stdlib`, and the `hf` extra exists only to validate and benchmark the graphs against the real HuggingFace implementation.

## The explicit pipeline, with weights as inputs

```python
import evoxc, evoxc_hf

loaded = evoxc_hf.load_hf("path/to/checkpoint")
fwd = evoxc_hf.forward_fn(loaded.config, seq_len=32)   # a pure @evoxc.defn
exe = evoxc_hf.build(fwd, loaded.config)               # trace → lower → compile → load
out = evoxc_hf.run(exe, loaded.params, input_ids)      # an evoxc.Tensor

exe = evoxc_hf.build(fwd, loaded.config, backend="iree")   # the compiled path
```

* **The compute function captures no parameters.** `forward_fn(config, seq_len=..., batch_size=..., pack_weights=...)` is the whole model pass `(params, input_ids) -> logits`, with every weight a graph **input** (`param_specs`) — never an `evoxc.constant` payload inlined into the program.
* **Static specialisation.** One config + one `seq_len`/`batch_size`/packing = one traced graph. `forward_fn` is memoised, so identical `build` calls return the identical `Executable`; a mismatched length is an explicit error, never a silent recompile.
* **Three layers, never blurred.** Loading (`load_config`/`load_params`/`load_hf` → `LoadedModel`, plain immutable data), the compute function (`forward_fn`/`param_specs`/`reference_logits`), staging (`build`/`run`/`predict` + the explicit `place_params`/`place_ids`). No loaded-model class, no hidden state.

## Devices are explicit

```python
device = evoxc.Device("cuda", 0)
exe = evoxc_hf.build(fwd, loaded.config, backend="iree", device=device)
out = evoxc_hf.run(
    exe,
    evoxc_hf.place_params(loaded.params, device),   # one Tensor.to per leaf
    evoxc_hf.place_ids(input_ids, device),
)
```

Place the weights **once** and keep the placed tree: feeding each run's device outputs back as the next run's inputs stages nothing per call and never round-trips to the host. A non-CPU executable rejects host inputs at the run boundary by design; `predict` is the one host-facing shorthand (it makes the single explicit `Tensor.to(Device("cpu", 0))` on the output).

## Packed weights (opt-in)

```python
fwd = evoxc_hf.forward_fn(loaded.config, seq_len=32, pack_weights=True)
plan = evoxc_hf.pack_plan(evoxc_hf.param_specs(loaded.config))
exe = evoxc_hf.build(fwd, loaded.config, pack_weights=True, backend="iree")
packed = evoxc_hf.pack_params(plan, loaded.params)     # pack on the host, then place
```

The canonical spec tree is bucketed by `(dtype, rank, shape[1:])` and concatenated along axis 0, and the graph unpacks with static `evoxc.slice` — so a 197-input model becomes a 5-input graph (124M `gpt2`: 196 keys → 4 weight inputs). It is generic over any `param_specs` tree, bit-identical to the canonical path, and mainly buys IREE's Python runtime headroom on LLVM-CPU.

## Supported models

| family | HF `model_type` | notes |
|---|---|---|
| `llama` | `llama`, `mistral`, `qwen2`, `qwen3` | RoPE + causal mask in-graph; Qwen3 adds per-head QK-Norm (`head_dim`-sized, applied after `split_heads`, before RoPE) |
| `gpt2` | `gpt2` | learned position embeddings, fused-QKV split at load, tied embeddings |

Parity is pinned against an independent pure-numpy oracle (`reference.py`) and, when `torch`+`transformers` are installed, against the real HuggingFace implementation.

## Known limits (all explicit errors, never silent fallbacks)

* **Static shapes only** — one graph per `(config, seq_len, batch_size)`; `seq_len > max_position_embeddings` is an `HFConfigError`.
* **No KV cache, incremental decode, sampling or training** — every call re-runs the whole graph.
* **Exact-erf GELU is numpy-backend-only** (`evoxc.erf` is a StableHLO export deferral); a compiled backend refuses it up front instead of substituting `tanh`-GELU.
* **The llama family is bias-free** (`attention_bias`/`mlp_bias` models are refused), and `rope_scaling` is unsupported.
* **Qwen3 is full-attention only** — any sliding-window request raises `HFUnsupportedError`.

## Development

This package uses a `src` layout (`evoxc_hf/src/evoxc_hf/`) inside the evoxc uv workspace, and its test suite stays colocated at `evoxc_hf/tests/`:

```bash
PYTHONPATH=evoxc_hf/src uv run --with pytest python -m pytest evoxc_hf/tests
```

## License

Apache-2.0 — see `LICENSE` (a tracked relative symlink to the repository's license).
