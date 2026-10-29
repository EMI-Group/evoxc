# evoxc-hf

Load a real HuggingFace checkpoint — `config.json` plus its safetensors or
`pytorch_model.bin` weights — and run the **whole model forward as ONE compiled
evoxc graph**, through the explicit `trace → lower → compile → load → run`
pipeline. Same evoxc semantics applied to a whole model: no hidden tracing, no
hidden weight embedding, no hidden device movement.

`evoxc-hf` is a separate distribution built on top of
[`evoxc`](https://github.com/EMI-Group/evoxc). Every weight is an ordinary graph
input, not an `evoxc.constant` payload; config values are static specialisation
values (one config = one graph); and the program is a first-class object you can
inspect, compile for several backends, and reuse.

## Install

```bash
pip install evoxc-hf
pip install "evoxc-hf[hf]"     # + torch / transformers — reference oracles only
```

The `hf` extra installs the reference oracles used to validate parity against the
real HuggingFace implementation. They are **never runtime requirements**: the
checkpoint readers are numpy + stdlib (`struct`, `json`, `zipfile`/`pickle`), and
nothing from torch/transformers is imported to load or run a model.

The compiled path is an **evoxc** extra, not this package's — install
`"evoxc[iree]"` (or `"evoxc[tvm]"`, or a user-provided PJRT plugin for xla).

## Quickstart

```python
import numpy as np
import evoxc_hf

loaded = evoxc_hf.load_hf("path/to/checkpoint")   # config + host params, nothing compiled
ids = np.zeros((1, 32), dtype=np.int32)           # static (batch_size, seq_len)

logits = evoxc_hf.predict(loaded, ids, seq_len=32)   # np.ndarray; numpy backend by default
```

Stage the graph yourself when you want the pipeline objects:

```python
import evoxc, evoxc_hf

fwd = evoxc_hf.forward_fn(loaded.config, seq_len=32)   # a pure @evoxc.defn, weights are inputs
specs = evoxc_hf.param_specs(loaded.config)            # dict[str, evoxc.TensorSpec]

exe = evoxc_hf.build(fwd, loaded.config)               # trace -> lower -> compile -> load
logits = evoxc_hf.run(exe, loaded.params, ids)         # evoxc.Tensor, on the executable's device
```

Compile it and run device-resident, where placement is explicit per leaf:

```python
device = evoxc.core.Device("cuda", 0)
exe = evoxc_hf.build(fwd, loaded.config, backend="iree", device=device)

out = evoxc_hf.run(
    exe,
    evoxc_hf.place_params(loaded.params, device),      # one explicit Tensor.to per weight
    evoxc_hf.place_ids(ids, device),
)                                                      # stays on the device — no host round-trip
```

## The API in three layers

The API is **functional** — module functions taking their inputs as arguments.
There is no model class with behaviour: `LoadedModel` is a plain immutable record.

| Layer | Answers | Surface |
|---|---|---|
| loading | *what is this checkpoint?* | `load_config` → `ModelConfig`, `load_params` → `dict[str, np.ndarray]`, `load_hf` → `LoadedModel` |
| the compute function | *what does it compute?* | `forward_fn` → a memoised `@evoxc.defn` whose weights are graph **inputs** (`param_specs`); `reference_logits` is the numpy oracle |
| staging | *where and how does it run?* | `build` (`trace → lower → compile → load`), `run`, `predict`, `place_params` / `place_ids` |

`forward_fn` captures no parameters, so the graph depends only on the static
`(config, seq_len, batch_size)` specialisation — two identical `build` calls
return the same `Executable`, and one graph serves every checkpoint with the same
config. `predict` is the single host-facing shorthand: build → run → one explicit
`Tensor.to(Device("cpu", 0))` → `numpy`.

## Supported models

- **`llama`** — HF `llama`, `mistral`, `qwen2`, `qwen3` (RMSNorm, RoPE, GQA,
  SwiGLU); the Qwen3 case adds per-head QK-Norm.
- **`gpt2`** — HF `gpt2`, `distilgpt2` (LayerNorm, learned positions, `gelu_new`,
  tied head).

Parity is pinned against an **independent pure-numpy oracle** (`reference_logits`,
a deliberately different formulation) and, when `torch` + `transformers` are
installed, against the real HuggingFace logits.

## Limitations

- **Static shapes only**: one graph per `(config, seq_len, batch_size)`. A
  `seq_len` above `max_position_embeddings` is an explicit `HFConfigError`, never
  a silent recompile.
- **The llama family is bias-free** in v1 — any `*.bias` checkpoint key is an
  explicit `HFUnsupportedError`.
- **No KV cache, no incremental decode, no sampling, no training**: every call
  re-runs the whole forward over `seq_len` tokens.
- **Exact-`erf` GELU is numpy-backend-only** (`hidden_act: "gelu"`/`"erf"`);
  compiler backends raise explicitly rather than substituting tanh.
- **`rope_scaling` is unsupported**, and Qwen3 is full-attention only — sliding
  window requests raise rather than silently diverging from HF.

## Relationship to evoxc

`evoxc-hf` is a **separate distribution** from `evoxc`: it is never part of the
`evoxc` wheel or sdist, and the dependency is strictly one-way — `evoxc_hf`
imports only the public `evoxc` API, and `evoxc` never imports `evoxc_hf`.
Requires Python ≥ 3.11, with `numpy` + `evoxc` as its only required dependencies.

## License

Apache-2.0. Source: <https://github.com/EMI-Group/evoxc/tree/main/evoxc_hf> ·
repository <https://github.com/EMI-Group/evoxc>.
