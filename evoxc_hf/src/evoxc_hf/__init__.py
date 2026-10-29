"""``evoxc_hf`` — load a HuggingFace checkpoint and run its forward as ONE evoxc graph.

``evoxc_hf`` sits **above** :mod:`evoxc` (a separate top-level package, never a
submodule): evoxc core's non-goals exclude NN layers, loaders and model zoos, so
model-level code lives here and only *depends on* the public evoxc API.

The API is **functional**, mirroring evoxc's own value model, with a hard split
between the two halves of the problem:

* **loading** (:mod:`evoxc_hf.load`) answers *what is this checkpoint?* — a static
  :class:`ModelConfig` (one config = one graph, see ``config.digest``) plus a
  canonical ``dict[str, np.ndarray]`` of host params (:class:`LoadedModel` is
  plain immutable data — no methods, no ``seq_len``);
* **the compute function** (:mod:`evoxc_hf.program`) answers *what does it
  compute?* — :func:`forward_fn` is a pure ``@evoxc.defn`` forward
  ``(params, input_ids) -> logits`` capturing **no** parameters, only the static
  specialisation ``(family, config, seq_len, batch_size, pack_weights)``;
* **staging** (:mod:`evoxc_hf.pipeline`) answers *where and how does it run?* —
  :func:`build` (``evoxc.build_cached``) and :func:`run` (``evoxc.run``), with
  :func:`predict` as the one documented host-facing shorthand.

The philosophy is evoxc's, applied to whole models — **make computation
explicit**:

* a checkpoint's weights are ordinary *graph inputs* (:func:`param_specs`
  ``TensorSpec`` trees), never ``evoxc.constant`` payloads, so nothing is silently
  inlined into the program;
* config values are static specialisation values: one config + one
  ``seq_len``/``batch_size``/packing = one traced graph, memoised so two
  identical :func:`build` calls return the identical ``Executable``;
* the pipeline is the explicit evoxc one — ``trace → lower → compile → load →
  run`` — reached through :func:`build` and :func:`run`, with no hidden tracing,
  compiling, weight embedding or device movement;
* moving weights *and* token ids to a device is an explicit, per-leaf
  ``Tensor.to(device)`` (:func:`place_params` / :func:`place_ids`) — a non-cpu
  executable rejects host inputs by design;
* every failure is an explicit, actionable error deriving from
  :class:`evoxc.core.EVOXCError` — never a silent fallback, dtype swap or positional
  guess.

Typical use::

    import evoxc, evoxc_hf

    loaded = evoxc_hf.load_hf("path/to/checkpoint")      # config + host params
    logits = evoxc_hf.predict(loaded, ids, seq_len=32)   # numpy backend, one call
    oracle = evoxc_hf.reference_logits(loaded.config, loaded.params, ids)

    # ...or stage the graph yourself:
    fwd = evoxc_hf.forward_fn(loaded.config, seq_len=32)     # a pure @evoxc.defn
    exe = evoxc_hf.build(fwd, loaded.config)                 # numpy interpreter
    exe = evoxc_hf.build(fwd, loaded.config, backend="iree") # compiled path
    logits = evoxc_hf.run(exe, loaded.params, ids)           # an evoxc.Tensor

    device = evoxc.core.Device("cuda", 0)
    exe = evoxc_hf.build(fwd, loaded.config, backend="iree", device=device)
    out = evoxc_hf.run(
        exe,
        evoxc_hf.place_params(loaded.params, device),
        evoxc_hf.place_ids(ids, device),
    )

Import-time constraint: ``import evoxc_hf`` requires only ``numpy`` (plus ``evoxc``).
torch / transformers / safetensors / iree are **never** imported at module scope.
"""

from __future__ import annotations

# pyproject.toml's `[project].version` is the SINGLE source of truth for the
# version — this package only reads it back from the installed distribution
# metadata, so the two can never drift. NOTE: the DISTRIBUTION name is
# hyphenated (`evoxc-hf`, PyPI-normalised to `evoxc_hf`) while the IMPORT name
# below is not.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
from evoxc_hf import (
    config,
    families,
    load,
    ops,
    pack,
    pipeline,
    program,
    reference,
    weights,
)
from evoxc_hf.config import ModelConfig
from evoxc_hf.errors import (
    HFConfigError,
    HFError,
    HFUnsupportedError,
    HFWeightError,
)
from evoxc_hf.families import FAMILIES, FamilySpec, family_for_model_type, get_family
from evoxc_hf.load import (
    LoadedModel,
    load_config,
    load_hf,
    load_params,
    validate_params,
)
from evoxc_hf.pack import PackPlan, pack_params, pack_plan
from evoxc_hf.pipeline import build, place_ids, place_params, predict, run
from evoxc_hf.program import forward_fn, param_specs, reference_logits

try:
    __version__ = _distribution_version("evoxc-hf")
except _PackageNotFoundError:  # a bare source checkout: no installed distribution
    __version__ = "0.0.0+unknown"

__all__ = [
    # -- loading (checkpoint -> static config + host params) ------------------
    "LoadedModel",
    "ModelConfig",
    "load_config",
    "load_hf",
    "load_params",
    "validate_params",
    # -- the compute function ------------------------------------------------
    "forward_fn",
    "param_specs",
    "reference_logits",
    # -- staging (build / run / predict / explicit placement) ----------------
    "build",
    "place_ids",
    "place_params",
    "predict",
    "run",
    # -- packed weights (opt-in) --------------------------------------------
    "PackPlan",
    "pack_params",
    "pack_plan",
    # -- family registry ----------------------------------------------------
    "FAMILIES",
    "FamilySpec",
    "family_for_model_type",
    "get_family",
    # -- errors -------------------------------------------------------------
    "HFConfigError",
    "HFError",
    "HFUnsupportedError",
    "HFWeightError",
    # -- submodules ---------------------------------------------------------
    "config",
    "families",
    "load",
    "ops",
    "pack",
    "pipeline",
    "program",
    "reference",
    "weights",
    # -- metadata -----------------------------------------------------------
    "__version__",
]
