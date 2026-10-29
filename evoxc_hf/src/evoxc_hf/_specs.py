"""Canonical param-spec trees, plus the packing **orientation** they imply.

A ``tie_word_embeddings=True`` model has **one** ``embed.weight``
``(vocab, hidden)`` graph input serving two consumers:

* the token gather, which indexes axis 0 and needs ``(vocab, hidden)``;
* the output projection, which needs the ``(in_features, out_features)`` layout
  every :func:`evoxc_hf.models._common.linear` consumes — i.e. ``(hidden, vocab)``.

Without a declaration the packed graph would have to transpose the whole
embedding table on EVERY forward call (154 MB for the 124M ``gpt2``, 622 MB for
``qwen3_0_6b``, 1.2 GB for ``qwen3_1_7b``) — the only per-call weight-sized
materialisation in the graph. Declaring the alias lets
:func:`evoxc_hf.pack.pack_plan` store ``np.ascontiguousarray(embed.T)`` **once**,
as an ordinary packed weight input that the graph slices statically, so the
traced program contains no layout op applied to a weight-sized operand at all.
The cost is one extra weight-sized host + device buffer (the gather keeps the
original), which ``./evoxc_hf/CONTEXT.md`` documents.

Why the declaration is *inferred from the spec tree* rather than carried by a
``dict`` subclass: a family's ``param_specs(cfg)`` return value is handed
straight to ``evoxc.trace``/``evoxc.build``/``evoxc.run``, and evoxc's pytree structure
matching keys on the **exact** container type — a ``dict`` subclass would trace
a tree that a plain ``dict`` of params then fails to match. The rule below is
static, deterministic and decided from the spec tree alone (the same metadata
``param_specs`` used to build it): a tree with an ``embed.weight`` and **no**
``lm_head.weight`` is a tied-embedding tree, so its head is the transposed
embedding. A tree that carries a real ``lm_head.weight`` (an untied config)
declares nothing, and :func:`evoxc_hf.pack.pack_plan`'s explicit
``transposed_aliases=`` argument overrides the inference entirely.

Import-time constraint: stdlib only — no ``evoxc``, ``numpy`` or sibling
``evoxc_hf`` import — so every layer of the package may import this module.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "EMBED_WEIGHT",
    "HEAD_WEIGHT",
    "TRANSPOSED_ALIASES",
    "transposed_aliases",
]

#: The canonical embedding key (the tied head's source, ``(vocab, hidden)``).
EMBED_WEIGHT: str = "embed.weight"
#: The canonical output-projection key (``(hidden, vocab)``), derived when tied.
HEAD_WEIGHT: str = "lm_head.weight"
#: derived key -> the canonical key its data is stored transposed from.
TRANSPOSED_ALIASES: Mapping[str, str] = {HEAD_WEIGHT: EMBED_WEIGHT}


def transposed_aliases(specs: Mapping[str, object]) -> dict[str, str]:
    """The derived (pre-transposed) pack entries a canonical spec tree declares.

    Args:
        specs: a family's canonical ``param_specs(cfg)`` tree
            (``{name: evoxc.TensorSpec}``).

    Returns:
        dict[str, str]: ``{derived_key: source_key}`` — ``{"lm_head.weight":
        "embed.weight"}`` for a tied-embedding tree (it has the embedding but no
        head of its own), else ``{}``. Static and deterministic: the same spec
        tree always declares the same aliases, so two plans built from equal
        trees are equal.
    """
    if EMBED_WEIGHT in specs and HEAD_WEIGHT not in specs:
        return dict(TRANSPOSED_ALIASES)
    return {}
