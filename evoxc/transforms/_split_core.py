"""Compatibility surface for the `split_by_inputs` structural graph split.

`split_by_inputs` is the FIRST structural (non-numeric) pass in this package:
it slices the graph DATA STRUCTURE into a `prepare` half (the fixed inputs'
contribution) and an `apply` half (the varying inputs on top of a prepared
boundary). It carries NO numerical authority — no derivatives, no batching,
no computation — so it lives in Python rather than in the Rust engine (see
`./CONTEXT.md`, "split_by_inputs").

THE PASS IS SPLIT ACROSS TWO MODULES, AND THEY ARE ONE PASS — not two
independent ones. Read them in this order:

- `_split_deps.py` — the READ half: what the graph says. `gates`,
  `dependence_masks`, `needed_set`, `select_boundary`, and `partition` /
  `op_id_partition` (the single validated derivation). Pure functions of the
  source module; nothing is cloned or built.
- `_split_build.py` — the CONSTRUCTION half: `prune_input_tree`,
  `rebuild_static_values`, `clone_module`, `build_prepare` / `build_apply`
  (the two `Graph`s), the per-half op-list selection and `split_graph` (the
  whole composition returning a `SplitPlan`).

This module is the original single-file surface kept as a thin re-export, so
`split.py`, `analyze.py`, `_split_types.py` and the test suite keep importing
`_split_core.<name>` unchanged. It defines nothing of its own — add new
machinery to the half that owns it and re-export it here.
"""

from __future__ import annotations

from evoxc.transforms._split_build import (
    _apply_ops,
    _boundary_value_types,
    _clone_op,
    _clone_region,
    _is_fixed_arity_node,
    _keep_predicate,
    _prepare_ops,
    _pruned_node_data,
    _symbolic_tree,
    _unused_varying_leaves,
    build_apply,
    build_prepare,
    clone_module,
    prune_input_tree,
    rebuild_static_values,
    split_graph,
)
from evoxc.transforms._split_deps import (
    _ALLOWED_EFFECTS,
    _CONSTANT_OP,
    _PYTREE_NODE_REGISTRY,
    _Partition,
    _entry_block,
    _gate_op,
    _iter_ops,
    _loc,
    _op_is_needed,
    _op_mask,
    _registered_base,
    _value_types,
    dependence_masks,
    gates,
    needed_set,
    op_id_partition,
    partition,
    select_boundary,
)

__all__ = [
    "gates",
    "dependence_masks",
    "needed_set",
    "select_boundary",
    "prune_input_tree",
    "rebuild_static_values",
    "clone_module",
    "build_prepare",
    "build_apply",
    "partition",
    "op_id_partition",
    "split_graph",
]
