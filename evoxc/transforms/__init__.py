"""evoxc.transforms — graph-to-graph transformations.

Frontend transforms over traced graphs: `vectorize` (the batching primitive),
`vmap` (transparent function-side sugar over vectorize), `batch_functions`
(merge N same-signature programs onto one graph with a static function axis),
and automatic differentiation (`grad` / `jvp` / `vjp`). Every transform produces
an ordinary `Graph` of ordinary ops — backends need no transform-specific
runtime support. No execution happens here: transforms never import
backends/pipeline.

One pass is STRUCTURAL rather than numeric: `split_by_inputs` slices a graph
into a `prepare` / `apply` pair at its input structure (which inputs stay fixed
across calls and which vary per call), and `analyze_inputs` reports the
per-input facts that split is derived from. It slices the graph data structure
and carries no numerical authority, so it is neither a native-engine transform
nor a numerical one.

Read `./CONTEXT.md` for the binding design: rule-call signatures, the
vmap⇔vectorize equivalence contract, AD semantics, `split_by_inputs`'s
correctness law, and v1 scope.
"""

from evoxc.transforms.batching import (
    batching_rules,
    register_batched_aux_remap,
    register_batching_rule,
)
from evoxc.transforms.autodiff import (
    jvp_rules,
    vjp_rules,
    register_jvp_rule,
    register_vjp_rule,
)
from evoxc.transforms.vectorize import vectorize
from evoxc.transforms.vmap import vmap
from evoxc.transforms.batch_functions import (
    BatchReport,
    SwitchRecord,
    batch_functions,
)
from evoxc.transforms.grad import grad
from evoxc.transforms.jvp import jvp
from evoxc.transforms.vjp import vjp
from evoxc.transforms.split import split_by_inputs
from evoxc.transforms.analyze import analyze_inputs
from evoxc.transforms._split_types import (
    BoundaryValue,
    InputAnalysis,
    InputInfo,
    SplitCallable,
    SplitPlan,
)
from evoxc.transforms._wrappers import TransformCallable

# Registers the builtin batching/JVP/VJP rules for the standard op set.
from evoxc.transforms import rules as _rules  # noqa: E402,F401

__all__ = [
    "vectorize",
    "vmap",
    "batch_functions",
    "BatchReport",
    "SwitchRecord",
    "grad",
    "jvp",
    "vjp",
    "split_by_inputs",
    "analyze_inputs",
    "SplitPlan",
    "BoundaryValue",
    "InputAnalysis",
    "InputInfo",
    "SplitCallable",
    "TransformCallable",
    "batching_rules",
    "register_batching_rule",
    "register_batched_aux_remap",
    "jvp_rules",
    "register_jvp_rule",
    "vjp_rules",
    "register_vjp_rule",
]
