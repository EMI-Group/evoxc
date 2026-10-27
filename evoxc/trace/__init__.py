"""evoxc.trace — tracing machinery: `@evoxc.defn`, `evoxc.trace`, `Graph`, control flow.

The tracer executes a Python function ONCE under an active `ir.Builder`;
tensor ops (in `evoxc.ops`) discover where to build via `current_builder()`.
Static Python values keep Python semantics and specialize the graph; runtime
tensor control flow is explicit (`cond` / `while_loop` / `scan`, traced into
IR regions). See `./CONTEXT.md` for the full contract.

Name shadowing (verified): `evoxc/__init__.py` re-exports the `trace` FUNCTION
under the attribute name `evoxc.trace`, so after `import evoxc` the ATTRIBUTE is
the function, NOT this submodule — and `import evoxc.trace as mod` therefore
binds the function too (it resolves the attribute, not `sys.modules`). To
reach this module's contents use `from evoxc.trace import current_builder`
(works — the import system consults `sys.modules` first) or
`import sys; sys.modules["evoxc.trace"]`. The tracing functions below are
re-exported at package level, so in practice you never need the module
object itself.
"""

from .builder import builder_stack, current_builder, with_builder
from .control_flow import cond, scan, while_loop
from .defn import Defn, defn
from .graph import Graph, StaticValue
from .trace import trace

__all__ = [
    "Defn",
    "Graph",
    "StaticValue",
    "builder_stack",
    "cond",
    "current_builder",
    "defn",
    "scan",
    "trace",
    "while_loop",
    "with_builder",
]
