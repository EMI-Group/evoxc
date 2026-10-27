# tests/trace — tracing / defn / control-flow test suite

## Intent

pytest suite validating the tracing machinery of `evoxc` (sibling — see
`../../evoxc/trace/CONTEXT.md` for the binding contract being tested):
`@evoxc.defn`, `evoxc.trace`, the `Graph` type, the active-builder context, and
runtime tensor control flow (`cond` / `while_loop` / `scan`). Tests assert
the CONTRACT — explicitness, static-value specialization, region-based IR,
structured pytree I/O — not incidental implementation behavior.

## Structure (flat — one file per module/feature)

| File | Covers |
|---|---|
| `test_defn.py` | `@evoxc.defn` decorator: Defn/marker, options, idempotence, metadata, always-raises `__call__` |
| `test_trace.py` | `evoxc.trace` core: symbolic dims, structured inputs, static specialization, closure capture, zero-IO, errors, fresh graphs |
| `test_graph.py` | `Graph` attrs, `print()`, `verify()`, `flatten_inputs`/`validate_inputs`, `unflatten_outputs`, save/load round-trip, `signature_info` |
| `test_builder_context.py` | `current_builder()` / `with_builder` (LIFO stack), ops outside trace, region routing |
| `test_cond.py` | `evoxc.cond`: runtime branch selection, `if`-op IR structure, pred validation, branch unification errors |
| `test_while_loop.py` | `evoxc.while_loop`: iteration counts, structured state, `while`-op IR, carried-type/static-leaf errors |
| `test_scan.py` | `evoxc.scan`: cumsum/max vs numpy, length override, structured xs/init, desugared IR, v1 static-length errors |
| `test_static_snapshot.py` | static dtype/Enum/slice/bool + dataclass-config specialization, trace-time snapshotting, run-time static validation |
| `conftest.py` | shared fixtures (below) |

## Shared fixtures (conftest.py)

- `run_graph(graph, *np_arrays)` — explicit staging pipeline
  `lower → compile → load(device="cpu") → run` (NOTE: `evoxc.build` accepts a
  defn/callable, NOT an already-traced Graph — this fixture wraps the
  explicit stages).
- `as_numpy(result)` — recursively converts `evoxc.Tensor` leaves of a
  structured result to numpy arrays for assertions.

## Constraints

- CPU only; numpy backend is the correctness reference. Small shapes, each
  test file < 0.5s in practice (< ~500 lines).
- Only numpy arrays (not numpy scalars) are accepted as run-time inputs —
  pass 0-d `np.array(v, dtype)` for scalar tensors.
- Inside traced functions, constants MUST be `evoxc.constant(evoxc.tensor(...))`
  or `evoxc.constant(evoxc.zeros(...))` — never capture a concrete `Tensor`.
- `evoxc.trace` is the FUNCTION (shadows the submodule as attribute); use
  `from evoxc.trace import ...` for module pieces (`StaticValue`,
  `current_builder`, `with_builder`, `Graph`).

## Notes for agents

- **BUG(evoxc) protocol**: tests that expose real contract violations in `evoxc`
  are KEPT FAILING with a `# BUG(evoxc): <desc>` comment + minimal repro. Never
  fix `evoxc` from this directory and never weaken such tests — the package
  root owns fixes. The suite is currently expected to be fully green; any
  failure is a regression in either evoxc or the tests.
- **Tree leaf markers**: trace-time `input_specs`/`output_tree` use private
  `_TensorSpecLeaf`/`_SymbolicLeaf` marker types by design (dataclass leaves
  would be recursed into). Compare tree SKELETONS (type/node_data/children,
  ignoring leaf types) — never full TreeSpec equality against live objects.
- Nested `evoxc.trace` inside a trace is unspecified by the contract — no
  assertion is made for it (see test_builder_context.py docstring).

## Test strategy

Run `python3 -m pytest -q tests/trace` from the repo root (root conftest.py
puts the repo on sys.path). Spec-compliance cross-cutting tests
(staging explicitness, bind-as-sugar, vmap≡vectorize) live in
`tests/test_spec_compliance.py` (owned by the package root) — do not
duplicate them here.
