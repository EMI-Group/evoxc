# evoxc/ir/op_defs — operation contracts and extensible registry

## Intent

This directory declares the canonical EvoXIR operation contracts and exposes the process-local operation-definition registry consumed by the Builder, verifier, and frontend.
Registration alone does not provide backend implementations or transform rules.

## API Surface

`__init__.py` defines frozen dataclasses `AttrSpec(name, type, default, description)` and `OpDef(name, category, description, arity, result_count, effect, attributes, shape_fn, regions, is_terminator)` plus the `ATTR_*` string tags.
`AttrSpec.required` uses identity against a private `_NO_DEFAULT` object, so an absent default and an explicit `None` default are distinct.
`ShapeInferenceFn` is a Python callable taking a tuple of `ValueType` objects and an attribute dict and returning result types.
`register_opdef` inserts a definition by name and returns that same object, `opdef` returns the live definition, `has_opdef` tests membership, and `op_names` and `all_opdefs` return sorted tuples.
Importing the registry imports every category module and executes its registration function.

## Constraints and Extension Semantics

The registry is an ordinary mutable module-global dict with no explicit locking, freeze/snapshot mechanism, unregister API, override API, schema validation, or definition-version identity.
Duplicate registration raises `ValueError` even for the same definition object, despite the registration docstring's misleading word "idempotent".
Unknown lookups raise `KeyError`.
The frozen dataclasses prevent normal field reassignment but do not recursively freeze arbitrary defaults, attribute contents, or callable state.
`ATTR_TYPE_NAMES` enumerates known tags but `AttrSpec` construction and `register_opdef` do not enforce membership or validate field values.
The schema allows arbitrary values only under explicitly declared `ATTR_ANY` keys, not arbitrary undeclared attribute names at Builder construction.
`ATTR_ANY` is documented as JSON-able but its Python representation is unrestricted at this layer.
Built-in uses include structured `result_specs`, block `static_args`, convolution padding, optional matrix-rank tolerance, flip/roll axes, and nan-to-num replacements.
`runtime_call` stores a string callback identifier, `external_call` stores a string kernel name, and `block_call` stores a string block name rather than executable code in their schemas.
Their `result_specs` attributes use `ATTR_ANY` for structured result-type data.
`runtime_call` and `external_call` have callback effect, whereas `block_call` has read effect and its description delegates actual effects to its declaration.

## Portability Boundary

The declarative fields are a plausible shared-language schema seam, but `shape_fn` is executable Python rather than a declarative inference program or stable hook identifier.
Most built-in hooks are imported from `../inference.py`; the zero-result return hook is defined locally in `control.py`.
A native core must preserve custom Python-hook support or explicitly constrain which definitions it can process rather than assuming every registered definition has native semantics.
Cross-language representations need explicit encodings for required-versus-null defaults, scalar-versus-interval arities, unbounded interval endpoints, unknown result counts, dtype names, symbolic shapes, ndarray constants, and structured `ATTR_ANY` payloads.
Builder normalization and verification behavior live in the parent IR node, not in these dataclasses or the registry.
`OpDef` contains no batching, JVP, VJP, lowering, or execution implementation fields, so registry ownership alone does not move transform or backend semantics into a native core.
Built-in callback identity is not one-to-one with op identity: every binary elementwise definition, including divide and logical ops, shares `infer_elementwise_binary`, most unary definitions share `infer_elementwise_unary`, and cumulative sum uses `infer_identity`.
A native transform must preserve the existing IR-level callback contract rather than infer frontend dtype behavior or derivative/batching rules from categories, descriptions, or callback identity.

## Routing Table

| Path | Area |
|---|---|
| `./__init__.py` | Registry, AttrSpec/OpDef, shape-hook type, attribute tags |
| `./elementwise.py` | Elementwise/comparison definitions and generic inference hooks |
| `./structure.py` | Shape/indexing and structural operation definitions |
| `./reduction.py` | Reduction/arg-reduction and cumulative operation definitions |
| `./sorting.py` | Sort and argsort definitions |
| `./linalg.py` | Dot, convolution, matrix and factorization definitions |
| `./control.py` | Constants, control flow, named callback/block/kernel calls, return terminator |
| `./collective.py` | Communication definitions and rank/world-size reads |
| `./sparse.py` | Sparse operation definitions |
| `./random.py` | Algorithm names/key types and functional random operation definitions |
