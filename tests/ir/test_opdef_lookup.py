"""Op-definition registry lookup and the cached-lookup-callable speed hook.

Contract pinned here (the ``./evoxc/`` package is READ-ONLY from the test
suite — the contract itself lives in ``../../evoxc/ir/CONTEXT.md``):

* ``evoxc.ir.opdef(name)`` is the registry lookup: an unregistered name raises
  ``KeyError`` (the registry is a plain ``dict``), and a failed lookup inserts
  nothing.
* ``evoxc.ir.has_opdef(name)`` answers registration without raising, while
  ``op_names()`` / ``all_opdefs()`` are the sorted enumeration surfaces and
  always stay consistent with ``opdef``.
* ``evoxc.ir.register_opdef(def)`` RETURNS the def it registered and rejects a
  duplicate name with ``ValueError`` (``"op '<name>' is already registered"``);
  the first registration wins.
* ``Op.opdef`` consults the registry on EVERY access: it returns the IDENTICAL
  registered object across accesses (``is``), and a name that raised
  ``KeyError`` BEFORE its ``OpDef`` was registered resolves correctly
  afterwards — there is NO negative memo and no result memo at all.
* The only thing ``evoxc.ir.op`` caches is the imported lookup CALLABLE
  (``_OPDEF_LOOKUP``) — the speed hook — never a ``name -> OpDef`` table.
* Registering an extra ``OpDef`` disturbs neither ordinary tracing nor
  ``verify()``.

The op-def registry is process-global, so every test-only ``OpDef`` here is
registered through the local ``throwaway_opdef`` fixture, which restores
``evoxc.ir.op_defs._REGISTRY`` to its exact pre-fixture state in a ``finally``
(sibling modules such as ``test_registry.py`` pin the canonical op set and
would break if a test-only name leaked).
"""

from collections.abc import Mapping

import pytest

import evoxc
from evoxc import ir
from evoxc.ir import op_defs as _op_defs  # private: registry cleanup (see fixture)

# Test-only name — never part of the canonical op set, so `test_registry.py`
# keeps its exact-set assertions.
THROWAWAY_NAME = "_test_lookup_probe"
UNREGISTERED_NAME = "_test_definitely_not_an_op"


# ---------------------------------------------------------------------------
# Helpers — local to this file (test modules never import each other's helpers)
# ---------------------------------------------------------------------------


def _make_throwaway_opdef() -> ir.OpDef:
    """A valid minimal ``OpDef`` under the test-only name."""
    return ir.OpDef(
        name=THROWAWAY_NAME,
        category="elementwise",
        description="test-only opdef for registry lookup tests",
        arity=1,
        result_count=1,
        effect=ir.EFFECT_PURE,
    )


@pytest.fixture
def throwaway_opdef():
    """Register a test-only ``OpDef``; restore the registry to its prior state."""
    prior = _op_defs._REGISTRY.pop(THROWAWAY_NAME, None)
    opdef = _make_throwaway_opdef()
    try:
        ir.register_opdef(opdef)
        yield opdef
    finally:
        _op_defs._REGISTRY.pop(THROWAWAY_NAME, None)
        if prior is not None:
            _op_defs._REGISTRY[THROWAWAY_NAME] = prior


def _trace_sin_graph():
    """Trace a tiny real graph; its entry block holds ``sin`` + ``return``."""
    return evoxc.trace(lambda x: evoxc.sin(x), evoxc.TensorSpec((3,), evoxc.float32))


# ---------------------------------------------------------------------------
# 1. Lookup of an unregistered name
# ---------------------------------------------------------------------------


def test_unknown_name_raises_keyerror_and_has_opdef_is_false():
    with pytest.raises(KeyError) as excinfo:
        ir.opdef(UNREGISTERED_NAME)
    assert UNREGISTERED_NAME in str(excinfo.value)
    assert ir.has_opdef(UNREGISTERED_NAME) is False


def test_failed_lookup_leaves_the_registry_untouched():
    names_before = ir.op_names()
    defs_before = ir.all_opdefs()
    with pytest.raises(KeyError):
        ir.opdef(UNREGISTERED_NAME)
    assert ir.op_names() == names_before
    assert ir.all_opdefs() == defs_before
    assert ir.has_opdef(UNREGISTERED_NAME) is False


# ---------------------------------------------------------------------------
# 2. Registration, identity, duplicate rejection, reversibility
# ---------------------------------------------------------------------------


def test_register_opdef_returns_its_argument():
    opdef = _make_throwaway_opdef()
    try:
        assert ir.register_opdef(opdef) is opdef
        assert ir.opdef(THROWAWAY_NAME) is opdef
    finally:
        _op_defs._REGISTRY.pop(THROWAWAY_NAME, None)


def test_registered_name_resolves_through_every_lookup_surface(throwaway_opdef):
    assert ir.has_opdef(THROWAWAY_NAME) is True
    assert ir.opdef(THROWAWAY_NAME) is throwaway_opdef
    assert THROWAWAY_NAME in ir.op_names()
    assert throwaway_opdef in ir.all_opdefs()


def test_repeated_lookup_returns_the_identical_object(throwaway_opdef):
    first = ir.opdef(THROWAWAY_NAME)
    assert first is throwaway_opdef
    assert ir.opdef(THROWAWAY_NAME) is first
    assert ir.opdef(THROWAWAY_NAME) is first


def test_duplicate_registration_raises_value_error(throwaway_opdef):
    duplicate = _make_throwaway_opdef()
    with pytest.raises(ValueError, match="is already registered"):
        ir.register_opdef(duplicate)
    # The first registration is untouched by the rejected duplicate.
    assert ir.has_opdef(THROWAWAY_NAME) is True
    assert ir.opdef(THROWAWAY_NAME) is throwaway_opdef


def test_registration_is_reversible_by_a_registry_snapshot():
    names_before = ir.op_names()
    defs_before = ir.all_opdefs()
    ir.register_opdef(_make_throwaway_opdef())
    try:
        assert ir.has_opdef(THROWAWAY_NAME) is True
    finally:
        _op_defs._REGISTRY.pop(THROWAWAY_NAME)
    assert ir.has_opdef(THROWAWAY_NAME) is False
    assert ir.op_names() == names_before
    assert ir.all_opdefs() == defs_before


# ---------------------------------------------------------------------------
# 3. ``Op.opdef`` identity + the cached-lookup-callable speed hook
# ---------------------------------------------------------------------------


def test_op_opdef_is_the_registered_object_for_every_op_of_a_traced_graph():
    graph = _trace_sin_graph()
    ops = graph.module.main.entry_block.ops
    assert [op.name for op in ops] == ["sin", "return"]
    for op in ops:
        assert op.opdef is ir.opdef(op.name)
        assert op.opdef is op.opdef
        assert op.opdef is op.opdef


def test_op_caches_only_the_lookup_callable_never_a_lookup_result():
    # Touch the hot path first so the callable cache is populated.
    for op in _trace_sin_graph().module.main.entry_block.ops:
        op.opdef
    cached = ir.op._OPDEF_LOOKUP
    assert cached is not None
    assert cached is _op_defs.opdef  # the registry function itself
    assert callable(cached)
    assert not isinstance(cached, Mapping)  # never a name -> OpDef table
    assert ir.op._load_opdef_lookup() is cached


# ---------------------------------------------------------------------------
# 4. No negative memo: an op looked up BEFORE registration resolves afterwards
# ---------------------------------------------------------------------------


def test_no_negative_memo_for_a_name_looked_up_before_registration():
    op = ir.Op(name=THROWAWAY_NAME, id=0)
    with pytest.raises(KeyError):
        op.opdef
    with pytest.raises(KeyError):
        op.opdef  # a failure is never cached either
    assert ir.has_opdef(THROWAWAY_NAME) is False

    opdef = _make_throwaway_opdef()
    try:
        ir.register_opdef(opdef)
        # The SAME Op instance — nothing is memoized on the Op itself.
        assert op.opdef is opdef
        assert op.opdef is op.opdef
        assert ir.opdef(THROWAWAY_NAME) is opdef
    finally:
        _op_defs._REGISTRY.pop(THROWAWAY_NAME)

    # Post-restore the registry is consulted again: no result was pinned.
    with pytest.raises(KeyError):
        op.opdef


# ---------------------------------------------------------------------------
# 5. An extra registry entry disturbs neither tracing nor verification
# ---------------------------------------------------------------------------


def test_tracing_and_verification_are_unaffected_by_an_extra_opdef(throwaway_opdef):
    graph = _trace_sin_graph()
    assert graph.verify() is None
    for op in graph.module.main.entry_block.ops:
        assert op.opdef is ir.opdef(op.name)
    assert ir.opdef(THROWAWAY_NAME) is throwaway_opdef


# ---------------------------------------------------------------------------
# 6. End-of-file hygiene guard (definition order): every throwaway opdef above
#    was removed again, so the canonical registry is intact here.
# ---------------------------------------------------------------------------


def test_registry_is_clean_after_the_throwaway_tests():
    assert ir.has_opdef(THROWAWAY_NAME) is False
    assert THROWAWAY_NAME not in ir.op_names()
