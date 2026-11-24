"""Custom op defs: ``custom_call`` (a StableHLO custom-call target) and
``inline_stablehlo`` (a raw StableHLO/MLIR snippet emitted verbatim).

Both ops mirror ``external_call``: they are opaque, declared-``result_specs``
ops with NO shape function — result types resolve from the declared
``result_specs`` attribute (or explicit ``result_types``), and verification
checks that the two agree.

Effects: both are ``callback`` — opaque external side effects, never
reordered, duplicated or eliminated.
"""

from __future__ import annotations

from ..effects import EFFECT_CALLBACK
from . import (
    ATTR_ANY,
    ATTR_BOOL,
    ATTR_INT,
    ATTR_STR,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "custom"


def _register_custom() -> None:
    register_opdef(
        OpDef(
            name="custom_call",
            category=_CATEGORY,
            description="Invoke a named StableHLO custom-call target "
            "(opaque compiler escape hatch; effect 'callback': never "
            "reordered/duplicated/eliminated). The target is declared by "
            "NAME plus an optional backend configuration string — graph "
            "artifacts carry no kernel code.",
            arity=(0, None),
            result_count=None,
            effect=EFFECT_CALLBACK,
            attributes=(
                AttrSpec(
                    name="call_target_name",
                    type=ATTR_STR,
                    description="StableHLO custom-call target symbol "
                    "(non-empty; identifies the external callable).",
                ),
                AttrSpec(
                    name="result_specs",
                    type=ATTR_ANY,
                    description="Declared result ValueTypes (serialized "
                    "structurally); must equal the op's result types.",
                ),
                AttrSpec(
                    name="backend_config",
                    type=ATTR_STR,
                    default="",
                    description="Opaque backend-specific configuration string "
                    "passed through with the call (empty = none).",
                ),
                AttrSpec(
                    name="api_version",
                    type=ATTR_INT,
                    default=1,
                    description="Custom-call API version (>= 1; the "
                    "StableHLO custom_call api_version field).",
                ),
                AttrSpec(
                    name="has_side_effect",
                    type=ATTR_BOOL,
                    default=False,
                    description="True if the target has observable side "
                    "effects (opaque to the compiler).",
                ),
            ),
        )
    )
    register_opdef(
        OpDef(
            name="inline_stablehlo",
            category=_CATEGORY,
            description="Emit a raw StableHLO/MLIR snippet as this op's "
            "lowering (opaque compiler escape hatch; effect 'callback': "
            "never reordered/duplicated/eliminated).",
            arity=(0, None),
            result_count=None,
            effect=EFFECT_CALLBACK,
            attributes=(
                AttrSpec(
                    name="body",
                    type=ATTR_STR,
                    description="Raw StableHLO/MLIR snippet emitted verbatim "
                    "(non-empty).",
                ),
                AttrSpec(
                    name="result_specs",
                    type=ATTR_ANY,
                    description="Declared result ValueTypes (serialized "
                    "structurally); must equal the op's result types.",
                ),
                AttrSpec(
                    name="has_side_effect",
                    type=ATTR_BOOL,
                    default=False,
                    description="True if the snippet has observable side "
                    "effects (opaque to the compiler).",
                ),
            ),
        )
    )


_register_custom()
