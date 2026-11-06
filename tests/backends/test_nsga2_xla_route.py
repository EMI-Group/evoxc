"""XLA-route argsort regression (the NSGA2-GPU backend-switch evidence, fix 71d721e).

The recommended NSGA2-on-GPU route is the xla adapter with the real PJRT
plugin — xla handles argsort at sorted-axis extent >= 32 natively: no
``stablehlo.sort`` bufferization bug, no while-init AffinityAnalysis SEGV
(see ``evoxc/backends/adapters/CONTEXT.md`` Known Issue 3). Pinned here: an
argsort axis >= 32 graph (the count-mode trigger class) must build+run on the
xla adapter bit-exact vs numpy whenever a PJRT plugin is resolvable.

The xla adapter has no hard pip dependency — it drives the PJRT C API via
ctypes over a plugin ``.so`` exporting ``GetPjRtApi``, resolved in the order
``plugin_path`` -> ``EVOXC_PJRT_PLUGIN`` -> the fixed well-known paths -> an
installed ``jax_plugins/*/*.so`` (auto-discovery; ``pip install
jax-rocm7-pjrt`` / ``jax-cuda12-pjrt``; opt out with
``EVOXC_PJRT_PLUGIN_DISCOVERY=0``). Without one the tests skip with an
actionable message (the same gate as ``tests/backends/test_mixed_dtype.py``
and ``test_adapter_xla.py``): set
``EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so`` or pass
``plugin_path=...`` to the compile options.

**Derived run device.** A resolved plugin may be an ACCELERATOR plugin, so the
run device comes from its platform (``tests/backends/_xla_plugin_probe.py``:
``rocm``/``hip`` -> ``Device("rocm", 0)``, ``cuda`` -> ``Device("cuda", 0)``,
else ``Device("cpu", 0)``): build with ``device=<that device>``, place the
input with ``Tensor.to(device)`` and read the result back through
``.to(cpu)`` before comparing with the numpy reference.
"""

import numpy as np
import pytest

import evoxc

from tests.backends import _xla_plugin_probe as probe


def _require_xla():
    """Skip (per-test) when no PJRT plugin is resolvable; return its device."""
    return probe.require_plugin()


def _argsort_axis1(x):
    return evoxc.argsort(x, axis=1)


def _argsort_axis1_desc(x):
    return evoxc.argsort(x, axis=1, descending=True)


_RNG = np.random.default_rng(7)
_X = _RNG.standard_normal((8, 64)).astype(np.float32)
_SPEC = evoxc.TensorSpec((8, 64), evoxc.float32)

#: (id, fn, numpy-reference) — the xla pair emission sorts the (value, iota)
#: pair ascending; descending = flip (the numpy composition). fp32 noise has
#: no ties, so the stable reference is exact.
CASES = [
    ("xla_argsort_axis64", _argsort_axis1,
     np.argsort(_X, axis=1, kind="stable")),
    ("xla_argsort_axis64_desc", _argsort_axis1_desc,
     np.flip(np.argsort(_X, axis=1, kind="stable"), axis=1)),
]


@pytest.mark.parametrize(
    "fn,ref", [(fn, ref) for _, fn, ref in CASES], ids=[c[0] for c in CASES]
)
def test_xla_argsort_axis_ge_32(fn, ref):
    """The backend-switch evidence: on xla the axis-64 argsort (the iree
    count-mode trigger class) builds and runs bit-exact vs numpy."""
    device = _require_xla()
    exe = evoxc.build(fn, _SPEC, backend="xla", device=device)
    (x,) = probe.place_args((_X,))
    got = np.asarray(probe.to_host(evoxc.run(exe, x)))
    assert np.array_equal(got, ref), f"{got} != {ref}"
