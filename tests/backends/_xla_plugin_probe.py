"""Shared xla PJRT-plugin probe: which platform/device does xla actually run on?

The xla adapter drives a PJRT C API plugin resolved at RUN time —
``options['plugin_path']`` -> ``EVOXC_PJRT_PLUGIN`` -> the fixed well-known
paths -> an installed ``jax_plugins/*/*.so`` (auto-discovery; opt out with
``EVOXC_PJRT_PLUGIN_DISCOVERY=0``). "A plugin is configured" therefore no
longer implies "the plugin is a CPU plugin": on a GPU box ``pip install
jax-rocm7-pjrt`` makes ``xla.register()`` succeed with a ROCm plugin, and the
adapter then correctly REFUSES the CPU load path — ``Device("cpu", 0)`` on a
``rocm`` platform would run on the accelerator while the executable claims
CPU (see ``evoxc/backends/adapters/xla.py::_check_kind_served``).

Every xla test file derives its RUN device from the plugin the adapter
actually resolved, ONCE per process::

    path = xla_util._find_plugin_path()
    plugin = xla_util._load_plugin({"plugin_path": path})
    client = xla_util.acquire_client(plugin)
    platform = client.platform_info()[0]
    xla_util.release_client(plugin)

and maps the platform name: ``rocm``/``hip`` -> ``Device("rocm", 0)``,
``cuda`` -> ``Device("cuda", 0)``, anything else (``cpu``, or an unlisted
test-only platform such as the fake plugin's ``"fake"``) ->
``Device("cpu", 0)``. On an accelerator the files use the same EXPLICIT
placement rules as the other device-resident suites: inputs go through
``Tensor.to(device)`` (:func:`place_args`) and outputs come back through
``.to(Device("cpu", 0)).numpy()`` (:func:`to_host`).

PREFERENCE NOTE (why :func:`place_args` touches the registry):
``Tensor.to`` resolves the PREFERRED backend's per-backend provider slot, and
preference is recorded ONLY by ``evoxc.backends.registry.get`` — never by a
plain ``xla.register()``. An earlier adapter activation in the same process
(e.g. ``test_mixed_dtype.py`` building its iree legs) would otherwise own the
``rocm``/``cuda`` slot, so the probe re-records xla's preference with an
``evoxc.backends.get("xla")`` lookup immediately before placing data.

Import package-qualified (``from tests.backends import _xla_plugin_probe as
probe``): ``tests/backends`` is a package and bare imports break collection.
The underscore prefix keeps this module out of collection entirely
(``_rocm_guard.py`` / ``_adapter_utils.py`` precedent). ``evoxc`` and the xla
adapter are imported LAZILY inside the functions, so importing this helper
never requires a plugin.
"""

import functools

import numpy as np
import pytest

#: Cross-compiler float tolerance (the ``_adapter_utils`` convention: real
#: compiler fusion/reordering vs the pure-numpy interpreter reference).
RTOL = 1e-5
ATOL = 1e-5


# ---------------------------------------------------------------------------
# plugin / platform resolution (memoized once per process)
# ---------------------------------------------------------------------------


def _ensure_registered():
    """Import + register the xla adapter (idempotent) and return it.

    ``xla.register()`` is the adapter's own availability probe (bindings
    integrity, plugin discovery, ABI gate, a live client create/destroy
    round-trip) — a ``core.BackendError`` here means "no usable plugin".
    """
    from evoxc.backends.adapters import xla

    xla.register()
    return xla


@functools.lru_cache(maxsize=1)
def plugin_platform():
    """The resolved plugin's PJRT platform name (e.g. ``"rocm"``/``"cpu"``)."""
    from evoxc.backends.adapters import xla_util

    _ensure_registered()  # raises core.BackendError when no plugin exists
    path = xla_util._find_plugin_path()
    plugin = xla_util._load_plugin({"plugin_path": path})
    client = xla_util.acquire_client(plugin)
    try:
        return client.platform_info()[0]
    finally:
        xla_util.release_client(plugin)


@functools.lru_cache(maxsize=1)
def run_device():
    """The ``core.Device`` the resolved plugin physically serves.

    ``rocm``/``hip`` -> ``Device("rocm", 0)``, ``cuda`` ->
    ``Device("cuda", 0)``, everything else (``cpu`` and unlisted platforms)
    -> ``Device("cpu", 0)``.
    """
    from evoxc import core

    platform = plugin_platform().strip().lower()
    if platform in ("rocm", "hip"):
        return core.Device("rocm", 0)
    if platform == "cuda":
        return core.Device("cuda", 0)
    return core.Device("cpu", 0)


def is_accelerator():
    """Whether the derived run device is an accelerator (not ``cpu``)."""
    return run_device().kind != "cpu"


def require_plugin():
    """The per-test skip gate: return the derived run device or ``pytest.skip``.

    Only the genuine "no usable PJRT plugin" signature (a
    ``core.BackendError`` from the adapter's own probe) skips; any other
    error is a real failure and propagates.
    """
    from evoxc import core

    try:
        return run_device()
    except core.BackendError as exc:
        pytest.skip(
            "the xla adapter requires a PJRT C API plugin (.so exporting "
            "GetPjRtApi) and none is available; set "
            "EVOXC_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so, pass "
            "plugin_path=... to the compile options, or install a JAX "
            "plugin (pip install jax-rocm7-pjrt / jax-cuda12-pjrt). "
            f"Details: {exc}"
        )


# ---------------------------------------------------------------------------
# explicit placement / readback helpers
# ---------------------------------------------------------------------------


def _note_xla_preference():
    """Record xla as the PREFERRED transfer backend before placing data.

    ``Tensor.to(Device(kind, N))`` resolves the preferred backend's
    per-backend provider slot, and preference is recorded ONLY by
    ``evoxc.backends.registry.get`` — so look the backend up (a cheap dict
    hit after the first activation) on every placement.
    """
    import evoxc

    evoxc.backends.get("xla")


def place_args(args):
    """Place host arguments on the derived run device (explicit transfer).

    On a CPU platform the arguments are returned unchanged (the llvm-cpu-style
    host-input path). On an accelerator each host value is wrapped as a
    ``core.Tensor`` and moved with the ONLY transfer primitive,
    ``Tensor.to(device)``; an argument that already lives on a non-cpu device
    is passed through untouched.
    """
    args = tuple(args)
    if not is_accelerator():
        return list(args)

    import evoxc
    from evoxc import core

    _note_xla_preference()  # xla must own the (kind, backend) provider slot
    device = run_device()
    placed = []
    for arg in args:
        tensor = arg if isinstance(arg, core.Tensor) else evoxc.tensor(arg)
        if tensor.device.kind != "cpu":
            placed.append(tensor)  # already device-resident
        else:
            placed.append(tensor.to(device))
    return placed


def to_host(out):
    """Explicit readback of a run output as a numpy array.

    A device-resident output takes the explicit ``.to(Device("cpu", 0))`` hop
    (``.numpy()`` on a non-cpu tensor raises ``core.DeviceError``); a host
    output just materializes. Keyed off the TENSOR's real device so a host
    tensor produced alongside an accelerator run still reads back.
    """
    from evoxc import core

    if out.device.kind == "cpu":
        return np.asarray(out.numpy())
    return np.asarray(out.to(core.Device("cpu", 0)).numpy())


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------


def _assert_close(actual, expected, rtol, atol):
    from evoxc import core

    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)), (
            f"expected {type(expected).__name__} output, got "
            f"{type(actual).__name__}"
        )
        assert len(actual) == len(expected), (
            f"output arity {len(actual)} != expected {len(expected)}"
        )
        for got, want in zip(actual, expected):
            _assert_close(got, want, rtol, atol)
        return
    assert isinstance(actual, core.Tensor), (
        f"expected a core.Tensor output, got {type(actual).__name__}"
    )
    assert actual.dtype == expected.dtype, (
        f"result dtype {actual.dtype} != expected {expected.dtype}"
    )
    assert tuple(actual.shape) == tuple(expected.shape), (
        f"result shape {tuple(actual.shape)} != expected {tuple(expected.shape)}"
    )
    np.testing.assert_allclose(
        to_host(actual), expected.numpy(), rtol=rtol, atol=atol
    )


def parity(fn, args, executable, *, rtol=RTOL, atol=ATOL):
    """Run ``fn`` through ``executable`` and check it against the numpy backend.

    The reference is the default numpy interpreter on the HOST arguments (the
    same formula-level contract as ``_adapter_utils.assert_parity``); the
    executable's arguments are placed with :func:`place_args` (a no-op on a
    CPU platform) and its outputs read back with :func:`to_host`. Result
    dtype and shape are pinned exactly, values at the cross-compiler fp
    tolerance.
    """
    import evoxc

    expected = evoxc.evaluate(fn, *args)  # default numpy reference
    actual = evoxc.run(executable, *place_args(args))
    _assert_close(actual, expected, rtol, atol)
