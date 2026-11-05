"""iree adapter ROCm/HIP while / scan device-resident regression suite (iree 3.13).

On the pinned iree 3.11.0 the HIP HAL aborted at run time on loop-carrying
graphs — ``hip/event_semaphore.c:786: ABORTED; the semaphore was aborted;
while invoking native function hal.fence.await`` — for a ``while_loop`` whose
carry is initialized from a graph CONSTANT (``evoxc.enp.zeros``, the same
class ``evoxc.scan`` desugars to), and a trivial ``evoxc.scan`` graph SIGSEGV'd
the process outright (``stream_command_buffer.c:452 INVALID_ARGUMENT`` /
``hal.device.queue.alloca`` / the same fence abort). Those observations come
from the TensorNEAT ROCm spike's ``repro_minimal.py`` + ``ROCM_FINDINGS.md``
(archived outside this repository).

``iree-base-compiler`` / ``iree-base-runtime`` 3.13.0 fixes that class: the
shapes below compile and RUN device-resident on the AMD GPU, so this suite
pins them as ordinary working paths instead of skipping them. Whatever the
graph LAYOUT, the explicit-placement contract is unchanged and asserted here
for each shape:

* ``evoxc.build(..., backend="iree", target_backends=["rocm"],
  device=Device("rocm", 0))`` compiles and runs on the AMD GPU;
* every run output is a payload-backed ``core.Tensor`` whose ``.device`` IS
  the executable's ``core.Device`` (nothing is auto-copied to the host), so
  ``.numpy()`` on it RAISES ``core.DeviceError``;
* the explicit ``t.to(core.Device("cpu", 0))`` hop is the only host path and
  its values match the numpy interpreter (the reference backend);
* inputs are placed once via the explicit transfer API — a rocm-resident run
  is a same-device pass-through (host inputs raise at the run boundary, as in
  ``test_iree_rocm_device_resident.py``; re-asserted here on the loop shapes).

Deliberately NO timing / benchmarking: four short correctness shapes only, so
the shared GPU box stays cheap. IREE runs the real MLIR compiler (about a
second per compile here), so one executable is built per shape and shared
across the module.

Note the one loop layout that is NOT pinned here because iree 3.13.0 still
rejects it at COMPILE time (``error: maximum iteration count exceeded in fixed
point pipeline``, both targets, no crash): an entirely INPUT-INDEPENDENT
counted loop — the carry is initialized from a graph constant, the condition's
bound is another graph constant (``evoxc.enp.full((), 5, dtype=evoxc.int32)``)
and the body touches only constants, so the whole ``stablehlo.while`` is loop
invariant. The graph-constant init/bound pair alone is fine as soon as the body
consumes the input (the second shape in section 1), and the static
Python-``int`` bound of section 1 — the historical 3.11 "abort" repro — runs
fine either way.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import evoxc
import evoxc.numpy as enp

from tests.backends import _rocm_guard as guard

# ---------------------------------------------------------------------------
# graph builders (the three historically-fatal layout classes)
# ---------------------------------------------------------------------------

F32 = evoxc.float32
I32 = evoxc.int32

#: The Python-int bound of the counted loop in ``_const_init_while``. A static
#: Python value specialises the graph (the documented value model) and keeps
#: the condition free of a second graph constant.
_ITERATIONS = 5


def _const_init_while(x):
    """``while_loop`` whose carry is initialized from a graph CONSTANT.

    The rank-0 ``int32`` counter starts as ``enp.zeros`` (a graph constant,
    not an input and not a Python value) and the condition compares it to a
    static Python ``int`` literal — the exact shape that aborted at run time
    on iree 3.11.0 (``hal.fence.await`` ABORTED) and the same constant-init
    class ``evoxc.scan`` desugars its carry into.
    """
    one_i32 = enp.full((), 1, dtype=I32)

    def cond_fn(carry):
        (counter,) = carry
        return evoxc.less(counter, _ITERATIONS)

    def body_fn(carry):
        (counter,) = carry
        return (evoxc.add(counter, one_i32),)

    (counter,) = evoxc.while_loop(cond_fn, body_fn, (enp.zeros((), I32),))
    return evoxc.multiply(x, evoxc.enp.astype(counter, F32))


#: The float-carry bound of the multi-carry loop below (a Python value).
_ACC_LIMIT = 20.0
_MAX_ITERS = 8


def _const_init_const_bound_while(x):
    """Constant-init carries AND a graph-constant condition bound.

    Both carried values start as graph CONSTANTS (the ``int32`` counter and the
    ``(4,)`` float accumulator, both ``enp.zeros``) and the condition compares
    the counter to another graph constant (``enp.full((), 5, int32)``) — the
    full ``evoxc.scan`` desugaring style. The body consumes the input, which is
    what keeps the program out of the one remaining iree 3.13 compile failure
    (see the module docstring): an entirely input-independent loop.
    """
    bound = enp.full((), _ITERATIONS, dtype=I32)
    one_i32 = enp.full((), 1, dtype=I32)

    def cond_fn(carry):
        _acc, counter = carry
        return evoxc.less(counter, bound)

    def body_fn(carry):
        acc, counter = carry
        return (evoxc.add(acc, x), evoxc.add(counter, one_i32))

    acc, _counter = evoxc.while_loop(
        cond_fn, body_fn, (enp.zeros((4,), F32), enp.zeros((), I32))
    )
    return acc


def _float_cond_while(x):
    """Multi-carry loop whose CONDITION reads the float carry.

    Three carried values (a float accumulator, a float scale vector and an
    int32 iteration counter); the condition consumes ``max(acc)`` — a runtime
    reduction over the float carry — beside the counter. This is the layout
    the 3.11 HIP HAL aborted on with the fence signature.
    """
    one_i32 = enp.full((), 1, dtype=I32)
    two = enp.full((), 2.0, dtype=F32)

    def cond_fn(carry):
        acc, _scale, i = carry
        return evoxc.logical_and(
            evoxc.less(evoxc.max(acc), _ACC_LIMIT),
            evoxc.less(i, _MAX_ITERS),
        )

    def body_fn(carry):
        acc, scale, i = carry
        return (
            evoxc.add(acc, evoxc.multiply(x, scale)),
            evoxc.multiply(scale, two),
            evoxc.add(i, one_i32),
        )

    acc, scale, i = evoxc.while_loop(
        cond_fn,
        body_fn,
        (enp.zeros((4,), F32), enp.ones((4,), F32), enp.zeros((), I32)),
    )
    return acc, scale, i


def _scan(xs):
    """An ``evoxc.scan`` graph (a trivial-in-3.11 body is enough to SIGSEGV).

    The carry starts as the graph constant ``enp.zeros((), float32)`` and each
    step reduces the sliced row — the desugared while carries a constant-init
    stack, which is exactly the 3.11 failure class. The body here does real
    work (``sum`` per step + a scaled output) and the amortised accumulator is
    an exactly-representable integer in fp32, so parity is EXACT.
    """
    init = enp.zeros((), F32)

    def step_fn(carry, x_step):
        return evoxc.add(carry, evoxc.sum(x_step)), evoxc.multiply(x_step, 2.0)

    carry, ys = evoxc.scan(step_fn, init, xs)
    return carry, ys


# ---------------------------------------------------------------------------
# data (deterministic; small shapes — GPU use stays short)
# ---------------------------------------------------------------------------

XV = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
XS = np.arange(20, dtype=np.float32) + 1.0  # 1..20

WHILE_SPECS = (evoxc.TensorSpec((4,), F32),)
SCAN_SPECS = (evoxc.TensorSpec((20,), F32),)

# ---------------------------------------------------------------------------
# helpers (the same explicit-host-access discipline as the resident suite)
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors -> ndarray / tuple-of-ndarrays (explicit D2H).

    A rocm payload's bare ``.numpy()`` raises, so host materialization goes
    through the explicit ``t.to(cpu)`` transfer.
    """
    if isinstance(v, evoxc.Tensor):
        return np.asarray(v.to(evoxc.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


def _place(host, device):
    """Place host numpy data on ``device`` via the explicit transfer API."""
    return evoxc.core.Tensor(np.ascontiguousarray(host)).to(device)


def _assert_close(got, want, rtol=1e-5, atol=1e-5):
    g, w = evoxc.tree_map(_np, got), evoxc.tree_map(_np, want)
    for gp, wp in zip(evoxc.tree_leaves(g), evoxc.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.allclose(gp, wp, rtol=rtol, atol=atol), f"{gp} != {wp}"


def _assert_exact(got, want):
    g, w = evoxc.tree_map(_np, got), evoxc.tree_map(_np, want)
    for gp, wp in zip(evoxc.tree_leaves(g), evoxc.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _assert_device_resident(out, device):
    """A run-out tensor is payload-backed, on ``device``, with a lazy host copy.

    Duck-typed payload assertions only (the payload class is private): not an
    ndarray, knows its shape/dtype/device, and exposes the ``to_host`` path.
    """
    assert isinstance(out, evoxc.Tensor)
    assert not isinstance(out.data, np.ndarray)  # NOT a numpy-backed tensor
    assert out.device == device  # the executable's core.Device label
    assert hasattr(out.data, "to_host")  # duck-typed payload protocol
    assert out.data.device == device  # the payload knows its device
    assert out.shape == out.data.shape
    assert out.dtype == out.data.dtype


def _assert_no_implicit_host_copy(out, device):
    """Every output leaf is device-resident and refuses a bare ``.numpy()``."""
    for leaf in evoxc.tree_leaves(out):
        _assert_device_resident(leaf, device)
        with pytest.raises(
            evoxc.core.DeviceError, match="no implicit device-to-host"
        ):
            leaf.numpy()


_EXECUTABLES: dict = {}


def _build_exe(fn, specs, device, target_backends=("rocm",)):
    """One rocm compile per (fn, specs, device, target_backends) — module-wide."""
    key = (
        fn,
        specs,
        device,
        None if target_backends is None else tuple(target_backends),
    )
    exe = _EXECUTABLES.get(key)
    if exe is None:
        kwargs = {"backend": "iree", "device": device}
        if target_backends is not None:
            kwargs["target_backends"] = list(target_backends)
        exe = evoxc.build(fn, *specs, **kwargs)
        _EXECUTABLES[key] = exe
    return exe


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rocm():
    """The ROCm evoxc device; ``pytest.skip`` when no HIP device is usable."""
    return guard.require_rocm()


# ---------------------------------------------------------------------------
# 1. while_loop with a graph-CONSTANT carry init (the scan failure class)
# ---------------------------------------------------------------------------


def test_constant_init_while_runs_device_resident(rocm):
    exe = _build_exe(_const_init_while, WHILE_SPECS, rocm)
    out = evoxc.run(exe, _place(XV, rocm))
    # Residency: payload-backed, labelled rocm, no bare host readback.
    _assert_no_implicit_host_copy(out, rocm)
    assert out.shape == (4,) and out.dtype == np.dtype("float32")
    # Counter 0 -> 5 straight from the graph-constant zero: x * 5 (exact).
    _assert_exact(out, XV * float(_ITERATIONS))
    # ... and the numpy interpreter (the reference) agrees.
    _assert_close(out, evoxc.evaluate(_const_init_while, XV, backend="numpy"))


def test_constant_init_and_constant_bound_while_runs_device_resident(rocm):
    """Constant-init carries + a graph-constant condition bound (scan style)."""
    exe = _build_exe(_const_init_const_bound_while, WHILE_SPECS, rocm)
    out = evoxc.run(exe, _place(XV, rocm))
    _assert_no_implicit_host_copy(out, rocm)
    assert out.shape == (4,) and out.dtype == np.dtype("float32")
    # Five body iterations, each adding x to the graph-constant zero carry: x*5.
    _assert_exact(out, XV * float(_ITERATIONS))
    _assert_close(
        out, evoxc.evaluate(_const_init_const_bound_while, XV, backend="numpy")
    )


def test_constant_init_while_is_a_same_device_pass_through(rocm, monkeypatch):
    """A second run on the same device never stages the host again."""
    exe = _build_exe(_const_init_while, WHILE_SPECS, rocm)
    x = _place(XV, rocm)

    import iree.runtime as rt

    calls = []

    def _forbid(*args, **kwargs):
        calls.append(args)
        raise AssertionError(
            "iree.runtime.asdevicearray must never be called: a rocm run with "
            "device-resident inputs is a same-device pass-through"
        )

    monkeypatch.setattr(rt, "asdevicearray", _forbid)
    out = evoxc.run(exe, x)
    _assert_exact(out, XV * float(_ITERATIONS))
    assert calls == []


# ---------------------------------------------------------------------------
# 2. multi-carry loop whose CONDITION reads the float carry
# ---------------------------------------------------------------------------


def test_multi_carry_float_condition_device_resident(rocm):
    exe = _build_exe(_float_cond_while, WHILE_SPECS, rocm)
    out = evoxc.run(exe, _place(XV, rocm))
    acc, scale, count = out
    _assert_no_implicit_host_copy(out, rocm)
    expected = (((4,), np.float32), ((4,), np.float32), ((), np.int32))
    for leaf, (shape, dtype) in zip(evoxc.tree_leaves(out), expected):
        assert leaf.shape == shape
        assert leaf.dtype == np.dtype(dtype)
    # Independent trace of the loop (hand-computed): three iterations, the
    # third's max(acc) = 28 crosses the float-carry bound of 20.
    #   acc: 1*[1,2,3,4] + 2*[1,2,3,4] + 4*[1,2,3,4] = [7, 14, 21, 28]
    #   scale: 1 -> 2 -> 4 -> 8 (produced, not consumed, by the exit iteration)
    #   count: 0 -> 1 -> 2 -> 3
    assert np.array_equal(_np(acc), np.array([7.0, 14.0, 21.0, 28.0], np.float32))
    assert np.array_equal(_np(scale), np.array([8.0, 8.0, 8.0, 8.0], np.float32))
    assert np.array_equal(_np(count), np.array(3.0, np.float32))
    # The numpy interpreter (the reference) agrees on all three carries.
    _assert_close(out, evoxc.evaluate(_float_cond_while, XV, backend="numpy"))


# ---------------------------------------------------------------------------
# 3. evoxc.scan (the 3.11 SIGSEGV class)
# ---------------------------------------------------------------------------


def test_scan_device_resident(rocm):
    exe = _build_exe(_scan, SCAN_SPECS, rocm)
    carry, ys = evoxc.run(exe, _place(XS, rocm))
    _assert_no_implicit_host_copy((carry, ys), rocm)
    assert carry.shape == () and ys.shape == (20,)
    assert ys.dtype == np.dtype("float32")
    # Exactly representable in fp32: the carry is sum(1..20) = 210, ys = 2*xs.
    assert np.array_equal(_np(carry), np.array(210.0, np.float32))
    assert np.array_equal(_np(ys), (XS * 2.0).astype(np.float32))
    # The numpy interpreter (the reference) agrees.
    _assert_close((carry, ys), evoxc.evaluate(_scan, XS, backend="numpy"))


# ---------------------------------------------------------------------------
# 4. the placement boundary is unchanged by the loop shapes
# ---------------------------------------------------------------------------


def test_loop_graphs_reject_host_inputs_at_run_boundary(rocm):
    """A rocm executable NEVER stages host inputs — loop graphs included."""
    exe = _build_exe(_scan, SCAN_SPECS, rocm)
    with pytest.raises(
        evoxc.core.DeviceError,
        match="no implicit device-to-host or host-to-device transfer happens "
        "at the run boundary",
    ):
        evoxc.run(exe, XS)
    with pytest.raises(evoxc.core.DeviceError) as excinfo:
        evoxc.run(exe, evoxc.core.Tensor(XS))
    msg = str(excinfo.value)
    assert "input at path [0]" in msg
    assert "t.to(" in msg  # the explicit placement remedy
    # Positive control: the explicitly placed input runs the scan fine.
    carry, ys = evoxc.run(exe, _place(XS, rocm))
    _assert_no_implicit_host_copy((carry, ys), rocm)
    _assert_exact(ys, XS * 2.0)
