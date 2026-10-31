"""Device-payload DLPack export + the zero-copy Triton/torch round trip.

Pins the device-side DLPack contract documented in ``evoxc/core/tensor.py``
(the optional payload protocol), ``evoxc/external.py`` ("Device-resident
kernels") and ``evoxc/backends/adapters/iree.py``:

* ``core.Tensor.__dlpack__`` / ``__dlpack_device__`` DELEGATE to a device
  payload's own ``__dlpack__(stream, max_version, copy, dl_device)`` /
  ``__dlpack_device__`` (the payload owns the buffer, so it owns the DLPack
  device-code mapping); a non-CPU payload WITHOUT the protocol keeps the
  historical ``DeviceError``, and a CPU-kind payload NEVER delegates (its
  explicit path stays ``.numpy()`` — host behavior unchanged).
* The iree payload (``IreeDevicePayload``) exports its HAL buffer through
  ``HalDevice.create_dlpack_capsule`` with NO copy: ``torch.from_dlpack`` on
  a device ``Tensor`` shares the memory (an in-place torch write is visible
  through evoxc, and vice versa) and ``numpy.from_dlpack`` REFUSES a
  non-CPU device instead of silently staging to the host.
* The round trip back into a graph: a ``device_resident=True`` external
  kernel receives the evoxc device tensors, imports them with
  ``torch.from_dlpack``, and RETURNS a torch tensor — the adapter imports it
  through ``HalDevice.from_dlpack_capsule`` into a device payload on the
  run's device, zero-copy, with no host-dispatch staging warning. A result
  on a different DLPack device raises an explicit ``BackendError``.
* Explicit limits: ``copy=True`` and a different ``dl_device`` raise
  ``core.DeviceError`` (zero-copy export only); the adapter adds NO fence,
  so a kernel whose work is still pending on another runtime's stream must
  synchronize that stream itself before returning (the tests below do).

Guarded like the other GPU suites: the export/round-trip legs skip unless a
real ROCm/HIP device is usable (``tests/backends/_rocm_guard.py``); the
protocol-mapping and llvm-cpu legs need only the iree compiler+runtime.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import evoxc
from evoxc.backends.adapters.iree import (
    IreeDevicePayload,
    _dlpack_device_id,
)

from tests.backends import _rocm_guard as guard

# ---------------------------------------------------------------------------
# data (exactly representable in fp32 -> bit-exact comparisons)
# ---------------------------------------------------------------------------

XH = np.arange(6, dtype=np.float32)  # [0, 1, ..., 5]
SPEC = evoxc.TensorSpec((6,), evoxc.float32)


def _np(v):
    """Explicit host access: ``.to(cpu).numpy()`` (never an implicit copy)."""
    if isinstance(v, evoxc.Tensor):
        return np.asarray(v.to(evoxc.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


def _place(host, device):
    return evoxc.core.Tensor(host).to(device)


_EXECUTABLES: dict = {}


def _build(fn, device=None, target_backends=None, specs=(SPEC,)):
    key = (fn, device, None if target_backends is None else tuple(target_backends))
    exe = _EXECUTABLES.get(key)
    if exe is None:
        kwargs = {"backend": "iree", "device": device}
        if target_backends is not None:
            kwargs["target_backends"] = target_backends
        exe = evoxc.build(fn, *specs, **kwargs)
        _EXECUTABLES[key] = exe
    return exe


def _double(x):
    return evoxc.multiply(x, 2.0)


# ---------------------------------------------------------------------------
# core delegation (payload protocol) — no device required
# ---------------------------------------------------------------------------


class _DlpackPayload:
    """A duck-typed payload implementing the OPTIONAL DLPack export protocol.

    Counts/records the delegated call so the tests can prove ``core.Tensor``
    forwards every DLPack argument verbatim and never touches the payload
    itself (no host materialization).
    """

    def __init__(self, device, shape=(2,), dtype="float32"):
        self._device = device
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.calls = []

    @property
    def device(self):
        return self._device

    def to_host(self):  # pragma: no cover - must never be called by DLPack
        raise AssertionError("DLPack export must not materialize a host copy")

    def __dlpack__(self, stream=None, max_version=None, copy=None, dl_device=None):
        self.calls.append(
            dict(
                stream=stream,
                max_version=max_version,
                copy=copy,
                dl_device=dl_device,
            )
        )
        return ("capsule", len(self.calls))

    def __dlpack_device__(self):
        return (10, self._device.index)


class _NoDlpackPayload:
    """A device payload WITHOUT the optional DLPack protocol (shape/dtype only)."""

    def __init__(self, device):
        self.shape = (2,)
        self.dtype = np.dtype("float32")
        self.device = device

    def to_host(self):
        return np.zeros((2,), dtype=np.float32)


class TestCoreDelegation:
    def test_tensor_delegates_dlpack_to_payload_verbatim(self):
        payload = _DlpackPayload(evoxc.core.Device("rocm", 1))
        t = evoxc.core.Tensor(payload)
        cap = t.__dlpack__(
            stream=7, max_version=(1, 0), copy=None, dl_device=(10, 1)
        )
        assert cap == ("capsule", 1)
        assert payload.calls == [
            dict(stream=7, max_version=(1, 0), copy=None, dl_device=(10, 1))
        ]
        assert t.__dlpack_device__() == (10, 1)

    def test_payload_without_protocol_keeps_explicit_transfer_error(self):
        t = evoxc.core.Tensor(_NoDlpackPayload(evoxc.core.Device("cuda", 0)))
        with pytest.raises(evoxc.core.DeviceError, match=r"no implicit device.*t\.to\("):
            t.__dlpack__()
        with pytest.raises(evoxc.core.DeviceError, match=r"no implicit device.*t\.to\("):
            t.__dlpack_device__()

    def test_cpu_kind_payload_never_delegates(self):
        # Host behavior is unchanged: a CPU-kind payload's DLPack path stays
        # the explicit .numpy() route, even when the payload offers the
        # protocol (the protocol is never consulted).
        payload = _DlpackPayload(evoxc.core.Device("cpu", 0))
        t = evoxc.core.Tensor(payload)
        with pytest.raises(evoxc.core.DeviceError, match=r"call \.numpy\(\) first"):
            t.__dlpack__()
        with pytest.raises(evoxc.core.DeviceError, match=r"call \.numpy\(\) first"):
            t.__dlpack_device__()
        assert payload.calls == []

    def test_host_tensor_dlpack_unchanged(self):
        arr = np.arange(4, dtype=np.float32)
        t = evoxc.core.Tensor(arr)
        out = np.from_dlpack(t)
        np.testing.assert_array_equal(out, arr)
        assert np.shares_memory(out, arr)  # zero-copy host path, as before
        assert t.__dlpack_device__() == (1, 0)

    def test_dlpack_device_id_mapping(self):
        assert _dlpack_device_id(evoxc.core.Device("cuda", 3)) == (2, 3)
        assert _dlpack_device_id(evoxc.core.Device("rocm", 1)) == (10, 1)
        assert _dlpack_device_id(evoxc.core.Device("hip", 1)) == (10, 1)
        assert _dlpack_device_id(evoxc.core.Device("vulkan", 0)) == (7, 0)
        with pytest.raises(evoxc.core.DeviceError, match="no DLPack export"):
            _dlpack_device_id(evoxc.core.Device("cpu", 0))


# ---------------------------------------------------------------------------
# llvm-cpu payloads: unchanged refusal (needs iree, no GPU)
# ---------------------------------------------------------------------------


def test_llvm_cpu_payload_dlpack_still_refuses_with_numpy_hint():
    exe = _build(_double)  # default iree device: llvm-cpu, cpu-kind payload
    out = evoxc.run(exe, evoxc.Tensor(XH))
    assert isinstance(out.data, IreeDevicePayload)
    assert out.data.device == evoxc.core.Device("cpu", 0)
    with pytest.raises(evoxc.core.DeviceError, match=r"call \.numpy\(\) first"):
        out.__dlpack__()
    with pytest.raises(evoxc.core.DeviceError, match=r"call \.numpy\(\) first"):
        out.__dlpack_device__()
    np.testing.assert_array_equal(out.numpy(), XH * 2)


# ---------------------------------------------------------------------------
# ROCm device legs (guarded)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rocm():
    return guard.require_rocm()


def _device_payload_tensor(rocm):
    exe = _build(_double, device=rocm, target_backends=["rocm"])
    out = evoxc.run(exe, _place(XH, rocm))
    assert isinstance(out.data, IreeDevicePayload)
    assert out.device == rocm
    return out


def test_rocm_tensor_reports_dlpack_rocm_device(rocm):
    out = _device_payload_tensor(rocm)
    assert out.__dlpack_device__() == (10, rocm.index)


def test_rocm_export_to_torch_is_zero_copy(rocm):
    torch = pytest.importorskip("torch")
    out = _device_payload_tensor(rocm)

    tt = torch.from_dlpack(out)  # no host round-trip
    assert tuple(tt.shape) == XH.shape
    assert tt.dtype == torch.float32
    assert tt.device.type == "cuda"  # torch names the ROCm device cuda

    # Shared memory: an in-place torch write is visible through evoxc.
    tt.add_(100.0)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(_np(out), XH * 2 + 100.0)

    # ... and an explicit evoxc-side read of the SAME buffer is the only
    # host access (no copy was made by the export).
    assert out.data.to_host().ctypes.data != 0


def test_rocm_export_is_refused_by_numpy_instead_of_host_copied(rocm):
    # numpy only imports CPU DLPack tensors: it must refuse (never silently
    # stage the device buffer to the host).
    out = _device_payload_tensor(rocm)
    with pytest.raises(BufferError):
        np.from_dlpack(out)


def test_rocm_export_copy_true_and_foreign_dl_device_raise(rocm):
    out = _device_payload_tensor(rocm)
    with pytest.raises(evoxc.core.DeviceError, match="copy=True"):
        out.__dlpack__(copy=True)
    with pytest.raises(evoxc.core.DeviceError, match="cross-device"):
        out.__dlpack__(dl_device=(2, 0))


def _torch_triple_kernel(x):
    """A device-resident 'Triton-like' kernel: torch in, torch out.

    Proves the whole contract the downstream consumer uses: operands arrive
    as device ``core.Tensor`` s (``IreeDevicePayload``), ``torch.from_dlpack``
    is the zero-copy import, and the returned ``torch.Tensor`` is imported
    back by the adapter. ``synchronize`` is REQUIRED: iree's HIP HAL runs on
    its own stream and the adapter adds no fence.
    """
    torch = pytest.importorskip("torch")
    assert isinstance(x.data, IreeDevicePayload)  # never host-staged
    tt = torch.from_dlpack(x)
    out = tt * 3.0
    torch.cuda.synchronize()
    return out


def _roundtrip_graph(x):
    y = evoxc.external_call(
        "t_dlpack_torch_triple", x, result=evoxc.TensorSpec((6,), evoxc.float32)
    )
    return y + 1.0


def _host_triple_kernel(x):
    """The numpy-backend reference for the same graph (host-mode kernel)."""
    return np.asarray(x) * 3.0


@pytest.fixture(scope="module")
def rocm_roundtrip(rocm):
    evoxc.register_external_kernel(
        "t_dlpack_torch_triple", _host_triple_kernel
    )
    evoxc.register_external_kernel(
        "t_dlpack_torch_triple", _torch_triple_kernel, backend="iree",
        device_resident=True,
    )
    try:
        yield rocm
    finally:
        evoxc.unregister_external_kernel("t_dlpack_torch_triple")


def test_rocm_torch_result_roundtrip_zero_copy(rocm_roundtrip):
    import warnings

    torch = pytest.importorskip("torch")
    exe = _build(
        _roundtrip_graph,
        device=rocm_roundtrip,
        target_backends=["rocm"],
    )
    with warnings.catch_warnings():
        # A fully device-resident boundary stages NOTHING host-side, so the
        # once-per-executable "host-dispatch" staging warning must not fire.
        warnings.simplefilter("error")
        out = evoxc.run(exe, _place(XH, rocm_roundtrip))

    want = evoxc.evaluate(_roundtrip_graph, XH)
    assert isinstance(out.data, IreeDevicePayload)
    assert out.device == rocm_roundtrip
    np.testing.assert_array_equal(_np(out), _np(want))
    np.testing.assert_array_equal(_np(out), XH * 3.0 + 1.0)

    # The graph output is an ordinary device tensor: as before the export
    # path works on it too.
    tt = torch.from_dlpack(out)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(tt.cpu().numpy(), _np(want))


def test_rocm_result_on_other_dlpack_device_raises(rocm_roundtrip):
    """A device kernel returning a CPU torch tensor is rejected explicitly."""
    torch = pytest.importorskip("torch")

    def kernel(x):
        return torch.arange(6, dtype=torch.float32)  # CPU tensor: DLPack (1, 0)

    name = "t_dlpack_wrong_device"
    evoxc.register_external_kernel(name, kernel, backend="iree", device_resident=True)
    try:
        def graph(x):
            return evoxc.external_call(
                name, x, result=evoxc.TensorSpec((6,), evoxc.float32)
            )

        exe = evoxc.build(
            graph,
            SPEC,
            backend="iree",
            device=rocm_roundtrip,
            target_backends=["rocm"],
        )
        with pytest.raises(
            evoxc.core.BackendError, match=r"DLPack tensor on device .*executable runs"
        ):
            evoxc.run(exe, _place(XH, rocm_roundtrip))
    finally:
        evoxc.unregister_external_kernel(name)
