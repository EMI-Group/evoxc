"""Short real-Intel Vulkan validation; run with ``python -m`` under a timeout.

Select the Intel ANV ICD and wheel loader dependencies in the subprocess
environment (see CONTEXT.md); this probe never changes the environment or ICD.
It is evidence tooling, not imported by evoxc and not a benchmark.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    import numpy as np
    import iree.runtime as rt
    import evoxc

    versions = {
        name: importlib.metadata.version(name)
        for name in ("iree-base-compiler", "iree-base-runtime", "numpy")
    }
    assert versions["iree-base-compiler"] == versions["iree-base-runtime"] == "3.13.0"
    infos = rt.get_driver("vulkan").query_available_devices()
    print(json.dumps({"versions": versions, "devices": infos}), flush=True)
    intel = [i for i, info in enumerate(infos) if "Intel" in info["name"]]
    assert len(intel) == 1, "Select exactly one Intel device via the ANV ICD"
    device = evoxc.Device("vulkan", intel[0])
    cpu = evoxc.Device("cpu", 0)

    @evoxc.defn
    def step(x, w):
        state = x + 1.0
        return state, evoxc.sum(state, axes=1), evoxc.dot(state, w)

    x = np.arange(24, dtype=np.float32).reshape(4, 6) / 24
    w = np.arange(30, dtype=np.float32).reshape(6, 5) / 30
    exe = evoxc.build(
        step, evoxc.TensorSpec(x.shape, x.dtype),
        evoxc.TensorSpec(w.shape, w.dtype), backend="iree", device=device,
    )
    assert exe.backend_executable.artifact.target == "vulkan"
    host = evoxc.from_numpy(x)
    state, weights = host.to(device), evoxc.from_numpy(w).to(device)
    guards = 0
    for invalid in (host, x):
        try:
            evoxc.run(exe, invalid, weights)
        except evoxc.core.DeviceError:
            guards += 1
        else:
            raise AssertionError("host input was silently accepted")

    # Placement occurs above; every invocation, including the first, must pass
    # device buffers through without the host-upload ABI helper.
    max_errors = []
    with patch.object(rt, "asdevicearray", side_effect=AssertionError("implicit upload")) as upload:
        for iteration in range(1, 6):
            state, reduction, product = evoxc.run(exe, state, weights)
            expected = x + iteration
            errors = []
            for result, reference in zip(
                (state, reduction, product),
                (expected, expected.sum(axis=1), expected @ w), strict=True,
            ):
                assert result.device == device
                try:
                    result.numpy()
                except evoxc.core.DeviceError:
                    guards += 1
                else:
                    raise AssertionError("implicit readback was silently accepted")
                actual = result.to(cpu).numpy()
                np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
                errors.append(float(np.max(np.abs(actual - reference))))
            max_errors.append(errors)
        assert upload.call_count == 0
    pci = Path("/sys/class/drm/renderD128/device")
    print(json.dumps({
        "device": str(device), "iterations": 5, "placement_guards": guards,
        "asdevicearray_calls_in_loop": upload.call_count,
        "max_abs_errors_state_reduce_matmul": max_errors,
        "renderD128_pci": {
            key: (pci / key).read_text().strip() for key in ("vendor", "device")
        }, "status": "PASS",
    }), flush=True)


if __name__ == "__main__":
    main()
