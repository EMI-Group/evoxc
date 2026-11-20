# Intel Panther Lake: short IREE Vulkan validation

## Provenance and scope

The validation recorded in commit `2e8cfaf` used Python 3.14.10, numpy 2.5.2 and the pinned IREE compiler/runtime 3.13.0 on Intel Panther Lake (`8086:b0a0`, xe render node `renderD128`) via Mesa 26.3.0 ANV.
The selected ICD enumerated `Intel(R) Graphics (PTL) (Intel open-source Mesa driver Mesa 26.3.0)`, not lavapipe.
This is small-graph correctness and residency evidence only, not performance, general Intel GPU coverage or model readiness.
The separate MI300X-host lavapipe validation remains software-Vulkan evidence; its driver and large-graph limitations are not invalidated by this Intel result.

## Recorded results

`probe_intel_vulkan.py` builds one graph with `(4,6)` f32 state and `(6,5)` f32 weights, returning elementwise state updates, row sums and matmul outputs.
Five same-device iterations passed numpy parity with `rtol=atol=1e-5` using explicit `.to(cpu).numpy()` readback.
Maximum absolute errors were `4.77e-7` (state), `0` (reduction) and `1.91e-6` (matmul).
The loop made zero `asdevicearray` calls and passed 17 guards: two host-input refusals and fifteen device `.numpy()` refusals.
The existing Vulkan residency, acquisition and target suites passed 101/101 with no skips, including the six device-guarded tests.
No adapter runtime blocker was found; no production adapter change was required.

## Reproduction

Run `python -m evoxc.bench.probe_intel_vulkan` through `uv run --locked --extra iree` under a timeout with the subprocess-scoped Intel ICD and loader-library environment.
The exact machine-local NixOS paths, enumeration command and test invocation are maintained once in [the adapter context's Intel GPU section](../backends/adapters/CONTEXT.md#intel-gpu-route-and-portability-boundaries).
Re-discover those paths after a Nix upgrade; never install this environment globally.
The small graph needs no `vulkan_target` override; this does not resolve the documented default-profile or large-transformer codegen limitations.
