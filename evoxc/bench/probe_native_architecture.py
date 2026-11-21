"""Native (Rust) transform engine — end-to-end evidence probe (CPU + Intel Vulkan).

Deterministic (``--seed 0``) public-API-only probe that measures and checks the
Rust transform engine on this host:

* graph construction and the native ``vectorize`` / ``vmap`` / ``grad`` cost on
  a ~1500-op chain (min/median of ``REPS``; the load average is reported);
* ``batch_functions`` on 64 identical programs and on a branchy population with
  ``switch_ops=True`` — merge time plus the 12-key ``graph.batch_report``;
* the CPU end-to-end explicit pipeline (``trace -> lower -> compile -> load``)
  on a vmap-batched per-sample graph (elementwise + reduction + dot), warm
  ``run`` timings and numpy parity;
* the SAME graph on the Intel Xe GPU through IREE Vulkan/SPIR-V — explicit
  ``Tensor.to(Device("vulkan", 0))`` placement, device-resident outputs, numpy
  parity, warm ``run`` timings, and the ``.numpy()`` refusal / ``.to(cpu)``
  readback guards;
* a native-transform + ``batch_functions`` merged graph executing on Vulkan,
  so the graph the GPU runs is the one the Rust engine produced.

Run it explicitly (this module is evidence tooling — ``evoxc`` never imports it
and pytest never collects it)::

    source scripts/dev-env.sh
    .venv/bin/python evoxc/bench/probe_native_architecture.py --seed 0

``--skip-vulkan`` runs the CPU half only on a host without a usable Vulkan
device.  Any correctness failure exits nonzero; an environment blocker on the
Vulkan leg is diagnosed, recorded in the output and also exits nonzero while
still printing the CPU results.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import warnings

REPS = 7  # transform / merge repetitions (min + median)
WARM = 20  # warm run repetitions
CHAIN_ITERATIONS = 560  # ~1500 IR ops (measured and reported)
BATCH_N = 64


# ---------------------------------------------------------------------------
# small timing / formatting helpers
# ---------------------------------------------------------------------------
def _timed(fn, reps):
    """Run ``fn`` ``reps`` times; return (last_result, min_ms, median_ms, samples)."""
    samples = []
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, min(samples), statistics.median(samples), samples


def _loadavg():
    try:
        return tuple(round(value, 2) for value in os.getloadavg())
    except OSError:
        return None


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _versions():
    versions = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
    }
    for name in ("iree-base-compiler", "iree-base-runtime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import evoxc

    versions["evoxc"] = importlib.metadata.version("evoxc")
    from evoxc import _native

    versions["native_available"] = bool(_native.available())
    versions["native_version"] = _native.version() if _native.available() else None
    return versions


# ---------------------------------------------------------------------------
# deterministic graph builders (public API only)
# ---------------------------------------------------------------------------
def _make_chain(iterations):
    """A ~3-op-per-iteration elementwise chain ending in a full reduction."""
    import evoxc

    @evoxc.defn
    def chain(x):
        y = x
        for i in range(iterations):
            remainder = i % 3
            if remainder == 0:
                y = evoxc.tanh(y)
            elif remainder == 1:
                y = evoxc.add(evoxc.multiply(y, 1.0001), 0.25)
            else:
                y = evoxc.add(y, evoxc.multiply(y, 0.5))
        return evoxc.sum(y)

    return chain


def _switch_ops(evoxc):
    """The four switchable elementwise variants.

    Bounded values keep the fp32 comparison well conditioned (no exp/square
    blow-up), so the merged graph's rows can be compared tightly.
    """
    return (evoxc.relu, evoxc.tanh, evoxc.negate, evoxc.abs)


def _make_program(variant):
    """One branchy population program: same skeleton, different unary op."""
    import evoxc

    ops = _switch_ops(evoxc)

    @evoxc.defn
    def program(x):
        y = evoxc.add(evoxc.multiply(x, 1.5), 0.25)
        y = ops[variant % len(ops)](y)
        y = evoxc.add(evoxc.multiply(y, y), 0.5)
        return evoxc.sum(y, axes=1)

    return program


def _make_per_sample_model():
    """The vmap-batched per-sample function: dot + elementwise + reduction."""
    import evoxc

    @evoxc.defn
    def per_sample(x, w, b):
        h = evoxc.tanh(evoxc.dot(x, w) + b)  # dot + elementwise
        return evoxc.sum(h, axes=(1,))  # per-sample reduction (static axis)

    return per_sample


def _per_sample_reference(x, w, b):
    import numpy as np

    return np.tanh(x @ w + b).sum(axis=2)


# ---------------------------------------------------------------------------
# batch-report summary
# ---------------------------------------------------------------------------
def _report_dict(graph):
    report = graph.batch_report
    if report is None:
        return None
    return json.loads(json.dumps(dataclasses.asdict(report)))


# ---------------------------------------------------------------------------
# staging helpers
# ---------------------------------------------------------------------------
def _numpy_executable(graph):
    import evoxc

    return evoxc.load(evoxc.compile(evoxc.lower(graph)))


def _target_for(device):
    """The iree compile target for a device, mirroring the pipeline's mapping."""
    return {
        "cpu": "llvm-cpu",
        "cuda": "cuda",
        "rocm": "rocm",
        "hip": "rocm",
        "vulkan": "vulkan-spirv",
    }[device.kind]


def _max_abs_error(got, want):
    import numpy as np

    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: {got.shape} != {want.shape}")
    return float(np.max(np.abs(got - want)))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def run_transform_costs(evoxc, summary, failures):
    graph = evoxc.trace(
        _make_chain(CHAIN_ITERATIONS),
        evoxc.TensorSpec((8, 16), evoxc.float32),
    )
    op_count = len(graph.module.main.entry_block.ops)
    _, trace_min, trace_median, _ = _timed(
        lambda: evoxc.trace(
            _make_chain(CHAIN_ITERATIONS),
            evoxc.TensorSpec((8, 16), evoxc.float32),
        ),
        REPS,
    )
    print(
        f"\n[A] graph construction: {op_count} IR ops "
        f"(trace min {trace_min:.1f} ms, median {trace_median:.1f} ms, "
        f"n={REPS})"
    )
    summary["transform_costs"] = {
        "chain_op_count": op_count,
        "trace_min_ms": trace_min,
        "trace_median_ms": trace_median,
        "loadavg": _loadavg(),
        "transforms": {},
    }

    cases = {
        "vectorize": lambda: evoxc.vectorize(graph, (0,)),
        "vmap": lambda: evoxc.vmap(graph, in_axes=0, out_axes=0),
        "grad": lambda: evoxc.grad(graph, argnums=0),
    }
    print(f"{'transform':<12}{'min_ms':>10}{'median_ms':>12}{'out_ops':>10}")
    for name, fn in cases.items():
        out, low, median, _ = _timed(fn, REPS)
        out_ops = len(out.module.main.entry_block.ops)
        print(f"{name:<12}{low:>10.2f}{median:>12.2f}{out_ops:>10}")
        summary["transform_costs"]["transforms"][name] = {
            "min_ms": low,
            "median_ms": median,
            "out_ops": out_ops,
        }
        if not out_ops:
            failures.append(f"{name} produced an empty graph")


def run_batch_functions(evoxc, summary, failures, seed):
    import numpy as np

    spec = evoxc.TensorSpec((4, 8), evoxc.float32)
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((BATCH_N, 4, 8)).astype(np.float32)

    def merge(fns, switch_ops):
        return lambda: evoxc.batch_functions(
            fns, axes=0, out_axes=0, switch_ops=switch_ops
        )

    # --- 64 identical programs --------------------------------------------
    identical_graph = evoxc.trace(_make_program(0), spec)
    identical_fns = [identical_graph] * BATCH_N
    merged_identical, ident_min, ident_median, _ = _timed(
        merge(identical_fns, None), REPS
    )
    ident_report = _report_dict(merged_identical)
    print(
        f"\n[B] batch_functions: identical N={BATCH_N} merge "
        f"min {ident_min:.2f} ms / median {ident_median:.2f} ms"
    )
    print(f"    report: {json.dumps(ident_report)}")

    # Row correctness on the numpy interpreter: row i == program(x[i]).
    try:
        got = (
            evoxc.run(_numpy_executable(merged_identical), evoxc.from_numpy(xs))
            .numpy()
        )
        ref0_exe = _numpy_executable(identical_graph)
        refs = np.stack(
            [evoxc.run(ref0_exe, evoxc.from_numpy(xs[i])).numpy() for i in range(BATCH_N)]
        )
        ident_err = _max_abs_error(got, refs)
        print(f"    row-correctness max_abs_err {ident_err:.3e}")
        if ident_err != 0.0:
            failures.append(f"identical batch_functions rows differ ({ident_err:g})")
    except Exception as exc:  # noqa: BLE001 - record, never hide
        ident_err = None
        failures.append(f"identical batch_functions run failed: {type(exc).__name__}: {exc}")
        print(f"    row-correctness FAILED: {type(exc).__name__}: {exc}")

    # --- 64 branchy programs, switch_ops=True ------------------------------
    branchy_fns = [evoxc.trace(_make_program(k), spec) for k in range(BATCH_N)]
    merged_branchy, branchy_min, branchy_median, _ = _timed(
        merge(branchy_fns, True), REPS
    )
    branchy_report = _report_dict(merged_branchy)
    print(
        f"[B] batch_functions: branchy switch_ops=True N={BATCH_N} merge "
        f"min {branchy_min:.2f} ms / median {branchy_median:.2f} ms"
    )
    print(f"    report: {json.dumps(branchy_report)}")

    # The same row-correctness check; this reference is also reused by the
    # Vulkan section (the merged graph's rows must match the programs).
    try:
        got = evoxc.run(
            _numpy_executable(merged_branchy), evoxc.from_numpy(xs)
        ).numpy()
        refs = []
        for row in range(BATCH_N):
            program = branchy_fns[row]
            exe = _numpy_executable(program)
            refs.append(evoxc.run(exe, evoxc.from_numpy(xs[row])).numpy())
        branchy_refs = np.stack(refs)
        branchy_err = _max_abs_error(got, branchy_refs)
        print(f"    row-correctness max_abs_err {branchy_err:.3e}")
        if branchy_err != 0.0:
            failures.append(f"branchy batch_functions rows differ ({branchy_err:g})")
    except Exception as exc:  # noqa: BLE001
        branchy_refs = None
        branchy_err = None
        failures.append(f"branchy batch_functions run failed: {type(exc).__name__}: {exc}")
        print(f"    row-correctness FAILED: {type(exc).__name__}: {exc}")

    summary["batch_functions"] = {
        "n": BATCH_N,
        "input_shape": list(xs.shape),
        "identical": {
            "merge_min_ms": ident_min,
            "merge_median_ms": ident_median,
            "report": ident_report,
            "row_max_abs_err": ident_err,
        },
        "branchy_switch_ops_true": {
            "merge_min_ms": branchy_min,
            "merge_median_ms": branchy_median,
            "report": branchy_report,
            "row_max_abs_err": branchy_err,
        },
    }
    return {
        "merged_branchy": merged_branchy,
        "branchy_report": branchy_report,
        "branchy_refs": branchy_refs,
        "xs": xs,
    }


def run_cpu_end_to_end(evoxc, summary, failures, seed):
    import numpy as np

    per_sample = _make_per_sample_model()
    x_spec = evoxc.TensorSpec((4, 8, 16), evoxc.float32)
    w_spec = evoxc.TensorSpec((16, 8), evoxc.float32)
    b_spec = evoxc.TensorSpec((8,), evoxc.float32)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 8, 16)).astype(np.float32)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    b = rng.standard_normal((8,)).astype(np.float32)
    reference = _per_sample_reference(x, w, b)

    # trace -> lower -> compile -> load, each stage timed explicitly.
    graph, trace_min, trace_median, _ = _timed(
        lambda: evoxc.vmap(per_sample, in_axes=(0, None, None), out_axes=0)(
            x_spec, w_spec, b_spec
        ),
        REPS,
    )
    start = time.perf_counter()
    lowered = evoxc.lower(graph)
    lower_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    artifact = evoxc.compile(lowered)
    compile_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    executable = evoxc.load(artifact)
    load_ms = (time.perf_counter() - start) * 1e3

    inputs = tuple(evoxc.from_numpy(array) for array in (x, w, b))

    def run_once():
        out = evoxc.run(executable, *inputs)
        executable.synchronize()
        return out

    result, run_min, run_median, _ = _timed(run_once, WARM)
    parity = _max_abs_error(result.numpy(), reference)
    print("\n[C] CPU end-to-end (backend='numpy', vmap per-sample graph)")
    print(f"    trace/vmap  min {trace_min:.2f} ms  median {trace_median:.2f} ms")
    print(f"    lower {lower_ms:.3f} ms | compile {compile_ms:.3f} ms | "
          f"load {load_ms:.3f} ms")
    print(f"    warm run (n={WARM}) min {run_min:.3f} ms  median {run_median:.3f} ms")
    print(f"    numpy parity max_abs_err {parity:.3e}")
    if parity > 1e-6:
        failures.append(f"CPU parity exceeded 1e-6: {parity:g}")
    if result.device.kind != "cpu":
        failures.append(f"CPU numpy output not on cpu: {result.device}")

    summary["cpu_end_to_end"] = {
        "backend": "numpy",
        "graph": "vmap(per_sample) [dot + elementwise + reduction]",
        "input_shapes": [list(x.shape), list(w.shape), list(b.shape)],
        "trace_min_ms": trace_min,
        "trace_median_ms": trace_median,
        "lower_ms": lower_ms,
        "compile_ms": compile_ms,
        "load_ms": load_ms,
        "warm_n": WARM,
        "run_min_ms": run_min,
        "run_median_ms": run_median,
        "max_abs_err": parity,
        "output_device": str(result.device),
    }


def run_vulkan_end_to_end(evoxc, summary, failures, seed):
    import numpy as np

    # Device enumeration + provenance.  iree.runtime is the environment's
    # device table, used only to NAME the GPU (evidence, not an evoxc API).
    import iree.runtime as rt

    infos = rt.get_driver("vulkan").query_available_devices()
    names = [info.get("name", "<unnamed>") for info in infos]
    print(f"\n[D] Vulkan enumeration: {json.dumps(names)}")
    if not infos:
        raise RuntimeError(
            "iree.runtime.get_driver('vulkan').query_available_devices() "
            "returned no devices (no usable Vulkan device/ICD visible to the "
            "loader; check VK_ICD_FILENAMES and `evoxc.devices('vulkan')`)"
        )
    intel = [index for index, info in enumerate(infos) if "Intel" in info.get("name", "")]
    index = intel[0] if intel else 0
    if not intel:
        warnings.warn(
            "no Intel device in the Vulkan enumeration; using index "
            f"{index} ({names[index]})",
            stacklevel=2,
        )
    device = evoxc.Device("vulkan", index)
    enumerated = evoxc.devices("vulkan")
    print(f"    selected {device} ({names[index]})")
    print(f"    evoxc.devices('vulkan') -> {enumerated}")
    if device not in enumerated:
        failures.append(
            f"evoxc.devices('vulkan') does not contain the selected {device}"
        )

    # The SAME vmap per-sample graph as the CPU leg.
    per_sample = _make_per_sample_model()
    x_spec = evoxc.TensorSpec((4, 8, 16), evoxc.float32)
    w_spec = evoxc.TensorSpec((16, 8), evoxc.float32)
    b_spec = evoxc.TensorSpec((8,), evoxc.float32)
    graph = evoxc.vmap(per_sample, in_axes=(0, None, None), out_axes=0)(
        x_spec, w_spec, b_spec
    )
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 8, 16)).astype(np.float32)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    b = rng.standard_normal((8,)).astype(np.float32)
    reference = _per_sample_reference(x, w, b)

    start = time.perf_counter()
    lowered = evoxc.lower(graph, backend="iree")
    lower_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    artifact = evoxc.compile(
        lowered, backend="iree", target_backends=[_target_for(device)]
    )
    compile_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    executable = evoxc.load(artifact, backend="iree", device=device)
    load_ms = (time.perf_counter() - start) * 1e3
    if artifact.target != "vulkan":
        failures.append(f"artifact target is {artifact.target!r}, expected 'vulkan'")

    cpu = evoxc.Device("cpu", 0)
    placed = tuple(evoxc.from_numpy(array).to(device) for array in (x, w, b))

    # Host-input guard: placement is explicit, never staged at the run boundary.
    try:
        evoxc.run(executable, *[evoxc.from_numpy(array) for array in (x, w, b)])
    except evoxc.core.DeviceError:
        host_rejected = True
    else:
        host_rejected = False
        failures.append("vulkan executable silently accepted host inputs")
    print(f"    host-input guard: rejected={host_rejected}")

    def run_once():
        out = evoxc.run(executable, *placed)
        executable.synchronize()
        return out

    result, run_min, run_median, _ = _timed(run_once, WARM)
    residence = (
        isinstance(result, evoxc.Tensor)
        and not isinstance(result.data, np.ndarray)
        and result.device == device
        and hasattr(result.data, "to_host")
    )
    numpy_raised = False
    try:
        result.numpy()
    except evoxc.core.DeviceError:
        numpy_raised = True
    readback_ok = True
    try:
        actual = result.to(cpu).numpy()
    except Exception:  # noqa: BLE001
        readback_ok = False
        actual = None
    parity = _max_abs_error(actual, reference) if actual is not None else None
    print(
        f"    lower {lower_ms:.2f} ms | compile {compile_ms:.2f} ms | "
        f"load {load_ms:.2f} ms | artifact target {artifact.target}"
    )
    print(f"    warm run (n={WARM}) min {run_min:.3f} ms  median {run_median:.3f} ms")
    print(
        f"    residency={residence} .numpy()_raised_DeviceError={numpy_raised} "
        f".to(cpu)_ok={readback_ok} parity max_abs_err {parity:.3e}"
    )
    if not residence:
        failures.append("vulkan output is not a device-resident payload Tensor")
    if not numpy_raised:
        failures.append(".numpy() on a vulkan output did not raise DeviceError")
    if not readback_ok or parity is None:
        failures.append(".to(cpu) readback failed on a vulkan output")
    elif parity > 1e-4:
        failures.append(f"vulkan parity exceeded 1e-4: {parity:g}")

    summary["vulkan_end_to_end"] = {
        "backend": "iree",
        "target": artifact.target,
        "device": str(device),
        "enumerated_devices": names,
        "graph": "vmap(per_sample) [dot + elementwise + reduction]",
        "lower_ms": lower_ms,
        "compile_ms": compile_ms,
        "load_ms": load_ms,
        "warm_n": WARM,
        "run_min_ms": run_min,
        "run_median_ms": run_median,
        "max_abs_err": parity,
        "device_resident": residence,
        "host_input_rejected": host_rejected,
        "numpy_raises_devicerror": numpy_raised,
        "to_cpu_ok": readback_ok,
    }


def run_native_graph_on_vulkan(evoxc, summary, failures, merge_state, seed):
    """Part (e): the Rust-produced batch_functions graph actually runs on Vulkan."""
    import numpy as np

    import iree.runtime as rt

    infos = rt.get_driver("vulkan").query_available_devices()
    intel = [index for index, info in enumerate(infos) if "Intel" in info.get("name", "")]
    device = evoxc.Device("vulkan", intel[0] if intel else 0)
    merged = merge_state["merged_branchy"]
    refs = merge_state["branchy_refs"]
    xs = merge_state["xs"]

    start = time.perf_counter()
    lowered = evoxc.lower(merged, backend="iree")
    lower_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    artifact = evoxc.compile(
        lowered, backend="iree", target_backends=[_target_for(device)]
    )
    compile_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    executable = evoxc.load(artifact, backend="iree", device=device)
    load_ms = (time.perf_counter() - start) * 1e3

    placed = evoxc.from_numpy(xs).to(device)

    def run_once():
        out = evoxc.run(executable, placed)
        executable.synchronize()
        return out

    result, run_min, run_median, _ = _timed(run_once, WARM)
    residence = (
        isinstance(result, evoxc.Tensor)
        and not isinstance(result.data, np.ndarray)
        and result.device == device
    )
    numpy_raised = False
    try:
        result.numpy()
    except evoxc.core.DeviceError:
        numpy_raised = True
    parity = None
    readback_ok = True
    try:
        actual = result.to(evoxc.Device("cpu", 0)).numpy()
        if refs is not None:
            parity = _max_abs_error(actual, refs)
    except Exception as exc:  # noqa: BLE001
        readback_ok = False
        failures.append(f"batch_functions Vulkan readback failed: {type(exc).__name__}: {exc}")

    print("\n[E] native batch_functions graph on Vulkan (Rust-produced graph)")
    print(
        f"    target {artifact.target} | lower {lower_ms:.2f} ms | "
        f"compile {compile_ms:.2f} ms | load {load_ms:.2f} ms"
    )
    print(f"    warm run (n={WARM}) min {run_min:.3f} ms  median {run_median:.3f} ms")
    print(
        f"    residency={residence} .numpy()_raised_DeviceError={numpy_raised} "
        f"parity max_abs_err {parity if parity is None else format(parity, '.3e')}"
    )
    if not residence:
        failures.append("batch_functions Vulkan output is not device-resident")
    if not numpy_raised:
        failures.append(".numpy() on a batch_functions Vulkan output did not raise")
    if parity is None:
        failures.append("no batch_functions Vulkan parity measured")
    elif parity > 1e-4:
        failures.append(f"batch_functions Vulkan parity exceeded 1e-4: {parity:g}")

    summary["vulkan_native_batch_functions"] = {
        "backend": "iree",
        "target": artifact.target,
        "device": str(device),
        "input_shape": list(xs.shape),
        "lower_ms": lower_ms,
        "compile_ms": compile_ms,
        "load_ms": load_ms,
        "warm_n": WARM,
        "run_min_ms": run_min,
        "run_median_ms": run_median,
        "max_abs_err": parity,
        "device_resident": residence,
        "numpy_raises_devicerror": numpy_raised,
        "report": merge_state["branchy_report"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument(
        "--skip-vulkan",
        action="store_true",
        help="run the CPU half only (host without a usable Vulkan device)",
    )
    args = parser.parse_args(argv)

    import evoxc

    versions = _versions()
    print("== native transform engine probe ==")
    print(f"seed={args.seed} python={versions['python']} numpy={versions['numpy']}")
    print(
        f"evoxc={versions['evoxc']} native={versions['native_version']} "
        f"(available={versions['native_available']})"
    )
    print(
        f"iree-base-compiler={versions['iree-base-compiler']} "
        f"iree-base-runtime={versions['iree-base-runtime']}"
    )
    print(f"platform={versions['platform']}")
    print(f"cpu={versions['cpu_model']}")
    print(f"loadavg(start)={_loadavg()}")

    summary = {"seed": args.seed, "versions": versions, "failures": []}
    failures = summary["failures"]
    if not versions["native_available"]:
        failures.append(
            "evoxc._evoxc_native is not available; build it with scripts/build-native.sh"
        )

    run_transform_costs(evoxc, summary, failures)
    merge_state = run_batch_functions(evoxc, summary, failures, args.seed)
    run_cpu_end_to_end(evoxc, summary, failures, args.seed)

    if args.skip_vulkan:
        print("\n[D/E] skipped (--skip-vulkan)")
        summary["vulkan_skipped"] = True
    else:
        summary["vulkan_skipped"] = False
        try:
            run_vulkan_end_to_end(evoxc, summary, failures, args.seed)
            run_native_graph_on_vulkan(
                evoxc, summary, failures, merge_state, args.seed
            )
        except Exception as exc:  # noqa: BLE001 - diagnose, never hide
            message = f"{type(exc).__name__}: {exc}"
            print(f"\nVULKAN_BLOCKER: {message}")
            failures.append(f"vulkan leg blocked: {message}")

    summary["loadavg_end"] = _loadavg()
    summary["status"] = "PASS" if not failures else "FAIL"
    print(f"\nloadavg(end)={summary['loadavg_end']}")
    print(f"STATUS: {summary['status']}")
    for failure in failures:
        print(f"  FAILURE: {failure}")
    print("SUMMARY_JSON: " + json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
