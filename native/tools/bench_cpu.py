"""Benchmark: native CPU engine vs the numpy backend, plus native compile time.

Seven workloads, each one ``@evoxc.defn`` graph:

``ew_chain_1e6``    an 8-op ARITHMETIC elementwise f32 chain over 1e6 elements
``ew_transcendental_1e6``  four transcendental ops (tanh/sigmoid/gelu/erf) over
                    1e6 f32 elements — the scalar-libm counterpart
``dot_sum_256``     ``dot((256,256), (256,256))`` + ``reduce_sum`` (f32)
``gp_gather_4096``  a gather/select-heavy GP-shaped chain over (4096, 64) f32,
                    built by a Python loop inside the defn over static indices
``gp_sort_4096``    a selection-shaped GP graph over (4096, 64) f32: row argsort,
                    a batched gather of the top-16 columns, a reduction and a
                    broadcast select against the row mean
``tf_softmax_8x128x384``  a transformer-shaped attention block over
                    (8, 128, 384) f32: row max/sum shifts, exp, tanh and gelu
``chain_2000``      a ~2000-op elementwise chain over a (64,) f32 array

For each workload the script measures

* ``native_compile_ms`` — the wall time of one
  ``evoxc._evoxc_native.native_compile(envelope)`` call (envelope decode +
  ``evoxc_cpu::plan::compile``), median of ``--compile-repeats`` runs on an
  envelope encoded ONCE up front (the Python trace/encode cost is reported
  separately as ``encode_ms``);
* ``native_run_ms`` — ``NativeProgram.run([...])``, median of ``--iterations``
  calls after ``--warmup`` warmup calls;
* ``numpy_run_ms`` — ``Executable.run`` of an executable built ONCE with
  ``evoxc.build(fn, *specs, backend="numpy", device="cpu")``, same warmup /
  iteration schedule.  (``evoxc.evaluate`` is NOT used: it re-derives specs,
  traces, lowers, compiles and loads on every call, so it cannot measure a
  warm call.)

Both run columns include the argument/result marshalling each public API
performs (numpy arrays ⇄ native buffers on one side, Tensor wrapping on the
other), because that is what a caller pays per call.

Run it::

    source scripts/dev-env.sh
    .venv/bin/python native/tools/bench_cpu.py [--iterations 25] [--warmup 5]
                                              [--compile-repeats 5] [--filter SUBSTR]

Exit status: 0 when every workload ran and the 2000-op compile-target check
passed (``<= 2 ms``), 1 otherwise, 2 when the extension is not built.  The last
lines are a copy-pasteable ``MEMO:`` line and the compile-target verdict.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

import evoxc
from evoxc._native_codec import encode_module

#: The documented compile-time target for the 2000-op graph (milliseconds).
COMPILE_TARGET_MS = 2.0


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


@evoxc.defn
def ew_chain_1e6(x, y):
    """Eight ARITHMETIC elementwise ops over a 1e6-element f32 array.

    Deliberately transcendental-free so the row measures the engine's
    elementwise dispatch + memory path rather than scalar libm.
    """
    a = evoxc.multiply(x, y)
    b = evoxc.add(a, x)
    c = evoxc.power(b, 2.0)
    d = evoxc.multiply(c, 0.5)
    e = evoxc.subtract(d, y)
    f = evoxc.divide(e, 3.0)
    g = evoxc.add(f, evoxc.maximum(x, y))
    return evoxc.minimum(g, evoxc.add(x, 2.0))


@evoxc.defn
def ew_transcendental_1e6(x):
    """Four transcendental ops over 1e6 f32 elements (scalar-libm probe)."""
    return evoxc.add(
        evoxc.tanh(x), evoxc.add(evoxc.sigmoid(x), evoxc.add(evoxc.gelu(x), evoxc.erf(x)))
    )


@evoxc.defn
def dot_sum_256(x, w):
    """A (256,256) @ (256,256) f32 dot plus a full reduction."""
    return evoxc.add(evoxc.dot(x, w), evoxc.reduce_sum(x, axes=(1,), keepdims=True))


@evoxc.defn
def gp_gather_4096(x):
    """Gather/select-heavy GP-shaped chain over (4096, 64) f32.

    Eight rounds of: gather 64 rows at static indices → row-wise sum →
    where-style select against that sum → scale on one branch → tile back to
    (4096, 64) → accumulate.  Every index vector is static Python data, so the
    graph is fixed while the row gathers dominate the op mix.
    """
    acc = x
    rows, width = 4096, 64  # the traced spec is static (4096, 64)
    for step in range(8):
        indices = evoxc.constant(
            evoxc.from_numpy(
                np.array(
                    [(step * 37 + position * 11) % rows for position in range(width)],
                    np.int32,
                )
            )
        )
        gathered = evoxc.gather(x, indices, axis=0)
        row_sum = evoxc.reduce_sum(gathered, axes=(1,), keepdims=True)
        mask = evoxc.greater(row_sum, 0.0)
        branch = evoxc.select(mask, gathered, evoxc.multiply(gathered, -1.0))
        acc = evoxc.add(acc, evoxc.tile(branch, (rows // width, 1)))
    return acc


@evoxc.defn
def gp_sort_4096(x):
    """Selection-shaped GP graph over (4096, 64) f32.

    Row argsort + row sort, a slice of the best 16 columns per row, two
    per-row reductions, a whole-row gather of the parent candidates and a
    broadcast select of a (4096, 1) score against the (4096, 64) population.
    That is the op mix a genetic program's selection step actually produces:
    sorting, a multi-axis gather, keepdims reductions and a broadcast select.
    """
    row_count, width = 4096, 64
    pick = evoxc.constant(
        evoxc.from_numpy(np.array([(step * 29) % row_count for step in range(width)], np.int32))
    )
    order = evoxc.argsort(x, axis=1)  # (4096, 64) int32
    values = evoxc.sort(x, axis=1)  # (4096, 64) f32, ascending
    head = evoxc.slice(values, (0, 0), (row_count, 16))  # (4096, 16) f32
    score = evoxc.reduce_sum(head, axes=(1,), keepdims=True)  # (4096, 1)
    rank = evoxc.cast(order, evoxc.float32)
    row_rank = evoxc.reduce_sum(rank, axes=(1,), keepdims=True)  # (4096, 1)
    parents = evoxc.gather(x, pick, axis=0)  # (64, 64) f32
    bonus = evoxc.reduce_sum(parents, axes=(0,), keepdims=True)  # (1, 64)
    kept = evoxc.select(evoxc.greater(score, row_rank), x, evoxc.multiply(x, 0.5))
    return evoxc.add(kept, bonus)


@evoxc.defn
def tf_softmax_8x128x384(x):
    """Transformer-shaped attention block over (8, 128, 384) f32.

    A per-row max-shifted softmax (reduce_max -> subtract -> exp -> reduce_sum
    -> divide) followed by the tanh/gelu elementwise mix a real block applies
    to its activations.  This is the "many mid-size 3-D tensors" shape rather
    than one flat array.
    """
    row_max = evoxc.reduce_max(x, axes=(2,), keepdims=True)
    weights = evoxc.exp(evoxc.subtract(x, row_max))
    total = evoxc.reduce_sum(weights, axes=(2,), keepdims=True)
    probs = evoxc.divide(weights, total)
    mixed = evoxc.add(evoxc.multiply(probs, 1.5), evoxc.tanh(evoxc.multiply(x, 0.5)))
    return evoxc.add(mixed, evoxc.gelu(evoxc.multiply(x, 0.25)))


@evoxc.defn
def chain_2000(x):
    """A ~2000-op elementwise chain over a (64,) f32 array.

    1000 iterations of ``add(value, 0.5)``; each scalar operand is a Constant
    op, so the graph carries ~2000 IR ops.
    """
    value = x
    for index in range(1000):
        value = evoxc.add(value, 0.5 + index * 1e-4)
    return value


@dataclass
class Workload:
    name: str
    fn: Any
    specs: tuple
    inputs: tuple
    note: str = ""


def build_workloads() -> list:
    rng = np.random.default_rng(20260202)
    f32 = evoxc.float32
    big = rng.standard_normal(1_000_000).astype(np.float32)
    big_other = rng.standard_normal(1_000_000).astype(np.float32)
    mat = rng.standard_normal((256, 256)).astype(np.float32)
    mat_other = rng.standard_normal((256, 256)).astype(np.float32)
    rows = rng.standard_normal((4096, 64)).astype(np.float32)
    activations = rng.standard_normal((8, 128, 384)).astype(np.float32)
    small = rng.standard_normal(64).astype(np.float32)
    return [
        Workload("ew_chain_1e6", ew_chain_1e6, (evoxc.TensorSpec((1_000_000,), f32),) * 2, (big, big_other)),
        Workload(
            "ew_transcendental_1e6",
            ew_transcendental_1e6,
            (evoxc.TensorSpec((1_000_000,), f32),),
            (big,),
            note="tanh/sigmoid/gelu/erf: scalar libm, not the memory path",
        ),
        Workload("dot_sum_256", dot_sum_256, (evoxc.TensorSpec((256, 256), f32),) * 2, (mat, mat_other)),
        Workload("gp_gather_4096", gp_gather_4096, (evoxc.TensorSpec((4096, 64), f32),), (rows,)),
        Workload(
            "gp_sort_4096",
            gp_sort_4096,
            (evoxc.TensorSpec((4096, 64), f32),),
            (rows,),
            note="row argsort + two-axis gather + broadcast select",
        ),
        Workload(
            "tf_softmax_8x128x384",
            tf_softmax_8x128x384,
            (evoxc.TensorSpec((8, 128, 384), f32),),
            (activations,),
            note="per-row softmax + tanh/gelu mix, many mid-size 3-D tensors",
        ),
        Workload(
            "chain_2000",
            chain_2000,
            (evoxc.TensorSpec((64,), f32),),
            (small,),
            note="the compile-time target workload",
        ),
    ]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    workload: Workload
    encode_ms: float
    native_compile_ms: float
    engine_compile_ms: float
    native_run_ms: float
    numpy_run_ms: float
    ops: int
    outputs_match: bool

    @property
    def speedup(self) -> float:
        return self.numpy_run_ms / self.native_run_ms if self.native_run_ms > 0 else float("inf")


def _median_ms(call, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def measure(extension, workload: Workload, args) -> Measurement:
    """Encode/trace once, then time compile and the two warm run loops."""
    flat = [np.ascontiguousarray(value) for value in workload.inputs]

    started = time.perf_counter()
    graph = evoxc.trace(workload.fn, *workload.specs)
    envelope = encode_module(graph.module)
    encode_ms = (time.perf_counter() - started) * 1000.0

    compile_samples = []
    program = None
    for _ in range(args.compile_repeats):
        started = time.perf_counter()
        program = extension.native_compile(envelope)
        compile_samples.append((time.perf_counter() - started) * 1000.0)
    native_compile_ms = statistics.median(compile_samples)
    engine_compile_ms = float(program.stats()["compile_ms"])

    native_reference = [np.asarray(out) for out in program.run(flat)]
    for _ in range(args.warmup):
        program.run(flat)
    native_run_ms = _median_ms(lambda: program.run(flat), args.iterations)

    executable = evoxc.build(workload.fn, *workload.specs, backend="numpy", device="cpu")
    numpy_reference = [np.asarray(leaf.numpy()) for leaf in evoxc.core.tree_leaves(evoxc.run(executable, *workload.inputs))]
    for _ in range(args.warmup):
        evoxc.run(executable, *workload.inputs)
    numpy_run_ms = _median_ms(lambda: evoxc.run(executable, *workload.inputs), args.iterations)

    match = len(native_reference) == len(numpy_reference) and all(
        got.shape == want.shape
        and got.dtype == want.dtype
        and np.allclose(got, want, rtol=1e-3, atol=1e-4, equal_nan=True)
        for got, want in zip(native_reference, numpy_reference)
    )
    return Measurement(
        workload,
        encode_ms,
        native_compile_ms,
        engine_compile_ms,
        native_run_ms,
        numpy_run_ms,
        int(program.stats()["ops"]),
        match,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def print_environment() -> None:
    load = ", ".join(f"{value:.2f}" for value in os.getloadavg())
    print(f"host cpu     : {cpu_model()}")
    print(f"python/numpy : {platform.python_version()} / {np.__version__}  ({platform.machine()})")
    print(f"loadavg      : {load}   cpus={os.cpu_count()}  time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_table(measurements: Sequence[Measurement], args) -> None:
    header = f"{'workload':18} {'ops':>5} {'native_compile_ms':>17} {'native_run_ms':>14} {'numpy_run_ms':>13} {'speedup':>8}  match"
    print(header)
    print("-" * len(header))
    for item in measurements:
        print(
            f"{item.workload.name:18} {item.ops:5d} {item.native_compile_ms:17.3f} "
            f"{item.native_run_ms:14.3f} {item.numpy_run_ms:13.3f} {item.speedup:7.2f}x  "
            f"{'yes' if item.outputs_match else 'NO'}"
        )
    print()
    print(
        "columns: native_compile_ms = one native_compile(envelope) call; native_run_ms / numpy_run_ms = "
        f"median warm run over {args.iterations} iterations after {args.warmup} warmup calls."
    )
    for item in measurements:
        print(
            f"  {item.workload.name}: trace+encode {item.encode_ms:.1f} ms, "
            f"engine-measured plan compile {item.engine_compile_ms:.3f} ms"
            + (f" — {item.workload.note}" if item.workload.note else "")
        )


def print_compile_target(measurements: Sequence[Measurement]) -> bool:
    chain = next((item for item in measurements if item.workload.name == "chain_2000"), None)
    if chain is None:
        print("COMPILE TARGET: chain_2000 not measured (filtered out)")
        return True
    verdict = "PASS" if chain.engine_compile_ms <= COMPILE_TARGET_MS else "MISS"
    decode_ms = chain.native_compile_ms - chain.engine_compile_ms
    print(
        f"COMPILE TARGET: chain_2000 ({chain.ops} ops) engine plan compile = {chain.engine_compile_ms:.3f} ms "
        f"(target <= {COMPILE_TARGET_MS:.0f} ms): {verdict}"
    )
    print(
        f"COMPILE TARGET: chain_2000 full native_compile = {chain.native_compile_ms:.3f} ms "
        f"(engine compile {chain.engine_compile_ms:.3f} ms + envelope decode/FFI {decode_ms:.3f} ms), "
        f"Python trace+encode {chain.encode_ms:.1f} ms"
    )
    return chain.engine_compile_ms <= COMPILE_TARGET_MS


def print_memo(measurements: Sequence[Measurement]) -> None:
    parts = [f"{item.workload.name} native {item.native_run_ms:.2f} ms vs numpy {item.numpy_run_ms:.2f} ms ({item.speedup:.1f}x)" for item in measurements]
    chain = next((item for item in measurements if item.workload.name == "chain_2000"), None)
    compile_part = f"; chain_2000 native_compile {chain.native_compile_ms:.2f} ms" if chain else ""
    print()
    print(
        f"MEMO: evoxc-cpu vs numpy on {cpu_model()} [loadavg {', '.join(f'{v:.2f}' for v in os.getloadavg())}] — "
        + "; ".join(parts)
        + compile_part
        + "."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=25, help="timed run iterations per workload (default 25)")
    parser.add_argument("--warmup", type=int, default=5, help="warmup runs per workload (default 5)")
    parser.add_argument("--compile-repeats", type=int, default=5, help="native_compile repetitions (default 5)")
    parser.add_argument("--filter", default=None, help="run only workloads whose name contains this substring")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.compile_repeats < 1:
        parser.error("--iterations/--compile-repeats must be >= 1 and --warmup >= 0")

    import evoxc._native as native_module

    if not native_module.available():
        print("BENCH: the native extension is not built — run scripts/build-native.sh", file=sys.stderr)
        return 2
    extension = native_module.require()

    workloads = build_workloads()
    if args.filter:
        workloads = [item for item in workloads if args.filter in item.name]
        if not workloads:
            print(f"BENCH: no workload matches --filter {args.filter!r}", file=sys.stderr)
            return 2

    print_environment()
    measurements = []
    for workload in workloads:
        print(f"[bench] running {workload.name} ...", file=sys.stderr, flush=True)
        measurements.append(measure(extension, workload, args))

    print_table(measurements, args)
    target_ok = print_compile_target(measurements)
    print_memo(measurements)
    matches = all(item.outputs_match for item in measurements)
    if not matches:
        print("BENCH: a workload's native outputs did not match the numpy backend", file=sys.stderr)
    return 0 if (target_ok and matches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
