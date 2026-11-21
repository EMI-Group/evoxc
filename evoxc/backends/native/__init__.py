"""Native CPU backend — executes graphs on the in-process Rust execution engine.

``evoxc.backends.get("native")`` stages a verified ``Graph`` through the SAME
``lower`` → ``compile`` → ``load`` → ``run`` contract as the numpy reference
backend, but the execution engine is the Rust ``evoxc-cpu`` crate reached
through the ``evoxc._evoxc_native`` PyO3 extension (``native_compile``). The
backend is OPTIONAL and lazily activated, exactly like the compiler adapters:
importing ``evoxc`` / ``evoxc.backends`` never imports the extension, and
``register()`` probes it on first use, raising an actionable
``core.BackendError`` when the extension (or its ``native_compile`` entry
point) is unavailable — never a silent fallback to numpy.

Staging (mirrors ``numpy/`` step for step):

- ``lower(graph, options)``: ``graph.verify()`` → capability pre-check (the
  live module walk: ``runtime_call`` / collective effects / ``block_call`` /
  sparse ops / dtypes) → ``inline_portables(..., keep_backend_impls="native")``
  through the SHARED ``../inline.py`` machinery (a block with neither a
  portable decomposition nor a registered ``native`` impl is an explicit
  ``BackendError`` naming the block) → conditional post-inline re-verify +
  re-check → **engine capability gate**: ``native_compile`` the module and
  read its ``stats()`` — the engine is the authority on which ops it
  implements, and any op it reports as unsupported raises
  ``core.BackendError`` naming the ops and pointing at ``backend="numpy"`` →
  record the ``Signature`` from the Graph's LIVE attributes → payload = the
  versioned, self-describing serialized ``ir.Module`` (so lowered programs and
  artifacts stay savable and portable).
- ``compile(lowered, options)``: validate ``lowered.backend == "native"``,
  scan the serialized module for the custom-op names it requires
  (``block_call`` / ``external_call`` / ``runtime_call``), record
  ``target="cpu"`` and the runtime dependencies (``numpy`` + the extension
  version). No lowering work happens here.
- ``load(artifact, device, options)``: accept only a CPU ``Device``; rebuild
  the engine program from the artifact payload (never re-lower); require a
  registered ``native`` impl for every remaining ``block_call`` (missing ⇒
  ``PersistenceError`` naming the block). External kernels and
  ``runtime_call`` callbacks stay LAZY (per the binding persistence contract:
  the registries are never serialized and must be re-established before RUN,
  not before load).

``NativeExecutable`` satisfies the ``Executable`` protocol and adds
``plan_text()`` / ``stats()`` / ``required_ops()`` so the engine's dispatch
plan and compile statistics are reachable for debugging (the user-facing
``Executable.summary()`` inspection API lives in ``evoxc/pipeline.py`` and
delegates to these).

Host-data contract: v1 is CPU-only and host-only. ``run`` receives
ndarray-backed ``core.Tensor`` inputs (host data ⇒ ``Device("cpu", 0)``, the
run-boundary device rule is enforced by ``evoxc.run`` before dispatch) and
returns ``core.Tensor`` outputs wrapping the engine's host arrays. There is no
device payload, no staging tier and no implicit transfer.

Deliberate v1 deferrals (every one an explicit error, never silent):

- sparse ops — ``Capabilities.sparse_ops=False``; the pre-check rejects the 16
  ``sparse_*`` ops naming the op and suggesting ``backend="numpy"`` (sparse is
  numpy-backend-only by contract);
- non-CPU devices — ``load`` raises ``core.BackendError`` naming the kind;
- any op the Rust engine does not implement — reported by the engine at
  ``lower()`` as a ``BackendError`` naming the ops (``backend="numpy"`` is the
  reference fallback);
- ``device_resident=True`` external-kernel slots — rejected by the resolver
  (host arrays cannot honor device-resident semantics);
- ``block_call``s whose impl exists only for another backend — the shared
  inliner requires a ``native`` impl or a portable decomposition.

Import acyclicity (binding, see ``../CONTEXT.md``): top-level imports are
``evoxc.core`` / ``evoxc.ir`` plus this package's siblings; ``evoxc.ops`` /
``evoxc.block`` / ``evoxc.external`` are imported only inside function bodies,
``evoxc.persist`` only inside ``save``/``load``, and never ``evoxc.pipeline``.
The extension itself is imported only inside ``engine.py`` functions.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from evoxc import core
from evoxc import ir
from evoxc.core import Device

from ..backend import Backend, Capabilities
from ..options import validate_options
from ..program import CompiledArtifact, LoweredProgram, Signature
from ..registry import register as _register_backend
from ..inline import inline_portables, iter_ops
from . import engine
from .resolver import NativeCallbackResolver

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from evoxc.core import Tensor
    from evoxc.ir import Module
    from evoxc.trace import Graph

__all__ = [
    "NativeBackend",
    "NativeExecutable",
    "NativeCallbackResolver",
    "native_backend",
    "register",
]

#: No per-stage options exist in v1: the CPU engine is driven entirely by the
#: graph. Keys are validated against the union of these sets (see
#: ``../options.py``), so an option meant for a compiler adapter fails loudly
#: here instead of being silently ignored — pass it to the adapter that owns
#: it (``backend="iree"``/``"xla"``/``"tvm"``).
_NO_OPTIONS: frozenset[str] = frozenset()


def _entry_function(module: "Module") -> ir.Function:
    """The module's entry function (``"main"``, else the single function)."""
    from ..numpy.interpreter import entry_function

    return entry_function(module)


def _record_signature(graph: "Graph", main_fn: ir.Function) -> Signature:
    """The program Signature, recorded EXACTLY like ``NumpyBackend.lower``.

    Every field comes from the Graph's LIVE attributes at ``lower()`` time
    (never re-derived downstream): the input/output TreeSpecs and per-leaf
    specs, the input static values, and the output static values.
    """
    return Signature(
        input_tree=graph.input_specs,
        output_tree=graph.output_tree,
        input_specs=tuple(graph.tensor_specs),
        output_specs=tuple(
            core.TensorSpec(shape=value_type.shape, dtype=value_type.dtype)
            for value_type in main_fn.output_types
        ),
        static_values=tuple(record.value for record in graph.static_values),
        output_static_values=tuple(
            record.value for record in graph.output_static_values
        ),
    )


def _scan_required_ops(module: "Module") -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The custom-op names the serialized module requires.

    Returns ``(blocks, external, runtime_calls)`` — sorted, deduplicated names
    of ``block_call`` / ``external_call`` / ``runtime_call`` ops. This scan is
    authoritative for the artifact's ``required_custom_ops`` (it is exactly
    what will run) and is cross-checked at ``lower()`` against the engine's own
    ``required_ops()`` report.
    """
    blocks: set[str] = set()
    external: set[str] = set()
    runtime_calls: set[str] = set()
    for op in iter_ops(module):
        if op.name == "block_call":
            blocks.add(op.attributes["block_name"])
        elif op.name == "external_call":
            external.add(op.attributes["name"])
        elif op.name == "runtime_call":
            runtime_calls.add(op.attributes["callback"])
    return tuple(sorted(blocks)), tuple(sorted(external)), tuple(sorted(runtime_calls))


class NativeBackend(Backend):
    """Backend executing graphs on the in-process Rust CPU engine.

    Registered lazily (``registry.OPTIONAL_ADAPTERS``): ``register()`` probes
    the extension first, so ``evoxc.backends.get("native")`` fails explicitly
    when the Rust ``evoxc-cpu`` crate is not built into the extension.
    """

    name: ClassVar[str] = "native"
    capabilities: ClassVar[Capabilities] = Capabilities(
        dynamic_shapes=True,  # the engine plans over symbolic dims bound at run time
        dtypes=engine.SUPPORTED_DTYPES,
        collectives=True,  # single-rank identity semantics
        runtime_calls=True,  # Python callbacks via the resolver's call_runtime
        external_calls=True,  # named kernels via the resolver's call_external
        custom_blocks=True,  # registered 'native' block impls via call_block
        async_collectives=False,  # execution is synchronous
        sparse_ops=False,  # sparse stays numpy-backend-only in v1 (explicit deferral)
    )
    KNOWN_OPTIONS: ClassVar[dict[str, frozenset[str]]] = {
        "lower": _NO_OPTIONS,
        "compile": _NO_OPTIONS,
        "load": _NO_OPTIONS,
        "run": _NO_OPTIONS,
    }

    # --------------------------------------------------------------- staging

    def lower(self, graph: "Graph", options: dict | None = None) -> LoweredProgram:
        """Stage a verified Graph into a LoweredProgram (serialized-IR payload).

        Steps (binding order — error precedence follows it):

        1. ``options`` key validation against ``KNOWN_OPTIONS`` (union of the
           stage sets), so an unknown key fails before any graph work.
        2. ``graph.verify()`` — surfaces ``core.VerificationError`` as-is.
        3. Capability pre-check over the CURRENT module state: effects vs
           ``Capabilities`` flags, sparse ops, and every op result dtype.
        4. ``inline_portables(..., keep_backend_impls="native")`` — the shared
           fixpoint driver; blocks with a registered ``native`` impl stay,
           blocks with only a portable decomposition are spliced in, and a
           block with neither is an explicit ``BackendError`` naming it.
        5. When (and only when) step 4 spliced something: re-verify and repeat
           the capability check (the module changed).
        6. ENGINE CAPABILITY GATE — ``native_compile`` the module and read its
           statistics; any op the engine reports as unsupported raises
           ``core.BackendError`` naming the ops and pointing at
           ``backend="numpy"``. This is the honest gate: the Rust engine, not
           this Python package, knows which ops it implements.
        7. Record the ``Signature`` from the Graph's LIVE attributes.
        8. ``payload`` = versioned self-describing
           ``ir.serialize_module(graph.module)``.

        The compiled program from step 6 is kept on the returned object as an
        in-process-only hint (never serialized) so the common
        build/compile/load path does not compile the same graph twice; an
        artifact loaded from disk always recompiles from its payload.
        """
        validate_options(options, self.KNOWN_OPTIONS, self.name, "lower")
        graph.verify()
        self._check_capabilities(graph.module)
        expansions = inline_portables(graph.module, keep_backend_impls="native")
        if expansions:
            graph.verify()  # the module changed — re-verify
            self._check_capabilities(graph.module)
        program = self._probe_engine(graph.module)
        signature = _record_signature(graph, _entry_function(graph.module))
        payload = ir.serialize_module(graph.module)
        lowered = LoweredProgram(backend=self.name, signature=signature, payload=payload)
        # In-process compile hint (documented, never serialized, never trusted
        # across a process boundary): reused by ``compile`` only for the very
        # same payload object.
        lowered._native_program = (payload, program)
        return lowered

    def compile(
        self, lowered: LoweredProgram, options: dict | None = None
    ) -> CompiledArtifact:
        """Wrap the serialized IR into a self-describing CompiledArtifact.

        Validates ``lowered.backend == "native"`` (mismatch ⇒
        ``core.BackendError``), scans the serialized module for the custom-op
        names it requires (``required_custom_ops`` — blocks and external
        kernels; runtime-call callback ids are recorded in
        ``runtime_dependencies``-adjacent diagnostics only through the scan and
        resolve lazily at run time), and records ``target="cpu"`` with runtime
        dependencies ``{"numpy": ..., "evoxc-native": ...}``. No lowering or
        engine work happens here.
        """
        validate_options(options, self.KNOWN_OPTIONS, self.name, "compile")
        if lowered.backend != self.name:
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the native backend"
            )
        module = ir.deserialize_module(lowered.payload)
        blocks, external, runtime_calls = _scan_required_ops(module)
        required_custom_ops = tuple(sorted(set(blocks) | set(external)))
        artifact = CompiledArtifact(
            backend=self.name,
            signature=lowered.signature,
            target="cpu",
            payload=lowered.payload,
            required_custom_ops=required_custom_ops,
            runtime_dependencies={
                "numpy": np.__version__,
                engine.ENGINE_VERSION_KEY: engine.engine_version(),
            },
        )
        hint = getattr(lowered, "_native_program", None)
        if isinstance(hint, tuple) and len(hint) == 2 and hint[0] is lowered.payload:
            # In-process-only compile reuse: the artifact carries the program
            # together with the exact payload object it was built from, so a
            # replaced/mutated payload can never reuse a stale program.
            artifact._native_program = (artifact.payload, hint[1])
        if runtime_calls:
            # Self-description of the callback dependency (never serialized
            # into the program: callbacks are registry-resolved at run time).
            artifact.runtime_call_ids = runtime_calls
        return artifact

    def load(
        self,
        artifact: CompiledArtifact,
        device: Device | None = None,
        options: dict | None = None,
    ) -> "NativeExecutable":
        """Rebuild the engine program from an artifact. Never re-lowers.

        - backend mismatch ⇒ ``core.PersistenceError``;
        - ``device`` must be ``None`` or a CPU ``Device`` (a non-``Device``
          object is a ``core.DeviceError``; a non-CPU kind a
          ``core.BackendError`` naming the kind — v1 is host-only);
        - every remaining ``block_call`` must have a registered ``block`` and
          a registered ``native`` impl ⇒ otherwise ``core.PersistenceError``
          naming the block (portables were inlined at ``lower()``);
        - external kernels and ``runtime_call`` callbacks are NOT required at
          load: the binding persistence contract keeps them lazy and resolves
          them at RUN time with an explicit ``core.BackendError`` naming the
          missing registration.

        The engine program is taken from the in-process hint when the artifact
        still carries the exact payload object it was compiled from; otherwise
        it is built from the payload here (``native_compile``) — never by
        re-lowering the graph.
        """
        validate_options(options, self.KNOWN_OPTIONS, self.name, "load")
        if artifact.backend != self.name:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; the "
                "native backend cannot load it"
            )
        if device is not None:
            if not isinstance(device, Device):
                raise core.DeviceError(
                    f"device must be a core.Device, got {type(device).__name__}"
                )
            if device.kind != "cpu":
                raise core.BackendError(
                    "the native backend v1 supports CPU devices only (it "
                    f"executes host numpy data), got device kind {device.kind!r}"
                )
        module = ir.deserialize_module(artifact.payload)
        self._require_block_impls(module)
        hint = getattr(artifact, "_native_program", None)
        if (
            isinstance(hint, tuple)
            and len(hint) == 2
            and hint[0] is artifact.payload
        ):
            program = hint[1]
        else:
            program = engine.compile_module(module)
        return NativeExecutable(
            program=program,
            signature=artifact.signature,
            artifact=artifact,
            device=device if device is not None else Device("cpu", 0),
        )

    # ------------------------------------------------------------- internals

    def _check_capabilities(self, module: "Module") -> None:
        """Capability drift net over the module's current (live) op set.

        Mirrors the numpy/compiler pre-checks: effect flags from
        ``Capabilities``, sparse ops (numpy-backend-only in v1), and the
        declared dtype set. The engine's own op support is gated separately
        (``_probe_engine``) — that is the Rust side's authority.
        """
        capabilities = self.capabilities
        for function in module.functions:
            for value_type in (*function.input_types, *function.output_types):
                self._check_value_dtype(value_type.dtype, "function signature")
        for op in iter_ops(module):
            if op.is_terminator:
                continue  # 'return' carries no capability risk
            if op.name == "runtime_call" and not capabilities.runtime_calls:
                raise core.BackendError(
                    "capability drift: the native backend cannot execute "
                    "runtime_call"
                )
            if op.name == "block_call" and not capabilities.custom_blocks:
                raise core.BackendError(
                    "capability drift: the native backend cannot execute "
                    "custom block ops (block_call)"
                )
            if op.effect == "collective" and not capabilities.collectives:
                raise core.BackendError(
                    f"capability drift: the native backend cannot execute "
                    f"collective op '{op.name}'"
                )
            if op.opdef.category == "sparse" and not capabilities.sparse_ops:
                raise core.BackendError(
                    f"capability drift: the native backend cannot execute "
                    f"sparse op '{op.name}' — sparse ops are numpy-backend-only "
                    "in v1; run this graph on backend='numpy' (or densify with "
                    "evoxc.sparse.to_dense)"
                )
            for result in op.results:
                self._check_value_dtype(result.type.dtype, f"op '{op.name}'")

    def _check_value_dtype(self, dtype: Any, where: str) -> None:
        """Raise ``core.BackendError`` when ``dtype`` is not declared."""
        if dtype in self.capabilities.dtypes:
            return
        raise core.BackendError(
            f"capability drift: the native backend cannot execute dtype "
            f"{np.dtype(dtype).name} ({where}) — its declared dtypes are "
            f"{sorted(d.name for d in self.capabilities.dtypes)}"
        )

    def _probe_engine(self, module: "Module"):
        """Compile the module on the engine and vet the result.

        Returns the ``NativeProgram`` (kept as an in-process hint). Raises
        ``core.BackendError`` naming every op the engine reports as
        unsupported — with ``backend="numpy"`` as the documented fallback —
        and cross-checks the engine's ``required_ops()`` / ``output_types()``
        reports against the module (a mismatch is an FFI contract drift and
        fails loudly rather than producing wrong results later).
        """
        program = engine.compile_module(module)
        stats = engine.program_stats(program)
        unsupported = engine.unsupported_ops(stats)
        if unsupported:
            raise core.BackendError(
                "the native CPU engine does not implement "
                f"{len(unsupported)} op(s) required by this graph: "
                f"{', '.join(unsupported)} — run this graph on backend="
                "'numpy' (the reference interpreter) or rewrite those ops"
            )
        self._cross_check(program, module)
        return program

    def _cross_check(self, program: Any, module: "Module") -> None:
        """Vet the engine's self-report against the module it compiled."""
        blocks, external, runtime_calls = _scan_required_ops(module)
        required = engine.program_required_ops(program)
        if required is not None:
            expected = {
                "blocks": blocks,
                "external": external,
                "runtime_calls": runtime_calls,
            }
            for key, names in expected.items():
                reported = required.get(key)
                if reported is None:
                    continue
                if tuple(sorted(reported)) != names:
                    raise core.BackendError(
                        f"native engine FFI mismatch: required_ops()[{key!r}] "
                        f"reports {tuple(reported)!r}, the graph's serialized "
                        f"module requires {names!r} — the extension and this "
                        "Python backend disagree"
                    )
        output_types = engine.program_output_types(program)
        if output_types is not None:
            main_fn = _entry_function(module)
            if len(output_types) != len(main_fn.output_types):
                raise core.BackendError(
                    "native engine FFI mismatch: the engine reports "
                    f"{len(output_types)} output(s), the entry function "
                    f"declares {len(main_fn.output_types)}"
                )
            from evoxc._native_codec import decode_type

            for index, (encoded, declared) in enumerate(
                zip(output_types, main_fn.output_types)
            ):
                decoded = decode_type(encoded)
                if np.dtype(decoded.dtype) != np.dtype(declared.dtype):
                    raise core.BackendError(
                        "native engine FFI mismatch: output "
                        f"{index} dtype is {np.dtype(decoded.dtype).name} on "
                        f"the engine and {np.dtype(declared.dtype).name} in "
                        "the graph"
                    )

    def _require_block_impls(self, module: "Module") -> None:
        """Every remaining ``block_call`` needs a registered ``native`` impl.

        Portables were inlined at ``lower()``, so a block surviving into the
        artifact must be dispatched at run time; a missing block/impl is a
        ``core.PersistenceError`` naming it — never a silent re-lower and
        never a fallback to another backend's impl.
        """
        blocks, _, _ = _scan_required_ops(module)
        if not blocks:
            return
        from evoxc.block import BlockError
        from evoxc.block import registry as block_registry

        for block_name in blocks:
            try:
                block_registry.get_block(block_name)
            except BlockError as exc:
                raise core.PersistenceError(
                    f"artifact requires custom block {block_name!r}, which is "
                    "not registered in this process"
                ) from exc
            if block_registry.get_impl(block_name, self.name) is None:
                raise core.PersistenceError(
                    f"artifact requires custom block {block_name!r}, but no "
                    "native impl is registered in this process — register one "
                    f"with BlockOp.impl('native', fn) (or run the graph on the "
                    "backend that owns the impl)"
                )


class NativeExecutable:
    """Backend executable running the graph on the Rust CPU engine.

    Satisfies the ``Executable`` protocol (``run`` / ``functions`` /
    ``device``; ``save``/``load`` per the artifact persistence contract) and
    adds engine diagnostics.

    Attributes:
        backend_name: ``"native"`` — the pipeline uses it for run-stage
            option/env resolution and ``Executable.backend``.
        functions: the compiled entry function name (the engine compiles ONE
            entry program; ``NativeProgram.entry()`` is authoritative).
        device: the executable's device (v1: CPU only).
        signature: input/output contract recorded at ``lower()`` time.
        artifact: the underlying ``CompiledArtifact`` — what ``save`` persists.
    """

    backend_name = engine.BACKEND_NAME

    def __init__(
        self,
        program: Any,
        signature: Signature | None = None,
        artifact: CompiledArtifact | None = None,
        device: Device | None = None,
    ) -> None:
        self._program = program
        self.signature = signature
        self.artifact = artifact
        self.device = device
        self.functions = (engine.program_entry(program),)
        self._resolver = NativeCallbackResolver()

    # ------------------------------------------------------------------- run

    def run(
        self, flat_input_tensors: list["Tensor"], options: dict | None = None
    ) -> list["Tensor"]:
        """Execute the program on flat host input tensors, returning flat outputs.

        Inputs are ``core.Tensor``s backed by host numpy arrays (the run
        boundary in ``evoxc.run`` has already enforced the device rule; this
        re-checks defensively — a non-CPU tensor raises ``core.DeviceError``
        because the native backend never transfers data implicitly). The
        engine receives plain arrays and returns plain arrays; each output is
        wrapped in a host ``core.Tensor`` and validated (count, dtype, and
        static shape dims) against the recorded signature — a mismatch is a
        ``core.BackendError``/``core.ShapeError``, never silently accepted.

        Execution is synchronous (``run`` returns after the work completes),
        so ``synchronize()`` is a documented no-op.
        """
        validate_options(options, NativeBackend.KNOWN_OPTIONS, self.backend_name, "run")
        inputs = self._validated_inputs(flat_input_tensors)
        outputs = engine.program_run(self._program, inputs, self._resolver)
        return self._wrap_outputs(outputs)

    def synchronize(self) -> None:
        """Documented no-op: the native engine's ``run`` already blocks.

        Present so the uniform ``Executable.synchronize()`` call shape works
        (a no-op on backends whose ``run`` completes the work — numpy, iree,
        native); output values are valid before the call either way.
        """
        return None

    # ------------------------------------------------------------ diagnostics

    def plan_text(self) -> str:
        """The engine's human-readable dispatch plan (debugging aid)."""
        return engine.program_plan_text(self._program)

    def stats(self) -> dict:
        """The engine's compile/plan statistics (ops, dispatches, compile_ms, …)."""
        return engine.program_stats(self._program)

    def required_ops(self) -> Any:
        """The engine's required-ops report (``None`` when unavailable)."""
        return engine.program_required_ops(self._program)

    # ------------------------------------------------------------ persistence

    def save(self, path: str | os.PathLike) -> None:
        """Save the underlying CompiledArtifact (reconstruct EXPLICITLY at load).

        Device handles and the compiled native program are never serialized:
        ``NativeExecutable.load`` re-runs ``native_compile`` from the artifact
        payload.
        """
        if self.artifact is None:
            raise core.BackendError(
                "this NativeExecutable has no CompiledArtifact to save — "
                "construct it via NativeBackend.load(artifact) or pass "
                "artifact= to the constructor"
            )
        self.artifact.save(path)

    @classmethod
    def load(
        cls, path: str | os.PathLike, device: Device | None = None
    ) -> "NativeExecutable":
        """Reconstruct an executable from a saved artifact (never re-lowers)."""
        artifact = CompiledArtifact.load(path)
        return native_backend.load(artifact, device)

    # ------------------------------------------------------------- internals

    def _validated_inputs(self, flat_input_tensors: list["Tensor"]) -> list:
        """Validate the flat host inputs and return the engine's array list."""
        if not isinstance(flat_input_tensors, (list, tuple)):
            raise TypeError(
                "NativeExecutable.run expects a list of core.Tensor inputs, "
                f"got {type(flat_input_tensors).__name__}"
            )
        specs = self.signature.input_specs if self.signature is not None else ()
        if specs and len(flat_input_tensors) != len(specs):
            raise core.BackendError(
                f"native executable expects {len(specs)} input tensor(s), got "
                f"{len(flat_input_tensors)}"
            )
        arrays = []
        for index, tensor in enumerate(flat_input_tensors):
            if not isinstance(tensor, core.Tensor):
                raise TypeError(
                    f"input {index} must be a core.Tensor, got "
                    f"{type(tensor).__name__} — wrap host arrays with "
                    "evoxc.from_numpy"
                )
            device = tensor.device
            if device is not None and device.kind != "cpu":
                raise core.DeviceError(
                    f"input {index} lives on device {device!r}; the native "
                    "backend executes host data only — transfer it explicitly "
                    "with tensor.to(core.Device('cpu', 0)) first (no implicit "
                    "device→host transfer)"
                )
            arrays.append(tensor.numpy())
        return arrays

    def _wrap_outputs(self, outputs: Any) -> list["Tensor"]:
        """Wrap the engine's host arrays in ``core.Tensor``s and validate them."""
        if not isinstance(outputs, (list, tuple)):
            raise core.BackendError(
                "the native CPU engine returned "
                f"{type(outputs).__name__} where a list of output arrays was "
                "expected"
            )
        outputs = list(outputs)
        specs = self.signature.output_specs if self.signature is not None else ()
        if specs and len(outputs) != len(specs):
            raise core.BackendError(
                f"the native CPU engine returned {len(outputs)} output(s), the "
                f"signature declares {len(specs)}"
            )
        tensors = []
        for index, array in enumerate(outputs):
            if not isinstance(array, np.ndarray):
                raise core.BackendError(
                    f"the native CPU engine returned {type(array).__name__} for "
                    f"output {index}, expected a numpy array"
                )
            if index < len(specs):
                self._check_output_spec(index, array, specs[index])
            tensors.append(core.Tensor(array))
        return tensors

    @staticmethod
    def _check_output_spec(index: int, array: np.ndarray, spec: Any) -> None:
        """Validate one output against its recorded ``TensorSpec``.

        dtype must match exactly (``core.BackendError``); shape is compared
        only on the dims the spec declares concretely (a symbolic or ``None``
        dim is runtime-dynamic and unchecked) — ``core.ShapeError`` otherwise.
        """
        if np.dtype(spec.dtype) != array.dtype:
            raise core.BackendError(
                f"output {index} dtype is {array.dtype.name} from the native "
                f"engine but {np.dtype(spec.dtype).name} in the graph signature"
            )
        declared = tuple(spec.shape or ())
        if not declared:
            if array.ndim != 0:
                raise core.ShapeError(
                    f"output {index} is rank {array.ndim} from the native "
                    "engine but the graph declares a scalar"
                )
            return
        if array.ndim != len(declared):
            raise core.ShapeError(
                f"output {index} rank is {array.ndim} from the native engine "
                f"but the graph declares rank {len(declared)}"
            )
        for axis, (dim, actual) in enumerate(zip(declared, array.shape)):
            if isinstance(dim, int) and not isinstance(dim, bool) and dim != actual:
                raise core.ShapeError(
                    f"output {index} axis {axis} is {actual} from the native "
                    f"engine but the graph declares {dim}"
                )


native_backend = NativeBackend()


def register() -> NativeBackend:
    """Probe the extension and register the singleton (idempotent).

    Called by ``registry.get("native")`` on first use (``OPTIONAL_ADAPTERS``
    auto-activation) and when loading a persisted ``native`` artifact. Raises
    ``core.BackendError`` with build instructions when the extension is
    missing/broken or was built without ``native_compile`` (the Rust
    ``evoxc-cpu`` crate); a successful probe registers the instance (the
    registry tolerates idempotent re-registration).

    Collectives need no installation here: the process-wide single-rank
    identity executor in ``evoxc.dist.context`` is installed by
    ``evoxc.backends.numpy`` at import (always, since numpy is the default
    backend), and the native engine's collective support is single-rank
    identity, so it uses that same contract.
    """
    engine.engine_module()
    return _register_backend(native_backend)
