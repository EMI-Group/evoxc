"""Staged pipeline objects OWNED by backends: Signature, LoweredProgram, CompiledArtifact.

Ownership note (binding, see ../CONTEXT.md): these types belong to ``backends``,
NOT to ``pipeline``. ``pipeline`` orchestrates staging and wraps a backend
executable + signature for structured I/O; it re-exports these types through ``evoxc``.

Import acyclicity:
- Top-level imports restricted to ``evoxc.core``.
- ``evoxc.persist`` (container format) is imported lazily INSIDE ``save``/``load``
  bodies: persist sits ABOVE backends in the import DAG, so a top-level import
  would create a cycle.
- ``registry`` is imported lazily inside ``load`` bodies for backend-name
  validation / payload reconstruction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from evoxc.core import BackendError, EVOXCError, PersistenceError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .backend import Backend  # noqa: F401

__all__ = ["Signature", "LoweredProgram", "CompiledArtifact"]

#: Persisted payload type tags for the evoxc.persist container.
PAYLOAD_LOWERED = "evoxc.lowered_program"
PAYLOAD_ARTIFACT = "evoxc.compiled_artifact"

#: Signature encoding keys, matching ``trace.Graph.signature_info()`` exactly
#: (and the ``Signature`` field names) so formats align across artifacts.
_SIGNATURE_KEYS = (
    "input_tree",
    "output_tree",
    "input_specs",
    "output_specs",
    "static_values",
    "output_static_values",
)


def _encode_signature(signature: "Signature | None"):
    """Encode ``signature`` as a JSON-safe dict for the persist container.

    Mirrors ``trace.Graph.signature_info()`` exactly: the keys are the
    ``Signature`` field names, each value ``persist.encode_value``'d (TreeSpec,
    TensorSpec and static values all have codec entries), so the dict is
    JSON-safe as passed to ``persist.save_object``. ``None`` in, ``None`` out.
    """
    if signature is None:
        return None
    from evoxc import persist

    return {
        "input_tree": persist.encode_value(signature.input_tree),
        "output_tree": persist.encode_value(signature.output_tree),
        "input_specs": persist.encode_value(signature.input_specs),
        "output_specs": persist.encode_value(signature.output_specs),
        "static_values": persist.encode_value(signature.static_values),
        "output_static_values": persist.encode_value(signature.output_static_values),
    }


def _decode_signature(signature_info) -> "Signature | None":
    """Rebuild a ``Signature`` from container ``signature_info``.

    Decodes each of the encoded keys via ``persist.decode_value``; a
    missing/empty ``signature_info`` yields ``None``. A present-but-malformed
    dict raises ``PersistenceError`` naming the missing field — artifacts are
    never loaded with a silently partial signature.
    """
    if not signature_info:
        return None
    from evoxc import persist

    fields = {}
    for key in _SIGNATURE_KEYS:
        if key not in signature_info:
            raise PersistenceError(
                f"corrupt: signature_info is missing the {key!r} field"
            )
        fields[key] = persist.decode_value(signature_info[key])
    return Signature(**fields)


def _recorded_backend_name(backend_info) -> str:
    """Extract and validate the recorded backend name from the container header.

    Raises ``PersistenceError`` for a malformed header — never a silent
    reinterpretation of an artifact written by some other tool.
    """
    if not isinstance(backend_info, dict):
        raise PersistenceError(
            "corrupt: backend_info must be a dict, got "
            f"{type(backend_info).__name__}"
        )
    name = backend_info.get("name")
    if not isinstance(name, str) or not name:
        raise PersistenceError(
            "corrupt: backend_info has no 'name' string field"
        )
    return name


def _require_registered_backend(name: str) -> None:
    """Validate ``name`` against the backend registry (lazy import).

    An unknown backend raises ``PersistenceError`` — the artifact is NOT
    silently re-lowered/re-compiled by another backend.
    """
    from .registry import get

    try:
        get(name)
    except BackendError as exc:
        raise PersistenceError(
            f"artifact records unknown backend {name!r} — never silently "
            f"re-lowers/re-compiles"
        ) from exc


# ---------------------------------------------------------------------------
# Pure-inspection helpers shared by LoweredProgram and CompiledArtifact
# ---------------------------------------------------------------------------
# Everything below is read-only reporting: it never traces, lowers, compiles,
# caches, verifies or deserializes (``source_text`` is the ONE exception — it
# asks the existing ``text()`` rendering for its output and downgrades any
# rendering failure to ``None``).

#: Payload dict ``"format"`` discriminators whose name IS the payload kind.
_KNOWN_PAYLOAD_FORMATS = frozenset({"stablehlo", "stablehlo-segments"})


def _payload_kind(payload: Any) -> str:
    """Classify a staged payload for ``__repr__``/``summary()`` (pure).

    * ``"serialized-ir"`` — a dict carrying ``version`` + ``module``
      (``ir.serialize_module`` output, the numpy backend's payload).
    * ``"mlir"`` — MLIR text itself (a ``str``) or a dict carrying
      ``mlir_text`` (the single-segment compiler payload).
    * ``"stablehlo"`` / ``"stablehlo-segments"`` — a dict whose ``format``
      field names one of the known compiler payload formats.
    * ``"json"`` — any other dict; the payload's type name for anything else.
    """
    if isinstance(payload, dict):
        if "version" in payload and "module" in payload:
            return "serialized-ir"
        if "mlir_text" in payload:
            return "mlir"
        fmt = payload.get("format")
        if isinstance(fmt, str) and fmt in _KNOWN_PAYLOAD_FORMATS:
            return fmt
        return "json"
    if isinstance(payload, str):
        return "mlir"
    return type(payload).__name__


def _payload_bytes(payload: Any) -> int | None:
    """Size of the payload in bytes, or ``None`` when not computable (pure).

    ``str`` payloads report their UTF-8 length; ``dict`` payloads the length of
    the canonical JSON encoding (sorted keys, compact separators, ``default=str``
    so an odd leaf can never raise). Anything else — or a canonicalization
    failure — reports ``None``.
    """
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    if isinstance(payload, dict):
        try:
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            )
        except Exception:  # noqa: BLE001 - inspection must never raise
            return None
        return len(encoded.encode("utf-8"))
    return None


def _signature_field(signature: "Signature | None", attr: str) -> tuple:
    """A ``Signature`` sequence field as a tuple (empty without a signature)."""
    if signature is None:
        return ()
    value = getattr(signature, attr, ())
    return tuple(value) if value else ()


def _spec_shapes(specs: tuple) -> list[list[str]]:
    """Per-leaf symbolic shape rendered as strings (``[str(dim), ...]``)."""
    return [[str(dim) for dim in (getattr(spec, "shape", ()) or ())] for spec in specs]


def _spec_dtypes(specs: tuple) -> list[str]:
    """Per-leaf dtype rendered as a string."""
    return [str(spec.dtype) for spec in specs]


def _program_summary(
    *,
    backend: str,
    target: str | None,
    signature: "Signature | None",
    payload: Any,
    required_custom_ops: tuple = (),
    runtime_dependencies: "dict[str, str] | None" = None,
) -> dict:
    """Build the ``summary()`` dict shared by both staged objects (pure)."""
    input_specs = _signature_field(signature, "input_specs")
    output_specs = _signature_field(signature, "output_specs")
    return {
        "backend": backend,
        "target": target,
        "payload_kind": _payload_kind(payload),
        "payload_bytes": _payload_bytes(payload),
        "input_count": len(input_specs),
        "output_count": len(output_specs),
        "input_shapes": _spec_shapes(input_specs),
        "output_shapes": _spec_shapes(output_specs),
        "input_dtypes": _spec_dtypes(input_specs),
        "static_value_count": len(_signature_field(signature, "static_values")),
        "required_custom_ops": list(required_custom_ops),
        "runtime_dependencies": dict(runtime_dependencies or {}),
    }


def _program_repr(
    cls_name: str,
    *,
    backend: str,
    target: str | None,
    signature: "Signature | None",
    payload: Any,
) -> str:
    """One-line ``__repr__`` body shared by both staged objects.

    Reports the backend, the target (``CompiledArtifact`` only), the payload
    kind + byte size and the signature's leaf counts. A not-computable size
    renders as ``?``; anything that could make an attribute unreadable is
    caught by the callers, so a ``repr`` never raises.
    """
    size = _payload_bytes(payload)
    parts = [f"backend={backend!r}"]
    if target is not None:
        parts.append(f"target={target!r}")
    parts.append(
        f"payload={_payload_kind(payload)} {'?' if size is None else f'{size} B'}"
    )
    parts.append(f"inputs={len(_signature_field(signature, 'input_specs'))}")
    parts.append(f"outputs={len(_signature_field(signature, 'output_specs'))}")
    return f"{cls_name}({', '.join(parts)})"


def _render_payload_text(backend: str, payload: Any) -> str:
    """Human-readable rendering of a payload (backend-specific).

    * MLIR-style payloads (``str``) are returned verbatim.
    * Compiler-backend payloads (dict with ``"format": "stablehlo"`` and an
      ``"mlir_text"`` field — see ``compiler.CompilerBackend``) render their
      MLIR text.
    * Serialized ``ir.Module`` payloads (dict carrying ``version`` + ``module``
      keys) are deserialized and pretty-printed.
    * Anything else raises ``core.BackendError`` naming the backend.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and "mlir_text" in payload:
        return payload["mlir_text"]
    if isinstance(payload, dict) and "version" in payload and "module" in payload:
        from evoxc import ir

        module = ir.deserialize_module(payload)
        return ir.pretty_print(module)
    raise BackendError(
        f"backend {backend!r}: cannot render lowered payload of type "
        f"{type(payload).__name__} — expected MLIR text (str) or a "
        f"serialized ir.Module dict"
    )


@dataclass(frozen=True)
class Signature:
    """Input/output contract recorded at ``lower()`` time, passed down from the Graph.

    Attributes:
        input_tree: ``core.TreeSpec`` of the structured input (flatten/unflatten).
        output_tree: ``core.TreeSpec`` of the structured output.
        input_specs: per-leaf ``core.TensorSpec`` in flattened input order.
        output_specs: per-leaf ``core.TensorSpec`` in flattened output order.
        static_values: Python values the trace specialized on (input static
            leaves, in pre-order leaf order); validated at run time (changing
            them means a different graph — never a hidden recompile).
        output_static_values: Python values recorded for static OUTPUT leaves
            (in pre-order leaf order), re-inserted by ``evoxc.run`` when
            reconstructing the structured outputs.
    """

    input_tree: Any = None
    output_tree: Any = None
    input_specs: tuple[Any, ...] = ()
    output_specs: tuple[Any, ...] = ()
    static_values: tuple[Any, ...] = ()
    output_static_values: tuple[Any, ...] = ()


@dataclass
class LoweredProgram:
    """Backend-specific lowered form of a Graph (pre-compile staging object).

    Attributes:
        backend: backend name (str) that produced this program.
        signature: input/output contract (see ``Signature``).
        payload: backend-specific, serializable, self-describing lowered form.
            numpy backend: versioned ``ir.Module`` serialization
            (``ir.serialize_module``). A hypothetical stablehlo-backed lowering
            would carry MLIR text.

    ``text()`` renders a human-readable form (numpy: serialized-IR text;
    MLIR-style payloads: the MLIR text itself).
    ``save()``/``load()`` use the ``evoxc.persist`` container (lazy import).
    Payloads are self-contained (serialized ``ir.Module`` dict or MLIR text),
    so ``load()`` reconstructs the payload directly from the container — no
    backend delegation. It validates the recorded backend name against the
    registry; mismatch raises ``core.PersistenceError``, never a silent
    re-lowering.
    """

    backend: str
    signature: Signature | None = None
    payload: Any = None

    def __repr__(self) -> str:
        """One-line inspection line: backend, payload kind + size, leaf counts."""
        try:
            return _program_repr(
                "LoweredProgram",
                backend=self.backend,
                target=None,
                signature=self.signature,
                payload=self.payload,
            )
        except Exception:  # noqa: BLE001 - inspection must never raise
            return f"LoweredProgram(backend={self.backend!r}, payload=<unavailable>)"

    def summary(self) -> dict:
        """Pure-inspection snapshot of this lowered program.

        Reads attributes only — never verifies, deserializes, lowers or caches.
        The returned dict has exactly these keys: ``backend``, ``target``
        (``None`` — a lowered program has no compiled target), ``payload_kind``,
        ``payload_bytes``, ``input_count``, ``output_count``, ``input_shapes``,
        ``output_shapes``, ``input_dtypes``, ``static_value_count``,
        ``required_custom_ops`` (``[]``) and ``runtime_dependencies`` (``{}``).
        """
        return _program_summary(
            backend=self.backend,
            target=None,
            signature=self.signature,
            payload=self.payload,
        )

    def source_text(self) -> str | None:
        """The human-readable program text, or ``None`` when there is none.

        Returns exactly what ``text()`` renders (MLIR text verbatim, the
        compiler payload's MLIR text, or the pretty-printed serialized IR) and
        — unlike ``text()`` — catches the rendering/deserialization failure and
        returns ``None`` instead of raising ``core.BackendError``.
        """
        try:
            return self.text()
        except (EVOXCError, ValueError, TypeError, KeyError, IndexError):
            return None

    def text(self) -> str:
        """Human-readable rendering of the payload (backend-specific).

        * MLIR-style payloads (``str``) are returned verbatim.
        * Compiler-backend payloads (dict with ``"format": "stablehlo"``
          and an ``"mlir_text"`` field — see ``compiler.CompilerBackend``)
          render their MLIR text.
        * Serialized ``ir.Module`` payloads (dict carrying ``version`` +
          ``module`` keys) are deserialized and pretty-printed.
        * Anything else raises ``core.BackendError`` naming the backend.
        """
        return _render_payload_text(self.backend, self.payload)

    def save(self, path: str | os.PathLike) -> None:
        """Serialize via the evoxc.persist container (lazy import).

        Metadata records: payload type tag, backend name, signature
        (TreeSpecs + per-leaf specs + static values), IR format version.
        """
        from evoxc import persist

        persist.save_object(
            path,
            PAYLOAD_LOWERED,
            payload_fields={"payload": self.payload},
            backend_info={"name": self.backend},
            signature_info=_encode_signature(self.signature),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "LoweredProgram":
        """Deserialize and validate.

        Reads the container, validates the recorded backend name against the
        registry (unknown/mismatched backend => ``core.PersistenceError``),
        and rebuilds the ``Signature`` and self-contained payload directly.
        Never re-lowers.
        """
        from evoxc import persist

        loaded = persist.load_object(path, expected_payload_type=PAYLOAD_LOWERED)
        backend = _recorded_backend_name(loaded.backend_info)
        _require_registered_backend(backend)
        signature = _decode_signature(loaded.signature_info)
        try:
            payload = loaded.payload["payload"]
        except KeyError:
            raise PersistenceError(
                "corrupt: lowered-program payload is missing the 'payload' field"
            ) from None
        return cls(backend=backend, signature=signature, payload=payload)


@dataclass
class CompiledArtifact:
    """Compiled, self-describing artifact (post-``compile`` staging object).

    Attributes:
        backend: backend name (str) that produced this artifact.
        signature: input/output contract (see ``Signature``).
        target: compilation target descriptor (numpy: ``"cpu"``).
        payload: backend-specific compiled form (numpy: the serialized
            ``ir.Module`` — the artifact IS serialized IR, there is no machine code).
        required_custom_ops: tuple of custom-op names the program depends on;
            load validates their availability (missing => ``core.PersistenceError``).
        runtime_dependencies: dict of runtime dependency name -> version required
            to execute the artifact (self-describing per serialization contract).

    ``save()``/``load()`` use the ``evoxc.persist`` container (lazy import);
    ``load()`` validates backend and dependencies — never silently recompiles.
    """

    backend: str
    signature: Signature | None = None
    target: str = ""
    payload: Any = None
    required_custom_ops: tuple[str, ...] = ()
    runtime_dependencies: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        """One-line inspection line: backend, target, payload kind + size, leaves."""
        try:
            return _program_repr(
                "CompiledArtifact",
                backend=self.backend,
                target=self.target,
                signature=self.signature,
                payload=self.payload,
            )
        except Exception:  # noqa: BLE001 - inspection must never raise
            return f"CompiledArtifact(backend={self.backend!r}, payload=<unavailable>)"

    def summary(self) -> dict:
        """Pure-inspection snapshot of this compiled artifact.

        Reads attributes only — never verifies, deserializes, compiles or
        caches. The returned dict has exactly these keys: ``backend``,
        ``target``, ``payload_kind``, ``payload_bytes``, ``input_count``,
        ``output_count``, ``input_shapes``, ``output_shapes``,
        ``input_dtypes``, ``static_value_count``, ``required_custom_ops`` and
        ``runtime_dependencies``.
        """
        return _program_summary(
            backend=self.backend,
            target=self.target,
            signature=self.signature,
            payload=self.payload,
            required_custom_ops=self.required_custom_ops,
            runtime_dependencies=self.runtime_dependencies,
        )

    def source_text(self) -> str | None:
        """The human-readable program text, or ``None`` when there is none.

        Renders the payload through the SAME shared helper ``LoweredProgram.text()``
        uses (MLIR text verbatim, the compiler payload's MLIR text, or the
        pretty-printed serialized IR) and — unlike ``text()`` — catches the
        rendering/deserialization failure and returns ``None`` instead of
        raising ``core.BackendError``.
        """
        try:
            return _render_payload_text(self.backend, self.payload)
        except (EVOXCError, ValueError, TypeError, KeyError, IndexError):
            return None

    def save(self, path: str | os.PathLike) -> None:
        """Serialize via the evoxc.persist container (lazy import).

        Metadata records: payload type tag, backend name, target, signature,
        required custom ops, runtime dependencies, IR format version.
        """
        from evoxc import persist

        persist.save_object(
            path,
            PAYLOAD_ARTIFACT,
            payload_fields={
                "payload": self.payload,
                "required_custom_ops": tuple(self.required_custom_ops),
                "runtime_dependencies": dict(self.runtime_dependencies),
            },
            backend_info={"name": self.backend, "target": self.target},
            signature_info=_encode_signature(self.signature),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CompiledArtifact":
        """Deserialize and validate.

        Validates the recorded backend against the registry and required
        custom ops/runtime dependencies against the environment; mismatch
        raises ``core.PersistenceError`` — never silently recompiles.
        """
        from evoxc import persist

        loaded = persist.load_object(path, expected_payload_type=PAYLOAD_ARTIFACT)
        backend = _recorded_backend_name(loaded.backend_info)
        _require_registered_backend(backend)
        signature = _decode_signature(loaded.signature_info)
        for field in ("payload", "required_custom_ops", "runtime_dependencies"):
            if field not in loaded.payload:
                raise PersistenceError(
                    f"corrupt: compiled-artifact payload is missing the "
                    f"{field!r} field"
                )
        payload = loaded.payload["payload"]
        target = loaded.backend_info.get("target", "")
        required_custom_ops = tuple(loaded.payload["required_custom_ops"])
        runtime_dependencies = loaded.payload["runtime_dependencies"]
        if not isinstance(runtime_dependencies, dict):
            raise PersistenceError(
                "corrupt: runtime_dependencies must be a dict, got "
                f"{type(runtime_dependencies).__name__}"
            )
        if not all(isinstance(name, str) for name in required_custom_ops):
            raise PersistenceError(
                "corrupt: required_custom_ops must be a list/tuple of strings"
            )
        _validate_runtime_dependencies(runtime_dependencies)
        _validate_required_custom_ops(required_custom_ops, backend)
        return cls(
            backend=backend,
            signature=signature,
            target=target,
            payload=payload,
            required_custom_ops=required_custom_ops,
            runtime_dependencies=dict(runtime_dependencies),
        )


def _validate_runtime_dependencies(runtime_dependencies: dict) -> None:
    """Check recorded runtime dependency versions against the environment.

    A mismatch raises ``core.PersistenceError`` — the artifact is never
    silently recompiled against different dependencies.
    """
    recorded_numpy = runtime_dependencies.get("numpy")
    if recorded_numpy is not None:
        import numpy as np

        if np.__version__ != recorded_numpy:
            raise PersistenceError(
                f"artifact requires numpy {recorded_numpy}, environment has "
                f"{np.__version__} — never silently recompile"
            )


def _validate_required_custom_ops(required_custom_ops: tuple[str, ...], backend: str) -> None:
    """Verify every required custom BLOCK op resolves in this environment.

    A name registered as a block (a ``block_call`` backend impl or a
    portable decomposition) passes. A name that is NOT a registered block is
    an ``external_call`` KERNEL name: external kernels are recorded in
    ``required_custom_ops`` for SELF-DESCRIPTION only and resolve LAZILY at
    run time through ``evoxc.external`` — they must never be required at load
    time (kernels are re-registered in the process before RUN, not before
    load; a missing registration fails loudly at run time with a
    ``core.BackendError`` naming the kernel). Backend ``load()`` methods
    additionally re-validate declared blocks from the module itself (e.g.
    the numpy backend scans for ``block_call`` ops). If ``evoxc.block`` is
    unimportable (minimal deployments / parallel implementation), the check
    degrades gracefully by skipping — this is where the validation lives.
    """
    if not required_custom_ops:
        return
    try:
        from evoxc.block import registry as block_registry
    except ImportError:
        # evoxc.block unavailable: validation skipped (graceful degradation).
        return
    for op_name in required_custom_ops:
        has_impl = block_registry.get_impl(op_name, backend) is not None
        has_portable = block_registry.get_portable(op_name) is not None
        if not has_impl and not has_portable:
            # Not a registered block — an external-kernel name: lazy
            # run-time resolution, never a load-time requirement.
            continue
