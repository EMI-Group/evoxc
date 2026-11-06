"""evoxc.backends.adapters — optional compiler backends (pluggable adapters).

This package is the home of the OPTIONAL compiler adapters. Each adapter is
a separate module subclassing the shared ``evoxc.backends.compiler``
``CompilerBackend`` / ``CompilerExecutable`` base classes:

- ``iree.py`` — IREE adapter (IMPLEMENTED): compiles the StableHLO payload
  with ``iree.compiler.compile_str`` and executes the VM flatbuffer through
  ``iree.runtime``. Serves the ``local-task`` (llvm-cpu), ``cuda``, ``hip``
  (rocm) and ``vulkan`` HAL drivers — ``Device(kind, N)`` selects the driver,
  and every GPU-class kind runs device-resident (explicit placement via
  ``Tensor.to``, no implicit host↔device staging). Singleton
  ``iree_backend``.
- ``xla.py`` — XLA-via-PJRT adapter (IMPLEMENTED): drives a PJRT C API
  plugin DISCOVERED at runtime (a ``.so`` exporting ``GetPjRtApi``) through
  pure-stdlib ctypes — the vendored ABI translation lives in
  ``_pjrt_c_api.py`` (provenance + version gate documented there) and the
  driver in ``xla_util.py``. The plugin is either an installed JAX PJRT
  plugin following the ``jax_plugins/<platform>/<name>_plugin.so``
  convention (e.g. ``pip install jax-rocm7-pjrt`` / ``jax-cuda12-pjrt``), a
  ``options["plugin_path"]`` or ``EVOXC_PJRT_PLUGIN`` path, or a source
  build from OpenXLA (``bazel build
  //xla/pjrt/c:pjrt_c_api_cpu_plugin``); it is dlopen-ed, never imported as
  a Python package (no jax/jaxlib). Singleton ``xla_backend``.
- ``tvm.py`` — TVM adapter (IMPLEMENTED): ``tvm.relax.frontend.stablehlo``
  importer + Relax VM execution (llvm target); vendored-translator
  compatibility shim in ``tvm_util.py`` (jaxlib is used ONLY for its
  bundled MLIR python bindings). Singleton ``tvm_backend``.

Adapter-module contract (binding for adapter authors):

- Subclass ``evoxc.backends.compiler.CompilerBackend`` (name, capabilities,
  ``check_available``, ``compile``, ``load``) and
  ``evoxc.backends.compiler.CompilerExecutable`` (``backend_name``, ``run``).
- Each module exposes a module-level ``register()`` function that (a)
  establishes the compiler dependency's availability and (b) calls
  ``evoxc.backends.registry.register(instance)`` (idempotent). The check is
  the adapter's own: iree/tvm probe their pip-installable packages and raise
  ``core.BackendError`` with an install hint (e.g. ``pip install
  evoxc[iree]``) when missing, while xla registers UNCONDITIONALLY because
  its dependency is a per-call user-provided plugin ``.so`` (``plugin_path``
  / ``EVOXC_PJRT_PLUGIN`` / a discovered ``jax_plugins`` plugin) that only
  ``compile``/``load`` can resolve — those stages raise the actionable
  plugin error, and ``check_available()`` is the explicit option-less
  preflight.
- Heavy-import rule (binding): imports of the compiler dependencies
  (iree/jaxlib/tvm/…) live ONLY inside function bodies, NEVER at module top
  level — ``import evoxc`` and ``import evoxc.backends`` must never import an
  adapter or its compiler dependency. The ``jax`` package is never imported
  anywhere (jaxlib is used only for its bundled MLIR bindings, tvm adapter).
- Activation is register-on-first-use: ``evoxc.backends.registry.get(name)``
  (via ``registry.OPTIONAL_ADAPTERS``) imports the adapter module and calls
  its ``register()`` on the first registry miss — the same path that
  auto-activates adapters when loading persisted artifacts.

This ``__init__.py`` deliberately exposes NO names and imports nothing
heavy — it is documentation only (plus the package marker).
"""
from __future__ import annotations

__all__: list[str] = []
