"""evoxc.formats — explicit, dependency-free tensor-only binary file formats.

Scope is the **tensor half** of the modern checkpoint containers:

* :mod:`evoxc.formats.safetensors` — safetensors read (``mmap`` zero-copy) and
  canonical write.
* :mod:`evoxc.formats.gguf` — GGUF v2/v3 read with metadata, tensor infos and
  block dequantization.

Everything here imports ``evoxc.core`` plus the standard library and numpy
only, and never traces, compiles, moves data or imports another evoxc
submodule.  A loaded tensor is ordinary host data:
:class:`~evoxc.core.Tensor` wrapping numpy memory on ``Device('cpu', 0)``.
Memory-mapped tensors are read-only views (copy explicitly if you need a
writable array).  Tokenizers, chat templates, generation configs and model
semantics are out of scope — those belong to a model-addon distribution.

Public surface (also documented in this directory's ``CONTEXT.md``):
:func:`load_safetensors`, :func:`save_safetensors`, :class:`SafetensorsFile`,
:func:`load_gguf`, :class:`GGUFFile`, :class:`TensorInfo`.
"""
from .gguf import GGUFFile, load_gguf
from .safetensors import SafetensorsFile, load_safetensors, save_safetensors
from .tensorinfo import TensorInfo

__all__ = [
    "GGUFFile",
    "SafetensorsFile",
    "TensorInfo",
    "load_gguf",
    "load_safetensors",
    "save_safetensors",
]
