"""The public tensor descriptor shared by every format reader.

``TensorInfo`` describes one tensor *as stored in a file*: where the format's
own type identifier, the shape a reader materializes and the stored payload
size all live in one immutable record.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TensorInfo"]


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for one tensor stored in a container file.

    Attributes:
        name: The tensor's name in the file.
        dtype: The canonical dtype name.  This is the numpy dtype name when
            numpy has a native equivalent (``"float32"``, ``"int16"``,
            ``"bool"``, ...); for types numpy cannot express it is the
            format's own canonical name (``"BF16"``, ``"F8_E4M3"`` for
            safetensors; ``"BF16"``, ``"Q4_K"``, ... for GGUF).  It describes
            the *stored* type, not necessarily the dtype
            :meth:`~evoxc.formats.SafetensorsFile.tensor` /
            :meth:`~evoxc.formats.GGUFFile.tensor` materializes.
        shape: The dense shape, in numpy row-major order.  GGUF stores its
            dimensions fastest-axis-first (``ne[0]`` first); the reader
            reverses them here so ``tensor(name).numpy().shape == info.shape``.
        nbytes: The tensor's stored payload size in bytes (the on-disk
            extent; for block-quantized GGUF types this is
            ``numel / block_size * type_size``).
        source_dtype: The source format's own type identifier, verbatim:
            a safetensors tag (``"F32"``, ``"BF16"``, ...) or a GGML type
            name (``"Q4_K"``, ...).
    """

    name: str
    dtype: str
    shape: tuple
    nbytes: int
    source_dtype: str

    @property
    def numel(self) -> int:
        """Number of logical elements (``nbytes`` is a different measure)."""
        count = 1
        for dim in self.shape:
            count *= int(dim)
        return count

    @property
    def rank(self) -> int:
        return len(self.shape)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}: {self.dtype}{tuple(self.shape)} ({self.nbytes} bytes)"
