"""Shared low-level byte/mmap helpers for the tensor file formats.

Both readers in this package (``safetensors`` and ``gguf``) work over a
:class:`MappedFile`: a read-only view of a whole file that either memory-maps it
(``mmap=True``, the default) or reads it eagerly into ``bytes``
(``mmap=False``).  Every accessor is **bounds-checked before a view is
created**, so a corrupt length/offset can never reach ``np.frombuffer`` or a
slice.
Byte-level constructs shared by the formats live here:

* little-endian scalar reads (``u8``/``u16``/``u32``/``u64``/``f32``),
* ``align_up`` for the container alignment rules,
* :meth:`MappedFile.read` for raw byte runs,
* :meth:`MappedFile.array` which returns a 1-D numpy view built through a
  ``uint8`` buffer (so an unaligned tensor start is fine) reshaped to the
  requested shape.

Read-only mmap semantics: a view returned by :meth:`MappedFile.array` aliases
the mapped file and is therefore read-only (``arr.flags.writeable`` is
``False``).  ``mmap=False`` callers pass ``copy=True`` to get an independent,
writable array instead.  :meth:`MappedFile.close` is idempotent and never
raises: if numpy views are still alive the OS mapping is released when the
last view is garbage-collected (numpy/mmap do not permit an eager close while
buffers are exported), while the file descriptor is closed immediately.
"""
from __future__ import annotations

import mmap as _mmap
import os
import struct
from typing import Tuple

import numpy as np

from ..core import PersistenceError

__all__ = ["align_up", "MappedFile"]


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    if not isinstance(alignment, int) or isinstance(alignment, bool):
        raise TypeError("alignment must be an int")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


class MappedFile:
    """A read-only file image with bounds-checked little-endian readers.

    Args:
        path: Path to an existing file.
        mmap: ``True`` (default) memory-maps the file through
            :func:`mmap.mmap` with ``ACCESS_READ``; ``False`` reads the whole
            file into ``bytes`` up front.

    Raises:
        PersistenceError: The file cannot be opened/read, or mapping fails.
    """

    __slots__ = ("path", "size", "buffer", "closed", "_file", "_mapped")

    def __init__(self, path, mmap: bool = True):
        self.path = os.fspath(path)
        self.closed = False
        self._mapped = False
        self.buffer = b""
        try:
            self._file = open(self.path, "rb")
        except OSError as exc:  # pragma: no cover - environment dependent
            raise PersistenceError(
                f"{self.path}: cannot open file: {exc}"
            ) from exc
        try:
            self.size = os.fstat(self._file.fileno()).st_size
            if mmap and self.size:
                self.buffer = _mmap.mmap(
                    self._file.fileno(), 0, access=_mmap.ACCESS_READ
                )
                self._mapped = True
            else:
                # An empty file cannot be mmap'ed; bytes() is also the eager path.
                self.buffer = self._file.read()
        except (OSError, ValueError) as exc:  # pragma: no cover - environment dependent
            self._file.close()
            raise PersistenceError(f"{self.path}: cannot map file: {exc}") from exc

    # -- introspection ------------------------------------------------------
    @property
    def is_mapped(self) -> bool:
        """Whether the file image is a live memory map (read-only) or bytes."""
        return self._mapped

    def __len__(self) -> int:
        return self.size

    # -- validation ---------------------------------------------------------
    def require_open(self, what: str) -> None:
        """Raise :class:`PersistenceError` if the file image was closed."""
        if self.closed:
            raise PersistenceError(f"{self.path}: cannot read {what}: file is closed")

    def check(self, begin: int, end: int, what: str) -> None:
        """Validate ``begin <= end`` and the byte range ``[begin, end)``.

        Raises:
            PersistenceError: The range is negative/reversed or exceeds the
                file size.  The message names the offending byte range, the
                file size and the construct being read.
        """
        self.require_open(what)
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or begin < 0
            or end < begin
            or end > self.size
        ):
            raise PersistenceError(
                f"{self.path}: {what} claims bytes [{begin}, {end}) but the "
                f"file is {self.size} bytes"
            )

    # -- raw reads ----------------------------------------------------------
    def read(self, offset: int, count: int, what: str) -> bytes:
        """Return ``count`` bytes starting at ``offset`` (bounds-checked)."""
        self.check(offset, offset + count, what)
        return bytes(self.buffer[offset : offset + count])

    def unpack(self, fmt: str, offset: int, what: str) -> Tuple:
        """``struct.unpack_from`` one little-endian value, bounds-checked."""
        size = struct.calcsize(fmt)
        self.check(offset, offset + size, what)
        return struct.unpack_from(fmt, self.buffer, offset)

    def u8(self, offset: int, what: str) -> int:
        return self.unpack("<B", offset, what)[0]

    def u16(self, offset: int, what: str) -> int:
        return self.unpack("<H", offset, what)[0]

    def u32(self, offset: int, what: str) -> int:
        return self.unpack("<I", offset, what)[0]

    def u64(self, offset: int, what: str) -> int:
        return self.unpack("<Q", offset, what)[0]

    def f32(self, offset: int, what: str) -> float:
        return self.unpack("<f", offset, what)[0]

    def array(
        self,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        offset: int,
        nbytes: int,
        what: str,
        copy: bool = False,
    ) -> np.ndarray:
        """Return ``shape``-shaped ``dtype`` view over ``[offset, offset+nbytes)``.

        The view is built through a ``uint8`` buffer so the start offset does
        not need to be aligned to ``dtype.itemsize``.  ``copy=True`` returns an
        independent writable array (the eager, ``mmap=False`` semantics).
        """
        numel = 1
        for dim in shape:
            numel *= int(dim)
        expected = numel * dtype.itemsize
        if expected != nbytes:
            raise PersistenceError(
                f"{self.path}: {what} needs {expected} bytes for "
                f"{shape} {np.dtype(dtype).name}, got {nbytes}"
            )
        self.check(offset, offset + nbytes, what)
        if numel == 0:
            return np.empty(shape, dtype=dtype)
        raw = np.frombuffer(self.buffer, dtype=np.uint8, count=nbytes, offset=offset)
        view = raw.view(dtype).reshape(shape)
        return np.array(view, copy=True) if copy else view

    def byte_view(self, offset: int, nbytes: int, what: str, copy: bool = False) -> np.ndarray:
        """Return a 1-D ``uint8`` view over ``[offset, offset+nbytes)``."""
        self.check(offset, offset + nbytes, what)
        if nbytes == 0:
            return np.empty((0,), dtype=np.uint8)
        raw = np.frombuffer(self.buffer, dtype=np.uint8, count=nbytes, offset=offset)
        return np.array(raw, copy=True) if copy else raw

    # -- lifetime -----------------------------------------------------------
    def close(self) -> None:
        """Release the mapping/fd.  Idempotent and never raises.

        Live numpy views keep the underlying mapping alive until they are
        collected (the OS releases it then); the file descriptor is closed
        immediately in either case.
        """
        if self.closed:
            return
        self.closed = True
        self.size = 0
        buffer = self.buffer
        self.buffer = b""
        try:
            self._file.close()
        except OSError:  # pragma: no cover - environment dependent
            pass
        if isinstance(buffer, _mmap.mmap):
            try:
                buffer.close()
            except BufferError:
                # Views are still exported: numpy keeps the mapping alive and
                # mmap.__del__ closes it once the last view is released.
                pass

    def __enter__(self) -> "MappedFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - interpreter teardown path
        try:
            self.close()
        except Exception:
            pass


def file_size(path) -> int:
    """Size of ``path`` in bytes (kept for symmetry with readers)."""
    return os.stat(os.fspath(path)).st_size
