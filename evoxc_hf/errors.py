"""Exception hierarchy for the ``evoxc_hf`` package.

Every error raised by ``evoxc_hf`` derives from :class:`HFError`, which itself
derives from :class:`evoxc.core.EVOXCError` so callers can catch the whole evoxc
error surface with a single ``except EVOXCError``.

Import-time constraint: this module imports **only** ``evoxc.core``. It must never
pull in numpy or the optional dependencies (torch / transformers / iree /
safetensors), so that ``import evoxc_hf.errors`` stays dependency-free.
"""

from evoxc.core import EVOXCError

__all__ = ["HFError", "HFConfigError", "HFWeightError", "HFUnsupportedError"]


class HFError(EVOXCError):
    """Base class for every error raised by ``evoxc_hf``.

    Catch this to handle any HuggingFace-integration failure without also
    catching unrelated errors from the underlying ``evoxc`` package.
    """


class HFConfigError(HFError):
    """Malformed or unsupported ``config.json``.

    Raised when a required field is missing/invalid or a config value names a
    feature outside the supported matrix. The message names the offending field
    and the file it came from.
    """


class HFWeightError(HFError):
    """Checkpoint reading, key-mapping, or tensor-shape problem.

    Raised when a checkpoint file cannot be parsed, a required weight key is
    missing, or a loaded tensor has a shape incompatible with the config. The
    message names the file/field/tensor involved.
    """


class HFUnsupportedError(HFError):
    """A well-formed request outside the v1 support matrix.

    Raised when the config and checkpoint are valid but the requested model
    family, activation, or backend path is not implemented in this version.
    """
