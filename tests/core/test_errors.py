"""Tests for the evoxc error hierarchy (``evoxc.core.errors``).

Contract (see ``evoxc/core/CONTEXT.md`` and the root error strategy):

- ``EVOXCError`` is the single base class for all public evoxc errors, deriving
  directly from ``Exception``.
- The eight subclasses ``TraceError, ShapeError, TransformError, BackendError,
  PersistenceError, DeviceError, DTypeError, VerificationError`` all derive
  from ``EVOXCError`` (and hence from ``Exception``).
- All nine names are re-exported from ``evoxc.core`` and the package root ``evoxc``
  (identical objects, not copies).
- Errors are ordinary exceptions: instantiable with a message, message
  preserved via ``str(exc)``/``exc.args[0]``, catchable as ``EVOXCError``.
"""

import pytest

import evoxc
import evoxc.core
from evoxc.core import EVOXCError, errors

SUBCLASS_NAMES = [
    "TraceError",
    "ShapeError",
    "TransformError",
    "BackendError",
    "PersistenceError",
    "DeviceError",
    "DTypeError",
    "VerificationError",
]

ALL_NAMES = ["EVOXCError", *SUBCLASS_NAMES]


def test_evoxc_error_base_is_exactly_exception():
    # errors.py declares `class EVOXCError(Exception)` — assert exactly that.
    assert EVOXCError.__bases__ == (Exception,)


@pytest.mark.parametrize("name", SUBCLASS_NAMES)
def test_subclass_derives_from_evoxc_error_and_exception(name):
    cls = getattr(errors, name)
    assert issubclass(cls, EVOXCError)
    assert issubclass(cls, Exception)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_error_is_instantiable_and_message_preserved(name):
    cls = getattr(errors, name)
    msg = f"{name}: something went wrong"
    exc = cls(msg)
    assert str(exc) == msg
    assert exc.args[0] == msg


@pytest.mark.parametrize("name", ALL_NAMES)
def test_error_is_catchable_as_exception(name):
    cls = getattr(errors, name)
    msg = f"{name}: boom"
    with pytest.raises(Exception, match=rf"{name}: boom"):
        raise cls(msg)


@pytest.mark.parametrize("name", SUBCLASS_NAMES)
def test_subclass_is_catchable_as_evoxc_error(name):
    cls = getattr(errors, name)
    msg = f"{name}: caught by base"
    with pytest.raises(EVOXCError, match=rf"{name}: caught by base"):
        raise cls(msg)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_re_exported_from_evoxc_core(name):
    # Same object as in errors, not a shadow copy.
    assert getattr(evoxc.core, name) is getattr(errors, name)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_re_exported_from_evoxc_root(name):
    # Same object as in errors, not a shadow copy.
    assert getattr(evoxc, name) is getattr(errors, name)
