# tests/native — private native-binding pins

## Intent

Pins the private Rust/PyO3 binding seam: the neutral envelope codec (`evoxc/_native_codec.py`) and the optional extension loader / typed façade (`evoxc/_native.py`) documented in `../../native/CONTEXT.md`.
CPU only and self-contained: no test here requires the Rust crate or a built extension.

## API under test

- `evoxc._native_codec`: `ENVELOPE_FORMAT`, `encode_module`, `decode_module`, `encode_type`, `decode_type`.
- `evoxc._native`: `available`, `require`, `version`, `infer`, `transform`, `batch_functions`, `batching_rule_keys`, `NATIVE_LIB_ENV`, `EXTENSION_MODULE_NAME`.

## Constraints

- Never import `evoxc._evoxc_native`; the present-extension path uses a fake `types.ModuleType` injected as `evoxc._native._MODULE` (reset via `evoxc._native._reset()`).
- Envelope format 1 deliberately omits `code_snippet`, module/function metadata and any IR format version; decode supplies `None`/`{}`/the current `ir.IR_FORMAT_VERSION`.
- Byte order is not preserved for constants: a non-native-endian array decodes value-equal in native order with the same dtype name and shape.

## Status

The transforms delegate to this seam (the engine is the single transform implementation), `batch_functions` consults `batching_rule_keys`, and trace-time inference still runs in Python (`ir/inference.py`); these tests pin the seam itself.
