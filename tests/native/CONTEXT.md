# tests/native — private native-binding pins

## Intent

Pins the private Rust/PyO3 binding seam: the neutral envelope codec (`evoxc/_native_codec.py`) and the optional extension loader / typed façade (`evoxc/_native.py`) documented in `../../native/CONTEXT.md`.
CPU only and self-contained: no test here requires the Rust crate or a built extension.

## API under test

- `evoxc._native_codec`: `ENVELOPE_FORMAT`, `encode_module`, `decode_module`, `decode_native_module`, `encode_type`, `decode_type`.
- `evoxc._native`: `available`, `require`, `version`, `infer`, `transform`, `batch_functions`, `batching_rule_keys`, `NATIVE_LIB_ENV`, `EXTENSION_MODULE_NAME`.

## Test files

| File | Covers | Tests |
|---|---|---|
| `test_codec.py` | envelope format 1: type/`Dim`/`DimExpr` round-trips, the FLAT envelope with inline `__evoxc_ndarray__` payloads (no `constants` table, no `sha256`), SSA ids/uses preserved, attrs (incl. Builder-normalized), locations (`None` stays `None`), constants incl. byte order and zero-dim, regions/control flow and multi-block, encode non-mutation, malformed-envelope errors naming the missing field, duplicate value ids, and the trusted (`decode_native_module`) vs validated decode | 36 |
| `test_native_loader.py` | the loader/façade: absent extension → `available()` False, `require()`'s remedy message and a raising `version()`; `NATIVE_LIB_ENV` override missing/unloadable paths; present extension via an injected fake module (`version`, `infer` type encode/decode and explicit result types, `transform` module round-trip + extra keys); `NativeError.kind` → evoxc error mapping (and unknown kinds re-raising unchanged); `import evoxc` never needs the extension | 20 |

## Constraints

- Never import `evoxc._evoxc_native`; the present-extension path uses a fake `types.ModuleType` injected as `evoxc._native._MODULE` (reset via `evoxc._native._reset()`).
- Envelope format 1 deliberately omits `code_snippet`, module/function metadata and any IR format version; decode supplies `None`/`{}`/the current `ir.IR_FORMAT_VERSION`.
- Byte order is not preserved for constants: a non-native-endian array decodes value-equal in native order with the same dtype name and shape.

## Status

The transforms delegate to this seam (the engine is the single transform implementation), `batch_functions` consults `batching_rule_keys`, and trace-time inference still runs in Python (`ir/inference.py`); these tests pin the seam itself.
