# tests/formats — `evoxc.formats` (safetensors + GGUF) test suite

## Intent

pytest suite for the tensor-only checkpoint readers/writers in `../../evoxc/formats/` (sibling — the API contract is `../../evoxc/formats/CONTEXT.md`).
Every fixture is built synthetically in `tmp_path` by the tests themselves (byte-for-byte writers / an independent reference encoder); no binary fixture is committed and no checkpoint is downloaded.

## Structure

- `safetensors` coverage lives in `test_safetensors.py`; GGUF in `test_gguf.py`; byte-level helpers shared by both live in `_builders.py` (package-qualified import: `from tests.formats._builders import ...`).

## Constraints

- CPU only, no network, no third-party dependency beyond `numpy`/`pytest`.
- A real checkpoint file is only used when one is already present on the host, guarded by an explicit `pytest.skip` reason; never downloaded.
- Corruption tests assert `core.PersistenceError` and that the message names the file and tensor/offset.

## Routing table

| Path | Area |
|---|---|
| `./test_safetensors.py` | safetensors round-trips (every writable dtype, metadata, multi-tensor, mmap both ways, read-only views, atomic write) + the documented corruption rejections |
| `./test_gguf.py` | GGUF v2/v3 byte-built containers: every supported ggml type vs an independent reference dequantizer, metadata KV value types, alignment variants, `raw()` vs `tensor()`, unsupported ids |
| `./_builders.py` | shared synthetic-file builders/reference dequantizer (not collected) |
