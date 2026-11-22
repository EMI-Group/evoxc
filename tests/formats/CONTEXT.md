# tests/formats — `evoxc.formats` (safetensors + GGUF) test suite

## Intent

pytest suite for the tensor-only checkpoint readers/writers in `../../evoxc/formats/` (sibling — the API contract is `../../evoxc/formats/CONTEXT.md`).
Every fixture is built synthetically in `tmp_path` by the tests themselves (byte-for-byte writers / an independent reference encoder); no binary fixture is committed and no checkpoint is downloaded.
The quantized-value assertions are a genuine cross-check, not a re-run: `_builders.py` encodes the containers and decodes every GGML block layout with its own per-element loops, built from the published format definitions and never importing `evoxc`.

## Structure

- `safetensors` coverage lives in `test_safetensors.py`, GGUF in `test_gguf.py`, the package surface in `test_surface.py`.
- Byte-level constructors (safetensors header/data container, GGUF v2/v3 header + KV table + tensor infos) and the independent GGUF reference decoders/quantizers live in `_builders.py` (package-qualified import: `from tests.formats._builders import ...`).
- Generated quantized blocks use only power-of-two f16 scales, so reference values are exact in float32 and are compared with `assert_array_equal`; tolerances appear only in the reference-quantizer round trips.

## Constraints

- CPU only, no network, no third-party dependency beyond `numpy`/`pytest`.
- A real checkpoint file is only used when one is already present on the host, guarded by an explicit `pytest.skip` reason naming the host guard; never downloaded.
- Corruption tests assert `core.PersistenceError` and that the message names the file and tensor/offset.
- Keep the module surface pin (`test_surface.py`) in step with `evoxc/formats/__init__.py`: it asserts the exact `__all__`, that no `evoxc.*` top-level name is added, and that the package's imports are stdlib + numpy + `evoxc.core` only (parsed with `ast`, so it fails on a new dependency or a sibling-submodule import).

## Test strategy

- `test_safetensors.py`: dtype round-trips for all 12 writable tags (incl. 0-d/zero-element/multi-dim), metadata, multi-tensor and header-order pins, `mmap=True` read-only views vs `mmap=False` writable copies (plus the zero-element fresh-array exception), a hand-parsed container-layout pin, atomic writes, write-side rejections (incl. the BF16/F8 read-only degradation and a non-cpu `core.Tensor` → `DeviceError`), and the 11 documented corruption rejections.
- `test_gguf.py`: one known-value case per supported ggml id against `reference_values`, `raw()` vs `tensor()` view semantics, every metadata value type (scalars and arrays, incl. empty/string/bool), alignment 1/16/32/64/256, metadata-only files with and without padding, unaligned per-tensor offsets, and the documented rejections (magic/version/counts/value-type/ggml-id/payload-bounds/overlap/`n_dims`/block-multiple/alignment/string-overflow/nested-array/invalid-UTF-8).
- Memory-mapped views are asserted read-only (`flags.writeable`/`owndata` false, assignment raises); dequantized and eager-copy results are asserted writable and independent of the file.

## Routing table

| Path | Area |
|---|---|
| `./test_safetensors.py` | safetensors round-trips (every writable dtype, metadata, multi-tensor, mmap both ways, read-only views, atomic write) + the documented corruption rejections |
| `./test_gguf.py` | GGUF v2/v3 byte-built containers: every supported ggml type vs an independent reference dequantizer, metadata KV value types, alignment variants, `raw()` vs `tensor()`, reference-quantizer round trips, unsupported ids and the documented rejections |
| `./test_surface.py` | `evoxc.formats` public surface (`__all__`, descriptor module) and dependency direction (stdlib + numpy + `evoxc.core` only) |
| `./_builders.py` | shared synthetic-file builders, GGUF reference dequantizers/quantizers and the real-checkpoint locator (not collected) |
