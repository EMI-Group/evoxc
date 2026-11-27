# evoxc.formats — tensor-only binary file formats (safetensors, GGUF)

## Intent

Explicit readers/writers for the tensor half of modern model checkpoint formats, so a heavy tensor payload can enter and leave evoxc without `torch`/`transformers`/`safetensors`/`gguf` Python dependencies.
Scope is **tensors only**: tensor data, dtype, shape, names and the container metadata needed to find them.
Tokenizers, chat templates, generation config, quantization kernels for inference and any model-family semantics are out of scope (those stay in a model-addon distribution such as `evoxc_hf`).
Nothing here traces, compiles or moves data: a loaded tensor is ordinary host data (`Device('cpu', 0)`), and reads are explicit.

## API Surface

Exported from `evoxc.formats` (no top-level `evoxc.*` names are added):

| Name | Kind | Notes |
|---|---|---|
| `load_safetensors(path, *, mmap=True) -> SafetensorsFile` | fn | parse + validate; `mmap=True` maps the file and returns zero-copy read-only views |
| `save_safetensors(path, tensors, *, metadata=None)` | fn | canonical safetensors layout (u64 LE header length, JSON header, 8-byte-aligned data), atomic write |
| `SafetensorsFile` | class | `metadata: dict[str, str]`, `tensor_names() -> list[str]`, `info(name) -> TensorInfo`, `tensor(name) -> core.Tensor`, `as_dict() -> dict[str, core.Tensor]`, `close()`, context manager, `in`/`len`/iteration |
| `load_gguf(path, *, mmap=True) -> GGUFFile` | fn | parse GGUF v2/v3 header, metadata KV table and tensor infos; data mapped lazily |
| `GGUFFile` | class | `metadata` (dict with the typed KV values), `version`, `alignment`, `tensor_names()`, `info(name) -> TensorInfo`, `tensor(name) -> core.Tensor` (dequantized to a dense dtype where the stored type is quantized), `raw(name) -> numpy.ndarray` (stored payload bytes, no dequantization), `close()`, context manager, `in`/`len`/iteration |
| `TensorInfo` | dataclass | `name`, `dtype` (canonical string), `shape: tuple[int, ...]`, `nbytes: int`, `source_dtype` (the source-format type identifier), plus `numel`/`rank` properties |

Both formats return `core.Tensor` values that wrap host numpy memory; a memory-mapped tensor is a READ-ONLY view and `.numpy()` exposes that read-only array (documented; copying is the caller's explicit choice).
A zero-element tensor cannot alias file memory, so it is a fresh (writable) empty array in both modes.

## Semantics (binding)

- `TensorInfo.dtype` is the canonical *stored* dtype name: the numpy dtype name when numpy has a native equivalent (`"float32"`, `"int16"`, `"bool"`, ...), otherwise the format's own name (`"BF16"`, `"F8_E4M3"`, `"F8_E5M2"` for safetensors; `"BF16"`, `"Q4_K"`, ... for GGUF).
- `TensorInfo.source_dtype` is the source format's own identifier verbatim: a safetensors tag (`"F32"`, `"BF16"`, ...) or a GGML type name (`"Q4_0"`, `"Q6_K"`, ...).
- `TensorInfo.shape` is always the numpy row-major shape; GGUF stores dimensions fastest-axis-first (`ne[0]` first) and the reader reverses them, so `tensor(name).numpy().shape == info.shape`.
- `TensorInfo.nbytes` is the stored payload size: for block-quantized GGUF types `numel / block_size * type_size`.
- safetensors `tensor()` returns the stored dtype; `BF16` surfaces as a raw `uint16` view and `F8_E4M3`/`F8_E5M2` as raw `uint8` views (evoxc has no bf16/f8 dtype), with the canonical name in `TensorInfo.dtype`.
- GGUF `tensor()` returns a zero-copy view in its own dtype for `F32`/`F16`, widens `BF16` to `float32`, and dequantizes every block-quantized type to a fresh dense `float32` array; `raw(name)` always returns a 1-D `uint8` view of the stored payload.
- `mmap=False` is a performance choice, never a semantic one: it reads eagerly and every returned array (`tensor`, `raw`, `as_dict`) is an independent writable copy.
- The safetensors reader accepts a non-8-multiple header length (the offsets are validated independently) but the writer always pads the header to an 8-byte multiple and aligns each tensor start to 8 bytes inside the data buffer.
- The GGUF data-section start is `align_up(end_of_tensor_infos, alignment)`; a metadata-only GGUF (zero tensors, e.g. a tokenizer file) may end without that padding, and is accepted.
- Per-tensor GGUF offsets are bounds-checked and overlap-checked against the data section; they are not required to be individually aligned (a writer that aligns them is still fine).
- Files stay under ~1000 lines; split by format (`safetensors.py`, `gguf.py`) with the shared byte/mmap helpers in `mmio.py` and the shared descriptor in `tensorinfo.py`.

## Validation coverage and v1 boundaries

Reader validation (all raise `core.PersistenceError` naming the file, tensor and offset/range):

- safetensors: file/header length bounds, JSON object shape, known dtype tag, non-negative integer dims, exactly two integer data offsets, `0 <= begin <= end <= data_len`, byte extent equal to `numel * itemsize`, no two non-empty extents overlapping, `__metadata__` being `{str: str}` strings.
- GGUF: magic, version ∈ {2, 3}, bounds-checked string lengths (UTF-8), metadata value type ids, array element types (nested arrays rejected), power-of-two `general.alignment`, `n_dims <= 4`, known ggml type ids, `numel` divisible by the quantized block size, every payload inside the data section, no overlapping payloads, and a header count that cannot fit in the remaining bytes.
- Writer: dtypes the format cannot represent (complex, float128, object, strings, ...) and non-string metadata are rejected; accepted inputs are `core.Tensor` (a non-cpu tensor raises `core.DeviceError` from `.numpy()` — no implicit D2H), numpy arrays and memmaps.

Supported safetensors tags: 15 accepted, 12 writable — `F64`, `F32`, `F16`, `I64`, `I32`, `I16`, `I8`, `U64`, `U32`, `U16`, `U8`, `BOOL` read and write, while `BF16`, `F8_E4M3` and `F8_E5M2` are read-only in v1 (they surface as raw `uint16`/`uint8` views, so writing them back would need a tag override the API does not take).
Supported GGML type ids: F32 (0), F16 (1), Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q8_1 (9), Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), BF16 (30).
Explicit v1 deferrals (loud errors, never wrong values): every other ggml type id (Q8_K, the IQ family, TQ, MXFP4/NVFP4, Q1_0/Q2_0, I8..I64, F64, F8_E4M3), writing BF16/F8 safetensors (their raw storage words are indistinguishable from U16/U8 without a tag override the API does not take), GGUF v1 (32-bit string lengths), and big-endian GGUF files.

## Notes for agents

- Imports are `evoxc.core` plus stdlib and numpy only; never import the package root or another evoxc submodule, and never add a third-party dependency (the distribution stays numpy-only).
- Do not use `np.ascontiguousarray` on user arrays in the writer: it promotes 0-d (scalar) tensors to shape `(1,)`; check `flags.c_contiguous` first.
- Numpy refuses to close an `mmap` while `frombuffer` views are exported, so `MappedFile.close()` swallows that `BufferError` and lets the mapping die with the last view; the file descriptor closes immediately.
- Dequantization order is load-bearing: each GGML block layout (interleaved low/high nibbles, `qh` bit positions, K-quant 6-bit scale packing) was verified bit-exactly against `gguf-py`'s `dequantize` on random blocks and against real checkpoints — do not "simplify" a reshape without re-running that comparison.
- Public surface lives in `__init__.py`; adding a name means updating it and this contract together.

## Measured performance (this host: Intel Core Ultra 7 356H, 16 threads, numpy 2.5.2, Python 3.14.10)

- safetensors write: 65.7 MB / 2000 tensors in 59 ms (~1.1 GB/s); a single 256 MB tensor in 176 ms (~1.5 GB/s).
- safetensors read: `mmap=True` parse of 65.7 MB / 2000 tensors 5.5 ms, `tensor(name)` lookup+view 4.3 µs, all 2000 tensors 4.5 ms; `mmap=False` parse 31 ms (66 MB read).
- One 256 MB tensor: `mmap=True` parse 0.44 ms and a zero-copy `tensor()` 26 µs; `mmap=False` parse 103 ms and a 24 ms copy.
- GGUF parse (`stories15M-q4_0.gguf`, 19 MB, 57 tensors, 20 metadata keys): 46 ms including metadata/tokenizer-free tensor infos; a 67 MB synthetic file parses in ~0.12 ms (only headers are touched).
- GGUF dequantization on 67 MB payloads: Q4_0 470 Melem/s (264 MB/s), Q4_K 404 Melem/s (227 MB/s), Q6_K 376 Melem/s (308 MB/s), Q2_K 394 Melem/s (129 MB/s); the full `stories15M-q4_0` model (24.4 M elements) dequantizes in 37 ms.

## Validation evidence

- Synthetic GGUF v2 and v3 files built byte-by-byte cover every supported ggml type (random blocks with exactly-representable f16 scales) and all metadata value types (scalars, bools, strings, arrays incl. strings/bools/numbers and an empty array), at alignments 1/16/32/64/256: dequantized values are **bit-exact** against `gguf-py`'s independent `dequantize`, plus a quantize→dequantize round trip for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0.
- Real files: `stories15M-q4_0.gguf` (19 MB, 57 tensors) matches the `gguf-py` reference on every tensor; `ggml-vocab-llama-spm.gguf` (metadata-only, 32k tokenizer strings) parses; `tiny-random-LlamaForCausalLM/model.safetensors` (21 F32 tensors) matches the independent `evoxc_hf` reader.
- Corrupt-file rejection exercised for 11 safetensors cases (short file, header overflow, bad JSON, non-object header, unknown dtype naming its offset, byte-extent mismatch, out-of-bounds offsets, negative dim, reversed offsets, overlapping tensors, non-string metadata) and 12 GGUF cases (bad magic, versions 1/4, absurd counts, unknown value type, unsupported ggml id 15/Q8_K, unknown id 999, payload beyond the data section, overlapping payloads, `n_dims` 5, non-block-multiple element count, bad alignment, string-length overflow).

## Routing Table

| Path | Area |
|---|---|
| `./safetensors.py` | safetensors read (mmap, zero-copy) and canonical write; header validation. |
| `./gguf.py` | GGUF v2/v3 container parse, metadata KV table, tensor infos, data views and block dequantization. |
| `./mmio.py` | shared little-endian readers, bounds-checked mmap/read helpers and view construction. |
| `./tensorinfo.py` | the shared public `TensorInfo` descriptor dataclass. |
| `./__init__.py` | the public surface above. |
