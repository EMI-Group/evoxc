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
| `SafetensorsFile` | class | `metadata: dict[str, str]`, `tensor_names() -> list[str]`, `info(name) -> TensorInfo`, `tensor(name) -> core.Tensor`, `as_dict() -> dict[str, core.Tensor]`, `close()`, context manager |
| `load_gguf(path, *, mmap=True) -> GGUFFile` | fn | parse GGUF v2/v3 header, metadata KV table and tensor infos; data mapped lazily |
| `GGUFFile` | class | `metadata` (typed KV mapping with `.get`/`keys`), `tensor_names()`, `info(name) -> TensorInfo`, `tensor(name) -> core.Tensor` (dequantized to a dense dtype where the stored type is quantized), `raw(name) -> numpy.ndarray` (stored payload, no dequantization), `close()`, context manager |
| `TensorInfo` | dataclass | `name`, `dtype` (canonical string), `shape: tuple[int, ...]`, `nbytes: int`, plus the source-format type identifier |

Both formats return `core.Tensor` values that wrap host numpy memory; a memory-mapped tensor is a READ-ONLY view and `.numpy()` exposes that read-only array (documented; copying is the caller's explicit choice).

## Binding rules

- Imports: `evoxc.core` only, plus stdlib and numpy. Never import another `evoxc` submodule, the package root or a backend.
- Byte layout is validated strictly: magic/version, header length bounds, every offset/length pair within the data section, alignment, no overlapping out-of-range tensors, unknown dtype identifier → `core.PersistenceError` naming the type and the offset. Never repair, pad or silently truncate.
- Never deserialize executable payloads (`pickle`, `torch.save`); these are data formats only.
- `mmap` is a performance choice, never a semantic one: `mmap=False` reads eagerly and returns an independent array.
- No new third-party runtime dependency: the distribution stays `numpy`-only.
- Files stay under ~1000 lines; split by format (`safetensors.py`, `gguf.py`) with shared byte/mmap helpers (`mmio.py`).
- GGUF dequantization covers the GGML types a real checkpoint uses at minimum: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, and the K-quants `Q2_K`/`Q3_K`/`Q4_K`/`Q5_K`/`Q6_K`; any other type raises `core.PersistenceError` naming the GGML type id, never a wrong value.
- safetensors dtype coverage: `F64`, `F32`, `F16`, `BF16`, `I64`, `I32`, `I16`, `I8`, `U64`, `U32`, `U16`, `U8`, `BOOL`, `F8_E4M3`, `F8_E5M2`; `BF16`/`F8_*` surface as raw `uint16`/`uint8` views with `TensorInfo.dtype` carrying the canonical name (evoxc has no bf16/f8 dtype).
- Writers only accept dtypes the format can represent; anything else is an explicit `core.PersistenceError`.

## Routing Table

| Path | Area |
|---|---|
| `./safetensors.py` | safetensors read (mmap, zero-copy) and canonical write; header validation. |
| `./gguf.py` | GGUF v2/v3 container parse, metadata KV table, tensor infos, data views and block dequantization. |
| `./mmio.py` | shared little-endian readers, bounds-checked mmap/read helpers and view construction. |
| `./__init__.py` | the public surface above. |
