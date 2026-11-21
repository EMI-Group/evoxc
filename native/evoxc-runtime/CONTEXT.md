# evoxc-runtime — the native runtime skeleton

## Intent

`evoxc-runtime` is the native home of the runtime *ownership model*: sessions, device contexts, buffers, submissions, completions and executables.
It exists so the Python runtime contract (`../evoxc/CONTEXT.md` "Device kinds", the `Tensor`/`Executable` placement rules, `prefetch`/`sink`/`synchronize` semantics) has a compiling, testable Rust shape BEFORE any device driver exists: the invariants live in types, and a future adapter fills in the driver side without redesigning lifetimes.
It is deliberately a skeleton with **real types and real invariants** — no empty functions that pretend to work — and it is the only runtime crate: it never depends on `evoxc-compiler` or PyO3, and it never sees a Python object.

Binding workspace contract: `../CONTEXT.md` (§12/§13 of `../../NATIVE_ARCHITECTURE_PROPOSAL.md` is the design source, this file is the crate contract).

## Ownership model

```text
Session ──owns──> Arc<SessionResources> { id, backend tag, closed flag, Arc<SubmissionTracker> }
   └──owns──> DeviceContext (one per served device)
                  └──leases Arc<SessionResources>
Buffer ──leases──> DeviceContext (+ BufferStorage + ValueType + readiness)
Executable ──leases──> DeviceContext (+ ProgramMetadata/ExecutableAbi)
Submission ──retains──> Vec<Buffer> inputs, owns outputs, one Status
Completion ──observes──> the same Arc<SubmissionInner> as its Submission
```

The cycle-breaking shape is what makes the §12 lifetime rules real: `DeviceContext` leases `SessionResources` strongly, while `Session` *also* owns its contexts and holds the same `Arc<SessionResources>`.
Consequences, each pinned by a test:

- Dropping the public `Session` handle does not unload anything a context, buffer or executable still leases.
- Cloning the handle shares one session identity (`SessionId`, one context per device, one tracker).
- `Session::close` refuses while work is pending and, once closed, only refuses **new** submissions; leases keep working for reads and transfers.
- An output buffer owns its storage, context and session lease independently of the executable that produced it.

## API surface

| Type | Module | Key operations |
|---|---|---|
| `Session`, `SessionConfig`, `SessionId`, `BackendTag` | `session.rs` | `Session::open(config)` (single validation gate: devices, label, retention), `device_context(&Device)`, `devices`/`contexts`, `close`, `pending_submissions`, `synchronize` (session scope) |
| `Device`, `DeviceKind`, `DeviceContext` | `device.rs` | `Device::new(kind, index)` (validated), `FromStr`/`Display` (`rocm:1`), `same_physical_device` (hip↔rocm), `DeviceContext::{same_context, compatible_with, check_transfer}` |
| `Buffer`, `BufferId`, `BufferStorage`, `DeviceHandle`, `Readiness`, `DependencyToken` | `buffer.rs` | `host`, `host_zeroed`, `from_handle`, `alloc`, `element_count`, `byte_len`, `is_host_resident`, `readiness`, `host_bytes`, `device_handle` |
| `Submission`, `Completion`, `Status`, `SubmissionId`, `SubmissionTracker` | `submission.rs` | `Executable::submit` (internal `Submission::new`), `completion()`, `poll`/`wait`, `complete_success`/`complete_failure`, bounded tracker (`capacity`, `len`, `pending_count`, `pruned`) |
| `Executable`, `ExecutableAbi`, `ProgramMetadata`, `ExecutableId` | `executable.rs` | `Executable::load(metadata, context)`, `submit(inputs) -> (Vec<Buffer>, Completion)`, `synchronize` (executable scope), `pending_submissions` |
| `Placement`, `check_transfer` | `placement.rs` | `SameContext` / `CrossContextImport` / `HostUpload` / `HostReadback`, `moves_data`, `is_host_staged` |

## Invariants (enforced, not documented-and-hoped)

1. **No implicit movement.** `Buffer::host` refuses a non-cpu-kind context; `host_bytes()` on device storage is a `DeviceError` telling the caller to transfer explicitly; `check_transfer` is the only classifier and errors (never falls back) for a cross-device or cross-backend import.
2. **No implicit host copies at the run boundary.** `Executable::submit` returns output handles on the executable's device; it never downloads anything for the caller.
3. **Pending is unreadable.** A submission arms its outputs `Pending(submission id)`; reading one raises an explicit `EVOXCError` naming the dependency, and a failed submission leaves its outputs pending forever while the failure stays observable through `Completion::wait`/`poll`.
4. **Retention is scoped.** A submission retains its input buffers until it settles and releases them on both outcomes; dropping the caller's tensor handle early is safe.
5. **Completion retention is bounded.** The tracker keeps at most `capacity` *settled* submissions and never evicts a pending one; `synchronize` over the retained window cannot require an ever-growing list.
6. **Scope is a snapshot.** `Executable::synchronize` covers only submissions issued through that executable before the call; `Session::synchronize` covers the whole session.
7. **Placement is exact.** A wrong device, a same-device foreign session, or a same-device foreign backend input is an explicit `DeviceError` naming both sides; `hip:0` and `rocm:0` are never silently reconciled (Device equality is spelling-sensitive, physical identity is not).
8. **Storage matches the type.** Host byte length must equal `element_count × itemsize` (`ShapeError`); a symbolic or dynamic dimension is rejected at allocation (`ShapeError`), never guessed.
9. **Values, not panics.** Every fallible path is `Result<_, evoxc_core::Error>` (`DeviceError`/`BackendError`/`ShapeError`/`DTypeError`/`General` = `EVOXCError`); there is no `unsafe`, no panic in `Drop`, no `unwrap`/`expect` on user input, and mutex poisoning is recovered instead of unwrapped.

## Known gaps (intentional)

- **No drivers, allocators, queues or kernels.** Device buffers are either host `Vec<u8>` or an opaque `DeviceHandle` (a `placeholder()` handle with no resource id for unallocated outputs); nothing dereferences a handle.
- **No execution.** A submission is settled by the side that created it (`complete_success`/`complete_failure`, reachable from the returned `Completion`); `wait`/`synchronize` on still-pending work is an explicit `BackendError`, never a hang or a fake success.
- **Static shapes only for storage.** Buffers and output types must be statically sized, so a symbolic *output* type is rejected at `Executable::load` (symbolic *inputs* are accepted: only dims already known are checked).
- **No `MappedView` / `ForeignExport` / DLPack surface.** Proposal §12/§15 objects that require a real payload are not modelled yet.
- **`Session::close` is cooperative.** It checks the session tracker under the close lock, but a submission created concurrently on another thread can land after the check; close is an explicit quiescence request, not a memory-safety barrier.
- **Thread safety is not asserted.** The skeleton uses `std::sync` primitives but makes no `Send + Sync` promise over backend handles (proposal §12: keep that adapter-qualified).
- **Relationship to the Python contract.** `../evoxc/CONTEXT.md` remains authoritative for behavior until a Python path delegates natively; the port must match this ownership model, not invent a second one.

## Test strategy

Unit tests live in-file (`#[cfg(test)]`) per module and run with `cargo test --manifest-path native/Cargo.toml -p evoxc-runtime` (61 tests; no GPU, no network, no Python).
They cover: session identity/default device resolution/config validation/handle-drop leases (`Arc::strong_count`/`Weak`, plus `#[cfg(test)]` lease helpers), context rejection and spelling sensitivity, buffer sizing/`ShapeError`/host-vs-device storage/pending reads, submission retention and release (weak-handle proof), first-outcome-wins settling, bounded retention with pending entries never evicted, completion `poll`/`wait` semantics, executable ABI errors (arity/dtype/rank/dim), placement/foreign-session/foreign-backend rejection, synchronize scope per executable vs per session, session close semantics, and all four transfer classifications.
Test-only helpers (`Buffer::lease_count`/`downgrade`, `DeviceContext::lease_count`, `Session::downgrade`) are `#[cfg(test)]`, never part of the built library.

## Routing Table

| Path | Area |
|---|---|
| `./src/lib.rs` | Crate docs (ownership model + invariants), module declarations, public re-exports, `VERSION`, the poisoning-safe `lock` helper. |
| `./src/session.rs` | `Session`/`SessionConfig`/`SessionId`/`BackendTag`, `SessionResources` (the lease target), open/close/synchronize. |
| `./src/device.rs` | `Device`/`DeviceKind` (validated kind strings), `DeviceContext` identity and lease. |
| `./src/buffer.rs` | `Buffer`/storage/readiness/`ValueType` metadata, static extent checks. |
| `./src/submission.rs` | `Submission`/`Completion`/`Status`/`SubmissionTracker` (bounded retention, wait scope). |
| `./src/executable.rs` | `Executable`/`ExecutableAbi`/`ProgramMetadata`, `submit` validation + enqueue. |
| `./src/placement.rs` | `Placement` outcomes and `check_transfer`. |
