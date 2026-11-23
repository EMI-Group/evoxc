//! `#[cfg(test)]` helpers shared by the `buffer` submodules' unit tests.
//!
//! Two groups live here: a garbage `Storage` no correct result can match (used
//! to prove a pooled result is fully overwritten), and the serialization of the
//! process-wide A/B switches ([`SPLIT_DISABLED`](super::rowsplit::SPLIT_DISABLED),
//! [`RUNS_DISABLED`](super::broadcast::RUNS_DISABLED)) the benchmark instruments
//! flip.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::pool::STORAGE_POOL;
use super::Storage;

/// A stock for the pool: `count` f32 elements' worth, filled with garbage
/// that no correct result can contain.
pub(super) fn garbage_f32(count: usize) -> Storage {
    let mut storage = Storage::F32(vec![f32::NAN; count]);
    STORAGE_POOL.with(|pool| {
        pool.borrow_mut().push(&mut storage);
    });
    storage
}

/// The ONE lock every test that reads or flips a process-wide A/B switch takes
/// (`SPLIT_DISABLED`, `RUNS_DISABLED`).
///
/// The switches are plain statics, so `cargo test`'s parallel threads share
/// them: a test measuring the reference path, or asking whether the row split
/// engages at all, must not observe a switch a sibling test set for its own
/// measurement.  Every such test holds this lock for its whole body (the
/// instruments in `bench_broadcast` included), which makes each one run as if
/// it were alone in the process.
pub(super) fn switch_guard() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// A drop guard that forces one A/B switch for its lifetime and restores the
/// previous value on drop — including on a panic, so a failing test can never
/// leave a switch flipped for the rest of the process.
///
/// The caller must hold [`switch_guard`]: the switch is process-wide, so a
/// forced value is only meaningful while every other reader is excluded.
pub(super) struct Switch<'a> {
    flag: &'a AtomicBool,
    previous: bool,
}

impl Drop for Switch<'_> {
    fn drop(&mut self) {
        self.flag.store(self.previous, Ordering::Relaxed);
    }
}

/// Force `flag` to `value` until the returned guard drops (see [`Switch`]).
pub(super) fn switch(flag: &AtomicBool, value: bool) -> Switch<'_> {
    Switch {
        flag,
        previous: flag.swap(value, Ordering::Relaxed),
    }
}
