//! `#[cfg(test)]` helper shared by the `buffer` submodules' unit tests: a
//! garbage `Storage` no correct result can match.

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
