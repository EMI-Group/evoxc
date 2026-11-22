//! The thread-local recycled-storage pool: retention caps, the LRU entry table,
//! `Storage`'s admission `Drop` and `take_pooled_vec`.

use std::cell::RefCell;

use evoxc_core::DType;

use crate::scalar::Scalar;

use super::Storage;

// ---------------------------------------------------------------------------
// Large-result allocation
// ---------------------------------------------------------------------------
//
// Every elementwise op over a multi-megabyte array produces a fresh `Vec` for
// its result, and those results die at their last use a step or two later.
// Left to the system allocator that is a page-fault treadmill: once a chain's
// live set grows past one temporary, glibc grows the heap, coalesces the freed
// temporaries back into the heap top, trims it again, and the *same* pages are
// re-faulted (and re-zeroed by the kernel) on the next call.  Measured on the
// `ew_chain_1e6` benchmark graph: 3874 minor faults and 9.4 ms per `run`, versus
// 0 faults and 4.2 ms when the churn is removed -- while the kernels themselves
// are already at numpy-parity speed (0.21 ms per 1e6-element flat map, i.e.
// DRAM-bandwidth bound, so there is nothing left to win in the loops).
//
// The engine therefore keeps its own small free list of large result storages:
// dropping a large enough [`Storage`] hands its allocation to a thread-local
// pool instead of back to `malloc`, and [`take_pooled_vec`] hands it out again
// to the next flat map that needs exactly that dtype and element count.  Those
// bytes then never leave the process, so they are never re-faulted.  This is
// plain safe Rust -- no `unsafe`, no `mallopt`, no allocator hook -- and it
// keeps `Buffer` an ordinary owning value: no kernel signature changes.
//
// Contract:
//  * A pooled vector is only handed to a caller that overwrites EVERY element
//    (the flat maps gate on an exact length match), so no stale value can reach
//    a result -- pinned by `recycled_results_are_fully_overwritten` below.
//  * The pool is thread-local: a value is only recycled on the thread that
//    dropped it, and nothing is shared or synchronized.
//  * Retention is bounded and never grows across unrelated calls: at most
//    [`POOL_MAX_ENTRIES`] values totalling [`POOL_MAX_BYTES`], and only values
//    of at least [`POOL_MIN_BYTES`] (small results keep the system allocator,
//    whose bin free lists already make them cheap, and would only crowd out the
//    entries that matter).
//  * The caps are enforced by evicting the LEAST RECENTLY USED entry, never by
//    refusing the incoming one: a pool that refuses new stocks once it is full
//    is poisoned for the rest of the thread's life by whichever workload filled
//    it.  Measured on this host, `gp_gather_4096` (a `(4096, 64)` f32 graph, so
//    1 MiB temporaries) runs a warm `run` in 0.55-0.60 ms with its own stocks
//    pooled and 1.5-2.1 ms when the pool instead holds sixteen 4 MiB stocks
//    from a preceding `ew_chain_1e6` -- the same engine, the same graph, only
//    the order of two unrelated benchmarks in one process; running
//    `gp_gather_4096` on a fresh thread (empty pool, same process heap) is fast
//    again, which is what pinned the poison on the pool rather than on malloc.
//    LRU eviction keeps the stocks of whatever is RUNNING and discards the ones
//    that stopped being used, so the numbers are order-independent.
//    `a_full_pool_evicts_instead_of_poisoning_later_workloads` pins it.
pub(super) const POOL_MIN_BYTES: usize = 1 << 12;
/// Total bytes the pool may retain per thread (the documented retention cap).
pub(super) const POOL_MAX_BYTES: usize = 64 << 20;
/// Number of retained storages per thread, across every size class.
pub(super) const POOL_MAX_ENTRIES: usize = 64;
/// Stocks retained per exact `(dtype, element count)` size class.
///
/// A graph reuses a couple of temporaries per size at most (one live result plus
/// the one it replaced), and a size-blind cap makes a graph with several
/// concurrent sizes thrash: measured on `gp_gather_4096` (whose 80 ops touch
/// four sizes: 1 MiB f32, 16 KiB f32, 4 KiB bool, 256 B f32), a flat
/// 16-entry cap produced 17 evictions and 8 allocation misses per run -- twice
/// the stocks the run actually needs -- and the churn is what made the row
/// order-dependent.  Keeping a few spares per size costs a bounded amount of
/// memory (<= `POOL_MAX_PER_SIZE` x the size class) and makes the steady state
/// allocation-free.
///
/// The cap must never be enforced by FREEING the incoming stock.  A graph that
/// keeps N + 1 buffers of one size live at once would then drop one per run and
/// re-allocate it on the next run, and whether `malloc` serves that megabyte
/// from its own free lists or with a fresh `mmap` + zero-fill depends on the
/// unrelated allocations earlier in the process -- exactly the order-dependent
/// behaviour this pool exists to remove.  A full size class therefore evicts its
/// OWN oldest stock and retains the newest, so a running workload always has
/// `POOL_MAX_PER_SIZE` of its own buffers and never a malloc round trip.
pub(super) const POOL_MAX_PER_SIZE: usize = 8;

thread_local! {
    pub(super) static STORAGE_POOL: RefCell<StoragePool> = const { RefCell::new(StoragePool::new()) };
    /// Cleared once the pool's own destructor starts, so a retained value that
    /// is dropped during thread teardown frees itself instead of re-entering the
    /// pool it is a part of (a `Cell<bool>` has no destructor, so it stays
    /// reachable for as long as the thread lives).
    static POOL_OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// One retained allocation plus the tick of its last use, which is what the
/// eviction policy orders by (smaller = older = evicted first).
struct PoolEntry {
    storage: Storage,
    last_use: u64,
}

/// The thread-local free list of large result storages.
pub(super) struct StoragePool {
    entries: Vec<PoolEntry>,
    bytes: usize,
    /// Monotonic use counter stamped into [`PoolEntry::last_use`]; `u64` never
    /// wraps in a process's lifetime (one tick per retained or served value).
    tick: u64,
}

impl StoragePool {
    const fn new() -> StoragePool {
        StoragePool { entries: Vec::new(), bytes: 0, tick: 0 }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    /// Retain `storage`'s allocation, returning `false` (and leaving `storage`
    /// intact, so the caller frees it as before) when it cannot usefully be
    /// retained: it is outside `POOL_MIN_BYTES .. POOL_MAX_BYTES`, or
    /// [`POOL_MAX_PER_SIZE`] stocks of its exact size are already retained.
    ///
    /// Room within the global caps is made by evicting the least recently used
    /// entry of ANOTHER size class, so a workload's own stocks are never the
    /// victim of its own churn, and a saturated pool of a different size can
    /// never starve the workload that is running (see the contract block above).
    pub(super) fn push(&mut self, storage: &mut Storage) -> bool {
        let bytes = storage.bytes();
        if !(POOL_MIN_BYTES..=POOL_MAX_BYTES).contains(&bytes) {
            return false;
        }
        let dtype = storage.dtype();
        let len = storage.len();
        if self.count_of(dtype, len) >= POOL_MAX_PER_SIZE {
            // A full size class keeps its newest stocks: evict this class's own
            // oldest spare rather than freeing the incoming one, so a workload
            // that keeps one more buffer of this size in flight than it retains
            // never pays a malloc round trip per run.
            if !self.evict_same_size(dtype, len) {
                return false;
            }
        }
        while self.entries.len() >= POOL_MAX_ENTRIES || self.bytes + bytes > POOL_MAX_BYTES {
            if !self.evict_for(dtype, len) {
                // Nothing of another size can be evicted (only possible when the
                // value alone exceeds the byte cap, excluded above), so the pool
                // keeps what it has and this stock is freed.
                return false;
            }
        }
        let last_use = self.next_tick();
        let retained = std::mem::replace(storage, Storage::Bool(Vec::new()));
        self.bytes += bytes;
        self.entries.push(PoolEntry { storage: retained, last_use });
        true
    }

    /// How many stocks of exactly `dtype` and `len` elements are retained.
    fn count_of(&self, dtype: DType, len: usize) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.storage.dtype() == dtype && entry.storage.len() == len)
            .count()
    }

    /// Evict the least recently used entry of exactly `dtype` and `len`, so the
    /// incoming stock can take its place.  False when the class is empty.
    fn evict_same_size(&mut self, dtype: DType, len: usize) -> bool {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.storage.dtype() == dtype && entry.storage.len() == len)
            .min_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index);
        let Some(index) = victim else { return false };
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.storage.bytes();
        true
    }
    /// Evict the least recently used entry that is NOT of the incoming size
    /// class (falling back to the plain LRU entry when every entry is).
    fn evict_for(&mut self, dtype: DType, len: usize) -> bool {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !(entry.storage.dtype() == dtype && entry.storage.len() == len))
            .min_by_key(|(_, entry)| entry.last_use)
            .or_else(|| {
                self.entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, entry)| entry.last_use)
            })
            .map(|(index, _)| index);
        let Some(index) = victim else { return false };
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.storage.bytes();
        true
    }

    /// Take a retained storage of exactly `dtype` and `len` elements.
    fn take(&mut self, dtype: DType, len: usize) -> Option<Storage> {
        let matching = |entry: &PoolEntry| entry.storage.dtype() == dtype && entry.storage.len() == len;
        // LIFO: hand back the MOST recently used matching stock, so a graph
        // cycles through a couple of hot buffers instead of walking its whole
        // retained size class and touching memory that has gone cold.
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| matching(entry))
            .max_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index)?;
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.storage.bytes();
        Some(entry.storage)
    }

}

impl Drop for StoragePool {
    fn drop(&mut self) {
        let _ = POOL_OPEN.try_with(|open| open.set(false));
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.bytes() < POOL_MIN_BYTES {
            return;
        }
        if !POOL_OPEN.try_with(|open| open.get()).unwrap_or(false) {
            return;
        }
        // On refusal (or during thread teardown) the allocation simply stays
        // here and is freed by the normal field drop that follows this call.
        let _ = STORAGE_POOL.try_with(|pool| match pool.try_borrow_mut() {
            Ok(mut pool) => pool.push(self),
            Err(_) => false,
        });
    }
}

/// A retained result vector of exactly `count` elements of `R`, or `None`.
///
/// The caller must overwrite every element before the vector becomes visible.
pub(crate) fn take_pooled_vec<R: Scalar>(count: usize) -> Option<Vec<R>> {
    let storage = STORAGE_POOL
        .try_with(|pool| pool.try_borrow_mut().ok()?.take(R::DTYPE, count))
        .ok()
        .flatten();
    match storage {
        Some(mut storage) => R::take_vec(&mut storage),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{map1, Storage};
    use evoxc_core::DType;

    /// Empty this thread's pool so a policy test starts from a known state
    /// (tests on the same thread share the thread-local).
    fn reset_pool() {
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.entries.clear();
            pool.bytes = 0;
        });
    }

    /// `count` f32 elements of `value` as a raw stock the pool can retain.
    fn pool_stock_f32(count: usize, value: f32) -> Storage {
        Storage::F32(vec![value; count])
    }

    #[test]
    fn stocks_are_retained_per_size_up_to_the_per_size_cap() {
        reset_pool();
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut stocks: Vec<Storage> = (0..POOL_MAX_PER_SIZE * 3)
            .map(|_| pool_stock_f32(count, 1.0))
            .collect();
        let mut retained = 0;
        for stock in stocks.iter_mut() {
            if STORAGE_POOL.with(|pool| pool.borrow_mut().push(stock)) {
                retained += 1;
            }
        }
        assert_eq!(retained, POOL_MAX_PER_SIZE * 3, "a full class evicts its own oldest stock");
        let kept = STORAGE_POOL.with(|pool| pool.borrow().count_of(DType::Float32, count));
        assert_eq!(kept, POOL_MAX_PER_SIZE, "one size class keeps at most its cap");
        reset_pool();
    }

    #[test]
    fn a_full_size_class_keeps_its_newest_stocks() {
        // The class is at its cap; pushing a fourth stock evicts the OLDEST of
        // that class (never frees the incoming one, never touches another
        // size), so a workload that keeps one buffer more in flight than it
        // retains never pays a malloc round trip per run.
        reset_pool();
        let count = (1 << 16) / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        let mut stock = pool_stock_f32(count / 2, 9.0);
        assert!(pool.push(&mut stock), "another size class is retained");
        for value in 0..POOL_MAX_PER_SIZE {
            let mut stock = pool_stock_f32(count, value as f32);
            assert!(pool.push(&mut stock), "stock {value} fits its own class");
        }
        let mut extra = pool_stock_f32(count, 42.0);
        assert!(pool.push(&mut extra), "a full class makes room instead of freeing");
        assert_eq!(pool.count_of(DType::Float32, count), POOL_MAX_PER_SIZE);
        assert_eq!(pool.count_of(DType::Float32, count / 2), 1, "another size class is untouched");
        // The newest stock is the one the next take serves (LIFO), so the
        // evicted entry is the oldest: the values prove which survived.
        let taken = pool.take(DType::Float32, count).unwrap();
        match &taken {
            Storage::F32(values) => assert_eq!(values[0], 42.0, "the newest stock is served first"),
            _ => panic!("expected f32 storage"),
        }
    }

    #[test]
    fn the_pool_evicts_the_least_recently_used_other_size_class() {
        // Fill the pool to its entry cap with DISTINCT sizes (so no per-size cap
        // triggers) and then push one more size: the victim must be the least
        // recently used entry of another size class, and never a stock of the
        // incoming size.
        let base = POOL_MIN_BYTES / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        for index in 0..POOL_MAX_ENTRIES {
            let mut stock = pool_stock_f32(base + index, index as f32);
            assert!(pool.push(&mut stock), "distinct size {index} fits");
        }
        assert_eq!(pool.entries.len(), POOL_MAX_ENTRIES);
        // Two stocks of one new size are retained -- the second only after
        // evicting a different size class (there is no room otherwise).
        let mut first = pool_stock_f32(base + POOL_MAX_ENTRIES + 1, -1.0);
        assert!(pool.push(&mut first));
        let mut second = pool_stock_f32(base + POOL_MAX_ENTRIES + 1, -2.0);
        assert!(pool.push(&mut second));
        assert_eq!(pool.entries.len(), POOL_MAX_ENTRIES);
        assert_eq!(pool.count_of(DType::Float32, base + POOL_MAX_ENTRIES + 1), 2);
        // The oldest sizes went first, and no entry ever exceeds the caps.
        assert_eq!(pool.count_of(DType::Float32, base), 0, "the oldest stock was evicted");
        assert_eq!(pool.count_of(DType::Float32, base + 1), 0);
        assert!(pool.bytes <= POOL_MAX_BYTES);
        // ... while a size that is still represented survives.
        assert_eq!(pool.count_of(DType::Float32, base + POOL_MAX_ENTRIES - 1), 1);
    }

    #[test]
    fn a_stock_larger_than_the_byte_cap_is_refused_and_evicts_nothing() {
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        let mut stock = pool_stock_f32(count, 1.0);
        assert!(pool.push(&mut stock));
        let mut huge = pool_stock_f32(POOL_MAX_BYTES / std::mem::size_of::<f32>() + 1, 2.0);
        assert!(!pool.push(&mut huge), "a value above the byte cap is never retained");
        assert_eq!(pool.entries.len(), 1);
        assert!(pool.take(DType::Float32, count).is_some(), "the small stock was not evicted");
    }

    #[test]
    fn a_differently_sized_predecessor_does_not_starve_later_workloads() {
        // The order-dependent regression this pins: `ew_chain_1e6`-shaped work
        // (4 MiB temporaries) fills the pool, and a following `gp_gather_4096`
        // (1 MiB temporaries) then never got a single stock back, which cost it
        // 2.5x on a warm run.  Filling the pool with the wrong size class must
        // not stop the next map from recycling ITS result.
        reset_pool();
        let big = (4 << 20) / std::mem::size_of::<f32>();
        let mut stocks: Vec<Storage> = (0..POOL_MAX_PER_SIZE)
            .map(|_| pool_stock_f32(big, f32::NAN))
            .collect();
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            for stock in stocks.iter_mut() {
                assert!(pool.push(stock), "the pool has room for these stocks");
            }
        });
        let count = (1 << 20) / std::mem::size_of::<f32>();
        assert!(take_pooled_vec::<f32>(count).is_none(), "nothing of this size is pooled yet");
        let source = vec![0.5f32; count];
        let result = map1(DType::Float32, &[count], &source, |value| value + 1.0)
            .expect("the map succeeds without a pooled stock");
        drop(result);
        assert!(
            take_pooled_vec::<f32>(count).is_some(),
            "a 4 MiB-filled pool must still recycle a 1 MiB result"
        );
        STORAGE_POOL.with(|pool| {
            let pool = pool.borrow();
            assert_eq!(pool.count_of(DType::Float32, big), POOL_MAX_PER_SIZE);
        });
        reset_pool();
    }

    #[test]
    fn stock_reuse_is_last_in_first_out() {
        // LIFO reuse keeps a graph cycling through the same few hot buffers
        // instead of walking its whole retained size class and touching memory
        // that has gone cold.
        reset_pool();
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut first = pool_stock_f32(count, 1.0);
        let mut second = pool_stock_f32(count, 2.0);
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            assert!(pool.push(&mut first));
            assert!(pool.push(&mut second));
        });
        let taken = take_pooled_vec::<f32>(count).expect("a stock of this size is pooled");
        assert_eq!(taken[0], 2.0, "the most recently freed stock is handed back first");
        reset_pool();
    }

    #[test]
    fn the_pool_keeps_only_large_values_and_respects_its_cap() {
        let small = Storage::F32(vec![0.0; POOL_MIN_BYTES / std::mem::size_of::<f32>() - 1]);
        let mut small = small;
        STORAGE_POOL.with(|pool| {
            pool.borrow_mut().push(&mut small);
        });
        let (entries, bytes) = STORAGE_POOL.with(|pool| {
            let pool = pool.borrow();
            (pool.entries.len(), pool.bytes)
        });
        assert_eq!((entries, bytes), (0, 0), "a small value must not be retained");
        // Then overfill: the caps hold.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 2;
        let mut values: Vec<Storage> = (0..POOL_MAX_ENTRIES * 2)
            .map(|_| Storage::F32(vec![1.0; count]))
            .collect();
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            for value in values.iter_mut() {
                pool.push(value);
            }
            assert!(pool.entries.len() <= POOL_MAX_ENTRIES);
            assert!(pool.bytes <= POOL_MAX_BYTES);
        });
    }
}
