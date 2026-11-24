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
//  * Retention is bounded PER TIER and never grows across unrelated calls.
//    The SMALL tier keeps values of at least [`POOL_MIN_BYTES`] and below
//    [`POOL_LARGE_MIN_BYTES`]: at most [`POOL_MAX_ENTRIES`] of them totalling
//    [`POOL_MAX_BYTES`], and at most [`POOL_MAX_PER_SIZE`] per exact size class.
//    The LARGE tier keeps values at or above [`POOL_LARGE_MIN_BYTES`]: at most
//    [`POOL_LARGE_PER_SIZE`] per exact size class, totalling
//    [`POOL_MAX_BYTES_LARGE`].
//    Values below [`POOL_MIN_BYTES`] always keep the system allocator, whose
//    bin free lists already make them cheap, so they cannot crowd out the
//    entries that matter.
//  * Eviction never crosses tiers: a small stock displaces only small stocks
//    and a large stock only large ones, so the workload that just ran keeps its
//    stocks whatever the following workload is -- the order-independent
//    contract above, extended to the two tiers.
//  * Within a tier the caps are enforced by evicting the LEAST RECENTLY USED
//    entry of that tier, never by refusing the incoming one: a pool that
//    refuses new stocks once it is full is poisoned for the rest of the
//    thread's life by whichever workload filled it.  Measured on this host,
//    `gp_gather_4096` (a `(4096, 64)` f32 graph, so
//    1 MiB temporaries) runs a warm `run` in 0.55-0.60 ms with its own stocks
//    pooled and 1.5-2.1 ms when the pool instead holds sixteen 4 MiB stocks
//    from a preceding `ew_chain_1e6` -- the same engine, the same graph, only
//    the order of two unrelated benchmarks in one process; running
//    `gp_gather_4096` on a fresh thread (empty pool, same process heap) is fast
//    again, which is what pinned the poison on the pool rather than on malloc.
//    LRU eviction keeps the stocks of whatever is RUNNING and discards the ones
//    that stopped being used, so the numbers are order-independent.
//    `a_full_pool_evicts_instead_of_poisoning_later_workloads` pins it.
//  * A stock larger than its tier's byte cap is refused outright (the caller
//    frees it); that can only happen for an incoming stock above
//    [`POOL_MAX_BYTES_LARGE`].
//  * The pool backs every full-overwrite destination, including `Storage::clone`
//    and the public `Storage::from_slice_pooled` (the PyO3 binding's input
//    hand-off), both through `storage::copy_slice_pooled`.
//
// Measured on this host: a fresh 64 MiB `vec![0.0; 16M]` plus one full write
// pass costs 26.0 ms (0.4 ms per MiB) -- mmap page faults, not copy -- and a flat
// `a * b` over 16M f32 measures 5.71 ms with the pool warm against 25.96 ms
// with the pool emptied before every call, so a 64 MiB result the pool cannot
// serve wastes ~20 ms of zero-fill-then-overwrite per op.
// A single flat 64 MiB cap retained at most ONE 64 MiB stock, while a real
// large graph keeps 2 inputs + 1 result (or a whole chain) live per run and so
// re-faulted ~2 x 64 MiB (~52 ms) on every run; one large size class at
// [`POOL_LARGE_PER_SIZE`] (4 x 64 MiB = 256 MiB) covers that live set with one
// spare.
pub(super) const POOL_MIN_BYTES: usize = 1 << 12;
/// Total bytes the SMALL tier may retain per thread (its documented cap).
pub(super) const POOL_MAX_BYTES: usize = 64 << 20;
/// Stocks at or above this size belong to the LARGE tier, with its own caps.
pub(super) const POOL_LARGE_MIN_BYTES: usize = 8 << 20;
/// Total bytes the LARGE tier may retain per thread (its documented cap).
pub(super) const POOL_MAX_BYTES_LARGE: usize = 256 << 20;
/// Stocks retained per exact size class within the LARGE tier.
///
/// 4 x 64 MiB = 256 MiB covers the 2 inputs + 1 result (+ 1 spare) a real large
/// graph keeps live per run, which is what removes the repeated first-touch
/// faults the one-flat-cap policy paid (see the measured numbers above).
pub(super) const POOL_LARGE_PER_SIZE: usize = 4;
/// Number of retained storages per thread, within the SMALL tier.
pub(super) const POOL_MAX_ENTRIES: usize = 64;
/// Stocks retained per exact `(dtype, element count)` size class in the SMALL
/// tier (the LARGE tier uses [`POOL_LARGE_PER_SIZE`]).
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

/// The thread-local free list of result storages, tiered by size.
pub(super) struct StoragePool {
    entries: Vec<PoolEntry>,
    /// Bytes retained in the SMALL tier (< [`POOL_LARGE_MIN_BYTES`] per stock).
    small_bytes: usize,
    /// Bytes retained in the LARGE tier (>= [`POOL_LARGE_MIN_BYTES`] per stock).
    large_bytes: usize,
    /// Monotonic use counter stamped into [`PoolEntry::last_use`]; `u64` never
    /// wraps in a process's lifetime (one tick per retained or served value).
    tick: u64,
}

impl StoragePool {
    const fn new() -> StoragePool {
        StoragePool { entries: Vec::new(), small_bytes: 0, large_bytes: 0, tick: 0 }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    /// Whether a retained value of this size belongs to the LARGE tier.
    fn is_large(storage: &Storage) -> bool {
        storage.bytes() >= POOL_LARGE_MIN_BYTES
    }

    /// `(byte cap, per-size cap, entry cap)` of the tier `large` selects.
    ///
    /// The large entry cap is implied by its byte cap: at least
    /// [`POOL_LARGE_MIN_BYTES`] bytes per stock means at most
    /// [`POOL_MAX_BYTES_LARGE`] / [`POOL_LARGE_MIN_BYTES`] of them can fit.
    const fn tier_caps(large: bool) -> (usize, usize, usize) {
        if large {
            (
                POOL_MAX_BYTES_LARGE,
                POOL_LARGE_PER_SIZE,
                POOL_MAX_BYTES_LARGE / POOL_LARGE_MIN_BYTES,
            )
        } else {
            (POOL_MAX_BYTES, POOL_MAX_PER_SIZE, POOL_MAX_ENTRIES)
        }
    }

    /// Bytes this pool retains in the tier `large` selects.
    fn tier_bytes(&self, large: bool) -> usize {
        if large {
            self.large_bytes
        } else {
            self.small_bytes
        }
    }

    /// Number of retained entries in the tier `large` selects.
    fn tier_entries(&self, large: bool) -> usize {
        self.entries
            .iter()
            .filter(|entry| Self::is_large(&entry.storage) == large)
            .count()
    }

    /// Add or subtract `storage`'s bytes from its own tier's counter.
    fn account(&mut self, storage: &Storage, add: bool) {
        let large = Self::is_large(storage);
        let bytes = storage.bytes();
        let counter = if large {
            &mut self.large_bytes
        } else {
            &mut self.small_bytes
        };
        if add {
            *counter += bytes;
        } else {
            *counter -= bytes;
        }
    }

    /// Retain `storage`'s allocation, returning `false` (and leaving `storage`
    /// intact, so the caller frees it as before) when it cannot usefully be
    /// retained: it is below [`POOL_MIN_BYTES`], above its tier's byte cap, or
    /// its tier already holds its per-size cap of exactly this size.
    ///
    /// Room within a tier's caps is made by evicting the least recently used
    /// entry of that tier, so a workload's own stocks are never the victim of
    /// its own churn, and a saturated other tier can never starve the workload
    /// that is running (see the contract block above).
    pub(super) fn push(&mut self, storage: &mut Storage) -> bool {
        let bytes = storage.bytes();
        if bytes < POOL_MIN_BYTES {
            return false;
        }
        let large = bytes >= POOL_LARGE_MIN_BYTES;
        let (byte_cap, per_size, entry_cap) = Self::tier_caps(large);
        if bytes > byte_cap {
            // A single value above its tier's byte cap can never be retained.
            return false;
        }
        let dtype = storage.dtype();
        let len = storage.len();
        if self.count_of(dtype, len) >= per_size {
            // A full size class keeps its newest stocks: evict this class's own
            // oldest spare rather than freeing the incoming one, so a workload
            // that keeps one more buffer of this size in flight than it retains
            // never pays a malloc round trip per run.
            if !self.evict_same_size(dtype, len, large) {
                return false;
            }
        }
        while self.tier_entries(large) >= entry_cap || self.tier_bytes(large) + bytes > byte_cap {
            if !self.evict_for(dtype, len, large) {
                // Nothing of this tier can be evicted (only possible when the
                // value alone exceeds the byte cap, excluded above), so the pool
                // keeps what it has and this stock is freed.
                return false;
            }
        }
        let last_use = self.next_tick();
        let retained = std::mem::replace(storage, Storage::Bool(Vec::new()));
        self.account(&retained, true);
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

    /// Evict the least recently used entry of exactly `dtype` and `len` in the
    /// tier `large` selects, so the incoming stock can take its place.  False
    /// when the class is empty.
    fn evict_same_size(&mut self, dtype: DType, len: usize, large: bool) -> bool {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                Self::is_large(&entry.storage) == large
                    && entry.storage.dtype() == dtype
                    && entry.storage.len() == len
            })
            .min_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index);
        let Some(index) = victim else { return false };
        let entry = self.entries.swap_remove(index);
        self.account(&entry.storage, false);
        true
    }
    /// Evict the least recently used entry of the tier `large` selects that is
    /// NOT of the incoming size class (falling back to the tier's plain LRU
    /// entry when every entry is).
    fn evict_for(&mut self, dtype: DType, len: usize, large: bool) -> bool {
        let in_tier = |entry: &PoolEntry| Self::is_large(&entry.storage) == large;
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                in_tier(entry)
                    && !(entry.storage.dtype() == dtype && entry.storage.len() == len)
            })
            .min_by_key(|(_, entry)| entry.last_use)
            .or_else(|| {
                self.entries
                    .iter()
                    .enumerate()
                    .filter(|(_, entry)| in_tier(entry))
                    .min_by_key(|(_, entry)| entry.last_use)
            })
            .map(|(index, _)| index);
        let Some(index) = victim else { return false };
        let entry = self.entries.swap_remove(index);
        self.account(&entry.storage, false);
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
        self.account(&entry.storage, false);
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
    use crate::buffer::{map1, map2, Storage};
    use evoxc_core::DType;

    /// Empty this thread's pool so a policy test starts from a known state
    /// (tests on the same thread share the thread-local).
    fn reset_pool() {
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.entries.clear();
            pool.small_bytes = 0;
            pool.large_bytes = 0;
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
        assert!(pool.small_bytes <= POOL_MAX_BYTES);
        // ... while a size that is still represented survives.
        assert_eq!(pool.count_of(DType::Float32, base + POOL_MAX_ENTRIES - 1), 1);
    }

    #[test]
    fn a_stock_above_its_tier_byte_cap_is_refused_and_evicts_nothing() {
        // The SMALL tier never holds a stock this large (anything at or above
        // POOL_LARGE_MIN_BYTES is classified large), so the refusal is a LARGE
        // stock above POOL_MAX_BYTES_LARGE.  `bytes()` counts a bool storage's
        // logical elements (one byte each) while `Vec<bool>` packs bits, so the
        // >256 MiB declared size here is only ~32 MiB of real allocation.
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        let mut stock = pool_stock_f32(count, 1.0);
        assert!(pool.push(&mut stock));
        let mut huge = Storage::Bool(vec![false; POOL_MAX_BYTES_LARGE + 1]);
        assert!(!pool.push(&mut huge), "a value above its tier's byte cap is never retained");
        assert_eq!(pool.entries.len(), 1);
        assert!(pool.take(DType::Float32, count).is_some(), "the small stock was not evicted");
    }

    #[test]
    fn a_large_size_class_retains_its_own_cap_and_evicts_its_oldest() {
        let mut pool = StoragePool::new();
        let count = POOL_LARGE_MIN_BYTES / std::mem::size_of::<f32>();
        for value in 0..POOL_LARGE_PER_SIZE + 3 {
            let mut stock = pool_stock_f32(count, value as f32);
            assert!(pool.push(&mut stock), "large stock {value} fits its own tier");
        }
        assert_eq!(
            pool.count_of(DType::Float32, count),
            POOL_LARGE_PER_SIZE,
            "a large size class keeps at most its own per-size cap"
        );
        assert!(pool.large_bytes <= POOL_MAX_BYTES_LARGE);
        // LIFO serve plus per-size eviction means the oldest stocks are gone and
        // the newest one is handed back first.
        let taken = pool.take(DType::Float32, count).expect("a large stock is pooled");
        match &taken {
            Storage::F32(values) => {
                assert_eq!(values[0], (POOL_LARGE_PER_SIZE + 2) as f32, "the newest stock is served first")
            }
            _ => panic!("expected f32 storage"),
        }
    }

    #[test]
    fn the_two_tiers_evict_only_within_their_own_tier() {
        let mut pool = StoragePool::new();
        let small = (1 << 20) / std::mem::size_of::<f32>();
        // A bool storage reports one byte per element while `Vec<bool>` packs
        // bits, so this 8 MiB-declared stock allocates ~1 MiB.
        let large = POOL_LARGE_MIN_BYTES;
        let mut small_stock = pool_stock_f32(small, 1.0);
        assert!(pool.push(&mut small_stock));
        let mut large_stock = Storage::Bool(vec![false; large]);
        assert!(pool.push(&mut large_stock));
        // Overflow the LARGE tier's per-size cap: only large stocks may go.
        for _ in 0..POOL_LARGE_PER_SIZE + 3 {
            let mut stock = Storage::Bool(vec![false; large]);
            assert!(pool.push(&mut stock), "the large tier makes room within itself");
        }
        assert_eq!(pool.count_of(DType::Float32, small), 1, "large churn left the small tier alone");
        assert_eq!(pool.count_of(DType::Bool, large), POOL_LARGE_PER_SIZE);
        let large_kept = pool.count_of(DType::Bool, large);
        // Overflow the SMALL tier's entry cap with distinct sizes: only small
        // stocks may go.
        for index in 0..POOL_MAX_ENTRIES + 10 {
            let mut stock = pool_stock_f32(POOL_MIN_BYTES / 4 + index, 2.0);
            assert!(pool.push(&mut stock), "the small tier makes room within itself");
        }
        assert_eq!(
            pool.count_of(DType::Bool, large),
            large_kept,
            "small churn left the large tier alone"
        );
        assert!(pool.tier_entries(false) <= POOL_MAX_ENTRIES);
        assert!(pool.small_bytes <= POOL_MAX_BYTES);
    }

    #[test]
    fn a_large_stock_at_the_large_byte_cap_evicts_the_lru_large_stock() {
        // Fill the large tier just below its 256 MiB byte cap with DISTINCT
        // 8 MiB classes (so no per-size cap triggers) and then push one more:
        // the incoming stock must displace the least recently used large stock,
        // never be refused.  `bytes()` counts a bool storage's logical elements
        // while `Vec<bool>` packs bits, so the ~256 MiB this reports is only
        // ~31 MiB of real allocation.
        let base = POOL_LARGE_MIN_BYTES;
        let classes = POOL_MAX_BYTES_LARGE / POOL_LARGE_MIN_BYTES - 1;
        let mut pool = StoragePool::new();
        for index in 0..classes {
            let mut stock = Storage::Bool(vec![false; base + index]);
            assert!(pool.push(&mut stock), "large class {index} fits below the byte cap");
        }
        assert!(pool.large_bytes <= POOL_MAX_BYTES_LARGE);
        let mut incoming = Storage::Bool(vec![false; base + classes]);
        assert!(
            pool.push(&mut incoming),
            "an at-cap large push evicts its LRU stock instead of freeing the incoming one"
        );
        assert!(pool.large_bytes <= POOL_MAX_BYTES_LARGE);
        assert_eq!(pool.count_of(DType::Bool, base), 0, "the oldest large stock was the victim");
        assert_eq!(
            pool.count_of(DType::Bool, base + classes),
            1,
            "the incoming stock was retained"
        );
    }

    #[test]
    fn a_stock_between_the_tiers_is_classified_and_capped_as_expected() {
        // One byte short of 8 MiB is a SMALL stock (8 per size class); exactly
        // 8 MiB is LARGE (4 per size class).
        let mut pool = StoragePool::new();
        let below = POOL_LARGE_MIN_BYTES - 1;
        let mut below_stocks: Vec<Storage> = (0..POOL_MAX_PER_SIZE + 2)
            .map(|_| Storage::Bool(vec![false; below]))
            .collect();
        for stock in below_stocks.iter_mut() {
            assert!(pool.push(stock), "a just-below-boundary stock is retained as a small one");
        }
        assert_eq!(pool.count_of(DType::Bool, below), POOL_MAX_PER_SIZE);
        let at = POOL_LARGE_MIN_BYTES;
        let mut at_stocks: Vec<Storage> = (0..POOL_LARGE_PER_SIZE + 2)
            .map(|_| Storage::Bool(vec![false; at]))
            .collect();
        for stock in at_stocks.iter_mut() {
            assert!(pool.push(stock), "the boundary stock is retained as a large one");
        }
        assert_eq!(pool.count_of(DType::Bool, at), POOL_LARGE_PER_SIZE);
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

    /// Measurement aid: one flat `a * b` over 16M f32 (64 MiB per tensor) with
    /// the pool warm versus emptied before every call, interleaved, medians of 9
    /// passes.  This is what decides the retention caps for large size classes:
    /// the difference is the per-call first-touch (mmap + kernel zeroing + page
    /// faults) that a 64 MiB result pays when the pool cannot serve it.
    #[test]
    #[ignore = "measurement aid: prints timings"]
    fn bench_pool_large() {
        use std::time::Instant;
        let count = 16usize << 20;
        let left = vec![1.5f32; count];
        let right = vec![2.5f32; count];
        let mut warm = Vec::new();
        let mut cold = Vec::new();
        for pass in 0..13 {
            if pass >= 2 {
                let started = Instant::now();
                let out = map2(DType::Float32, &[count], &left, &right, |a, b| a * b).unwrap();
                warm.push(started.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(&out);
                drop(out);
            }
            // The emptied pool forces the allocating path (and its first touch).
            reset_pool();
            let started = Instant::now();
            let out = map2(DType::Float32, &[count], &left, &right, |a, b| a * b).unwrap();
            cold.push(started.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&out);
            drop(out);
        }
        warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cold.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "map2 f32 16M: warm-pool median {:7.3} ms, pool-emptied median {:7.3} ms",
            warm[warm.len() / 2],
            cold[cold.len() / 2]
        );
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
            (pool.entries.len(), pool.small_bytes + pool.large_bytes)
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
            assert!(pool.small_bytes <= POOL_MAX_BYTES);
        });
    }
}
