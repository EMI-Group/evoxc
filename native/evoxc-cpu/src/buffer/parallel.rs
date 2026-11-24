//! The flat-map thread split: the measured thresholds, `parallel_threads`,
//! `split_parallel` and its operand `window` helper.

// ---------------------------------------------------------------------------
// Broadcast-free (flat) drivers
// ---------------------------------------------------------------------------
//
// The `walk`/`applyN` machines above serve every strided or broadcast kernel:
// they recompute one offset per operand per element.  A kernel whose operands
// all already have the output shape needs none of that -- element `i` of every
// operand is element `i` of the output, so the kernel is one flat `zip` over
// plain slices.  These drivers are monomorphized over the element types AND the
// closure, which is what lets LLVM turn the loop into vector moves; keep them
// that way (no `&dyn Fn`, no per-element `match` on an op code -- dispatch the
// op code once in the caller and let each arm's closure fold to one operation).
//
// Callers gate on [`Buffer::is_flat_for`], so every slice has the output's
// length; the `zip` therefore visits every element.  A slice that is somehow
// shorter would be a broken internal buffer, not user input.
//
// These maps are DRAM-bandwidth bound at ~0.21 ms per 1e6 f32 elements on one
// thread (numpy's ufuncs are single-threaded, so that alone only reaches
// parity), which is why a big enough result is split across scoped threads.
// Scoped-thread setup costs ~10 us per extra thread, so the split only pays for
// a genuinely large result.  Measured on this host (`bench_flat_map_large`,
// 64 MiB results, interleaved medians of 9, spawn+join in parentheses) --
//
//   1t 9.34 ms   2t 5.90 (0.11)   4t 4.85 (0.17)   6t 4.71 (0.20)
//   8t 4.57 (0.20)   12t 4.67 (0.27)   16t 4.77 (0.31)
//
// -- so at 64 MiB eight threads still lead (4.57 ms) and twelve/sixteen turn
// over (4.67 / 4.77, the hybrid P/E cores'), while at 16 MiB results four and
// eight tie (1.03 / 1.06 ms) and at 4 MiB they read 0.28 / 0.26 ms: the wider
// cap only belongs to genuinely large maps.  The lower threshold comes from
// `bench_thread_crossover` (one flat `add`, 1 thread / 4 threads, best of 40) --
//
//   2^16 elems 0.007 / 0.036 ms   2^18 elems 0.030 / 0.043 ms
//   2^17 elems 0.014 / 0.038 ms   2^19 elems 0.100 / 0.049 ms
//   1e6 elems  0.249 / 0.097 ms
//
// -- so the crossover sits between 2^18 and 2^19 elements: at 2^18 FOUR threads
// are 43 % slower than the calling thread, while the win grows to 2x at 2^19.
// The split therefore starts at 2^18 but hands one thread only 2^17 elements
// (two threads at 2^18, four at 2^19+), never the full four at the crossover;
// the `(4096, 64)` f32 tensors of a merged-GP-shaped graph are exactly 2^18
// elements and take that two-thread split.  From `PARALLEL_LARGE_ELEMS` (4M
// elements, 16 MiB of f32) the cap widens to eight, where the large-map sweep
// still wins.  Small maps stay on the calling thread, where spawning would
// dominate.
/// Element count from which a flat map is worth splitting across threads.
pub(super) const PARALLEL_MIN_ELEMS: usize = 1 << 18;
/// Elements one thread should own before it earns its setup cost: a split adds
/// `count / PARALLEL_MIN_PER_THREAD` threads (capped by the core count), so a
/// 2^18-element map (a `(4096, 64)` f32 tensor) runs on two threads -- measured
/// faster than the calling thread alone, unlike four -- while a >= 2^19-element
/// one gets the full four.
pub(super) const PARALLEL_MIN_PER_THREAD: usize = 1 << 17;
/// Upper bound on the threads a flat map below [`PARALLEL_LARGE_ELEMS`] uses
/// (the measured optimum for small and medium results).
pub(super) const PARALLEL_MAX_THREADS: usize = 4;
/// Element count at or above which a flat map may use the wider large-map cap:
/// 1 << 22 = 4M elements = 16 MiB of f32, the size from which eight threads
/// measure ahead of four.
pub(super) const PARALLEL_LARGE_ELEMS: usize = 1 << 22;
/// Upper bound on the threads a large flat map (>= [`PARALLEL_LARGE_ELEMS`])
/// uses: eight measure 4.57 ms on a 64 MiB result against four's 4.85, while
/// twelve (4.67) and sixteen (4.77) turn over.
pub(super) const PARALLEL_MAX_THREADS_LARGE: usize = 8;

/// How many threads a flat map over `count` elements should split across.
pub(crate) fn parallel_threads(count: usize) -> usize {
    if count < PARALLEL_MIN_ELEMS {
        return 1;
    }
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cores = *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1)
            .clamp(1, PARALLEL_MAX_THREADS_LARGE)
    });
    let cap = if count >= PARALLEL_LARGE_ELEMS {
        PARALLEL_MAX_THREADS_LARGE
    } else {
        PARALLEL_MAX_THREADS
    };
    cap.min(cores).min((count / PARALLEL_MIN_PER_THREAD).max(1))
}

/// Run `body(offset, chunk)` over disjoint chunks of `out` on up to `threads`
/// scoped threads.  Every chunk is visited exactly once, so a body that writes
/// all of its chunk writes all of `out`; `offset` is the chunk's start index,
/// which is how a caller slices its operands to match.
pub(crate) fn split_parallel<R: Copy + Send, F: Fn(usize, &mut [R]) + Sync>(
    out: &mut [R],
    threads: usize,
    body: F,
) {
    if threads <= 1 || out.len() < threads {
        body(0, out);
        return;
    }
    let chunk = out.len().div_ceil(threads);
    let mut chunks = out.chunks_mut(chunk);
    std::thread::scope(|scope| {
        // The leading chunk stays on the calling thread.  Every spawned chunk
        // starts at the number of elements the previous ones cover, which is
        // accumulated here rather than derived from the chunk length: the last
        // chunk of a size that does not divide evenly is shorter, so deriving
        // the offset from `chunk.len()` shifted every operand window of the last
        // chunk by the difference (a wrong result, not just a slow one).
        let Some(first) = chunks.next() else { return };
        let mut offset = first.len();
        for chunk in chunks {
            let start = offset;
            offset += chunk.len();
            let body = &body;
            scope.spawn(move || body(start, chunk));
        }
        body(0, first);
    });
}

/// The operand window of chunk `[offset, offset + len)` (empty past the end,
/// preserving `zip`'s stop-at-the-shortest behaviour).
#[inline]
pub(super) fn window<T>(values: &[T], offset: usize, len: usize) -> &[T] {
    values.get(offset..offset + len).unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_thread_split_scales_with_the_work_per_thread() {
        // The measured policy (`bench_thread_crossover` + `bench_flat_map_large`):
        // below 2^18 elements one thread, two at 2^18 (a `(4096, 64)` f32
        // tensor), the full four from 2^19, and the wider eight from
        // `PARALLEL_LARGE_ELEMS` (4M elements, 16 MiB of f32).  This is a
        // performance rule, not a correctness one, but a silent change to it
        // would undo the measurement.
        assert_eq!(parallel_threads(0), 1);
        assert_eq!(parallel_threads(PARALLEL_MIN_ELEMS - 1), 1);
        let cores = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1)
            .clamp(1, PARALLEL_MAX_THREADS_LARGE);
        assert_eq!(parallel_threads(PARALLEL_MIN_ELEMS), cores.min(2));
        assert_eq!(parallel_threads(1 << 19), cores.min(4));
        assert_eq!(parallel_threads(1 << 21), cores.min(4));
        assert!(parallel_threads(1 << 21) <= PARALLEL_MAX_THREADS);
        assert_eq!(parallel_threads(PARALLEL_LARGE_ELEMS), cores.min(8));
        assert_eq!(parallel_threads(1 << 24), cores.min(8));
        assert!(parallel_threads(1 << 24) <= PARALLEL_MAX_THREADS_LARGE);
        // Monotone non-decreasing in `count`, never above the machine's cores,
        // never above the tier's cap.
        let mut previous = 1;
        for shift in 0..=26 {
            let count = 1usize << shift;
            let threads = parallel_threads(count);
            assert!(threads >= previous, "the split never shrinks as the map grows");
            assert!(threads <= cores, "never more threads than cores");
            let cap = if count >= PARALLEL_LARGE_ELEMS {
                PARALLEL_MAX_THREADS_LARGE
            } else {
                PARALLEL_MAX_THREADS
            };
            assert!(threads <= cap);
            previous = threads;
        }
        assert_eq!(parallel_threads(1_000_000), parallel_threads(1 << 20));
    }

    #[test]
    fn the_thread_split_hands_every_chunk_its_running_offset() {
        // A count the split does not divide evenly makes the LAST chunk shorter,
        // so a chunk's start is the running total of the chunks before it and not
        // a multiple of the chunk length.  Deriving it from the chunk length made
        // every operand window of the trailing chunk read shifted elements — a
        // wrong result, not a slow one — which is why this is pinned: the chunks
        // must partition `out` in order, and each element must be written exactly
        // once by the chunk that owns it.
        for &len in &[(1 << 19) + 5, (1 << 18) + 3, 4099, 1000, 7, 3] {
            let mut out: Vec<u64> = vec![0; len];
            let chunks = std::sync::Mutex::new(Vec::new());
            split_parallel(&mut out, PARALLEL_MAX_THREADS, |offset, chunk| {
                chunks.lock().expect("chunk list").push((offset, chunk.len()));
                for (index, slot) in chunk.iter_mut().enumerate() {
                    *slot = (offset + index) as u64 + 1;
                }
            });
            let mut chunks = chunks.into_inner().expect("chunk list");
            chunks.sort_unstable();
            // The partition `chunks_mut` would walk: a full `chunk` per thread
            // except for the short tail.
            let chunk = if len >= PARALLEL_MAX_THREADS {
                len.div_ceil(PARALLEL_MAX_THREADS)
            } else {
                len
            };
            let mut expected = Vec::new();
            let mut start = 0;
            while start < len {
                let end = (start + chunk).min(len);
                expected.push((start, end - start));
                start = end;
            }
            assert_eq!(chunks, expected, "len {len}: the chunks partition `out` in order");
            let want: Vec<u64> = (1..=len as u64).collect();
            assert_eq!(out, want, "len {len}: every element exactly once, own offset");
        }
        // Four chunks with a short last one: the shape the old arithmetic broke
        // (an offset derived from the chunk length lands 3 elements late).
        let len: usize = (1 << 19) + 5;
        let chunk = len.div_ceil(PARALLEL_MAX_THREADS);
        assert_eq!(
            len - (PARALLEL_MAX_THREADS - 1) * chunk,
            chunk - 3,
            "the last chunk is short, so the offsets are not multiples of `chunk`"
        );
    }

    /// Measurement aid, not a correctness test: `cargo test --release -p
    /// evoxc-cpu bench_flat_map_large -- --ignored --nocapture` measures a flat
    /// `add` over LARGE (DRAM-resident) results at 1/2/4/8/16 threads and the
    /// bare spawn+join cost of a `split_parallel` call, which is what the
    /// large-map thread policy is chosen from.  Thread counts are interleaved
    /// inside one rep loop (this host is shared), and every cell reports the
    /// MEDIAN of [`BENCH_REPS`] timed passes after [`BENCH_WARMUPS`] warmups.
    #[test]
    #[ignore = "measurement aid: prints timings"]
    fn bench_flat_map_large() {
        use std::time::Instant;
        const BENCH_WARMUPS: usize = 3;
        const BENCH_REPS: usize = 9;
        let thread_counts = [1usize, 2, 3, 4, 6, 8, 12, 16];
        println!("cpus {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
        for count in [1usize << 18, 1 << 20, 1 << 22, 4 << 20, 16 << 20] {
            let left: Vec<f32> = vec![1.5; count];
            let right: Vec<f32> = vec![2.5; count];
            let mut out = vec![0.0f32; count];
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); thread_counts.len()];
            let mut spawns: Vec<Vec<f64>> = vec![Vec::new(); thread_counts.len()];
            for _ in 0..BENCH_WARMUPS {
                for threads in thread_counts {
                    split_parallel(&mut out, threads, |offset, chunk| {
                        let len = chunk.len();
                        for ((slot, x), y) in chunk
                            .iter_mut()
                            .zip(window(&left, offset, len))
                            .zip(window(&right, offset, len))
                        {
                            *slot = *x + *y;
                        }
                    });
                }
            }
            for _ in 0..BENCH_REPS {
                for (index, threads) in thread_counts.iter().copied().enumerate() {
                    let started = Instant::now();
                    split_parallel(&mut out, threads, |offset, chunk| {
                        let len = chunk.len();
                        for ((slot, x), y) in chunk
                            .iter_mut()
                            .zip(window(&left, offset, len))
                            .zip(window(&right, offset, len))
                        {
                            *slot = *x + *y;
                        }
                    });
                    samples[index].push(started.elapsed().as_secs_f64() * 1e3);
                    let started = Instant::now();
                    split_parallel(&mut out, threads, |_offset, _chunk| {});
                    spawns[index].push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
            let mut line = format!("count {count:>9} ({:>3} MiB):", (count * 4) >> 20);
            for (index, threads) in thread_counts.iter().enumerate() {
                samples[index].sort_by(|a, b| a.partial_cmp(b).unwrap());
                spawns[index].sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = samples[index][samples[index].len() / 2];
                let spawn = spawns[index][spawns[index].len() / 2];
                line.push_str(&format!(
                    "  {threads:2}t {med:7.3} ms (+{spawn:5.3} spawn)"
                ));
            }
            println!("{line}");
        }
    }

    /// Measurement aid: the cost of one flat `add` **including a fresh 64 MiB
    /// result allocation**, i.e. what a map that the storage pool cannot serve
    /// pays on top of the copy (`vec![0.0; n]` first touch = mmap + kernel
    /// zeroing + every page faulted in on the write).
    #[test]
    #[ignore = "measurement aid: prints timings"]
    fn bench_allocation_first_touch() {
        use std::time::Instant;
        let count = 16usize << 20;
        for label in ["vec![0.0; 16M]", "with_capacity + fill 16M"] {
            let mut samples = Vec::new();
            for _ in 0..7 {
                let started = Instant::now();
                let mut values: Vec<f32> = match label {
                    "vec![0.0; 16M]" => vec![0.0; count],
                    _ => {
                        let mut values = Vec::with_capacity(count);
                        values.resize(count, 0.0);
                        values
                    }
                };
                for (index, slot) in values.iter_mut().enumerate() {
                    *slot = index as f32;
                }
                std::hint::black_box(&values);
                samples.push(started.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("{label:28} median {:7.3} ms", samples[samples.len() / 2]);
        }
    }

    /// Measurement aid, not a correctness test: `cargo test --release -p
    /// evoxc-cpu bench_thread_crossover -- --ignored --nocapture` prints the
    /// per-element cost of one flat `add` at several sizes and thread counts,
    /// which is what the `PARALLEL_MIN_ELEMS` threshold is chosen from.
    #[test]
    #[ignore = "measurement aid: prints timings"]
    fn bench_thread_crossover() {
        use std::time::Instant;
        let left: Vec<f32> = (0..(1 << 20)).map(|index| index as f32 * 0.5).collect();
        let right: Vec<f32> = (0..(1 << 20)).map(|index| index as f32 * 0.25).collect();
        for count in [1usize << 16, 1 << 17, 262144, 1 << 19, 1_000_000] {
            let mut line = format!("count {count:>8} ({:>4} KiB):", count * 4 / 1024);
            for threads in [1usize, 2, 3, 4] {
                let mut out = vec![0.0f32; count];
                let mut best = f64::INFINITY;
                for _ in 0..40 {
                    let started = Instant::now();
                    split_parallel(&mut out, threads, |offset, chunk| {
                        let len = chunk.len();
                        for ((slot, x), y) in chunk
                            .iter_mut()
                            .zip(window(&left, offset, len))
                            .zip(window(&right, offset, len))
                        {
                            *slot = *x + *y;
                        }
                    });
                    best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
                line.push_str(&format!("  {threads}t {best:7.3} ms"));
            }
            let single = {
                let mut out = vec![0.0f32; count];
                let mut best = f64::INFINITY;
                for _ in 0..40 {
                    let started = Instant::now();
                    for ((slot, x), y) in out.iter_mut().zip(&left).zip(&right) {
                        *slot = *x + *y;
                    }
                    best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            line.push_str(&format!("   plain {single:7.3} ms"));
            println!("{line}");
        }
    }
}
