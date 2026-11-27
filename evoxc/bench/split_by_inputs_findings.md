# `evoxc.split_by_inputs` — does one prepare + N applies beat N unsplit runs?

Decision evidence for the graph-level program-splitting transform: it splits a traced `Graph` into a `prepare` graph over the inputs that are *fixed* across turns and an `apply` graph over the prepared values plus the *varying* inputs.
The question is the one a real application asks: at the reuse count it actually uses, does one `prepare` plus N `apply` runs beat N unsplit runs, and where does the boundary cost hurt?
The probe, the raw records and the reproduction command are in `./probe_split_by_inputs.py` and `./split_by_inputs_results.json`; every number below is a field of that JSON or arithmetic on recorded fields.

## Provenance and protocol

| | |
|---|---|
| Command | `python evoxc/bench/probe_split_by_inputs.py --seed 0 --reps 21 --reuse 1,4,16,64 --sizes 512,4096 --boundary-size 16384 --workloads a,b,c --threads 1 --backends numpy --out evoxc/bench/split_by_inputs_results.json` |
| Session (`date_utc`) | 2026-11-27T18:53:28+00:00 |
| Host | Linux-6.18.63 x86_64, 13th Gen Intel(R) Core(TM) i5-13400F, glibc 2.44 |
| Python / numpy / evoxc | 3.14.10 / 2.5.5 / 0.0.0+unknown (`native_extension: null`) |
| Backend | `numpy` (the default interpreter), host memory on `Device('cpu', 0)` |
| Repetitions | 21 per call shape, `--seed 0` |
| Thread pinning | `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` |
| CPU count | `nproc` 4 vs `os.cpu_count()` 16 (affinity 16) |
| Load average, start → end | `[1.66, 3.15, 2.78]` → `[1.66, 3.15, 2.78]` |
| Result | `status: PASS`, `failures: []` |

**Three accounting tables, and why the third one is the claim.**
This host runs in slow-moving fast/slow modes: the *same* call in one 21-repetition cell samples as low as `1.271 ms` and as high as `35.856 ms` (`A_fixed_precompute|4096`, `run.unsplit`), and `18.755 ms` for a 0.2 ms prepare (`A_fixed_precompute|512`, `run.prepare`).
Two formulations' separate medians can therefore land in different modes and be compared unfairly: in `A_fixed_precompute|512` every `unsplit` sample stays at or below `0.250 ms` while a `prepare` that is a strict subset of it samples `18.755 ms` (`run_samples`), the same effect `_meta.shared_host_caveat` summarizes as `the SAME call samples anywhere from ~1.4 to ~59 ms in one 21-repetition cell`.
So every cell records three tables:

- `amortized` — per-generation ms and ratios from each call shape's **median**.
- `amortized_from_min` — the same from each shape's **min**.
- `amortized_per_repetition_ratio` — **the primary table**: each repetition times every call shape once, so `unsplit / split` computed *inside* one repetition divides two samples taken in the same host mode; the reported ratio is the median of those 21 paired ratios, and `crossover_reuse_from_per_repetition_ratio` is read off it.

The raw per-repetition samples behind all three are recorded too (`run_samples`), so the mode mixture is auditable rather than hidden behind min/median/max.
In the recorded session the median table and the paired-ratio table agree to within 3.4 % in every cell (e.g. `A_fixed_precompute|4096` at reuse 64: `10.69` vs `11.05`), which is the check that the session itself sat in one mode; the absolute milliseconds remain indicative, the ratios are the result.

Every formulation is checked, not assumed: all five cells report `parity_max_abs_err` of `0.0` for `unsplit`, `split` and `hand_written` against a plain-numpy oracle, `source_digest_unchanged_by_split: true`, `dropped_ops: 0`, `unused_varying_leaves: []`, and outputs on `Device(kind='cpu', index=0)`.

## The three workload families

| Cell | Graph | Fixed → varying | Boundary |
|---|---|---|---|
| `A_fixed_precompute` (rows 512, 4096) | normalize `X`, `tanh(Xn @ W1)`, then `mean(tanh_out @ W2)` — 16 ops | `(X, W1)` → `W2` | 1 value, `tanh` of shape `[rows, 64]` f32 = **131072 B** (512 rows) / **1048576 B** (4096 rows) |
| `B_boundary_dominated` (rows 16384) | elementwise scale of a fixed dataset, then a reduce — 4 ops | `(X,)` → `(scale,)` | 1 value, `multiply` of shape `[16384, 64]` f32 = **4194304 B** |
| `C_state_variable` (rows 512, 4096) | loop-free recurrence step `mean((dataset @ W1 + tanh(state @ W1)) * W2)` — 6 ops | `(dataset, W1)` → `(state, W2)` | **2** values: `W1` as a pass-through fixed input (`is_input: true`, 16384 B) plus the `dot` result `[rows, 64]` = 131072 B (512) / 1048576 B (4096) |

Split op counts (`prepare_ops` / `apply_ops` out of `source_ops`): A `12 / 4` of `16`, B `2 / 2` of `4`, C `1 / 5` of `6`.
The hand-written pair matches the split's op counts in every cell (`hand_written.ops_match_split: true`) — the transform derives exactly the pair a user would write by hand.

## Where the reuse crossover sits

`speedup_unsplit_over_split` from `amortized_per_repetition_ratio` (>1 means the split is faster):

| Cell | 1 turn | 4 turns | 16 turns | 64 turns | crossover |
|---|---|---|---|---|---|
| `A_fixed_precompute|512` | 0.73 | 1.10 | 2.17 | 3.98 | 4 |
| `A_fixed_precompute|4096` | 0.85 | 2.25 | 5.95 | 11.05 | 4 |
| `B_boundary_dominated|16384` | 0.85 | 1.16 | 1.32 | 1.37 | 4 |
| `C_state_variable|512` | 0.66 | 0.87 | 1.10 | 1.22 | 16 |
| `C_state_variable|4096` | 0.84 | 1.13 | 1.26 | 1.30 | 4 |

The crossover is **4 turns** in every cell but the smallest state-variable one (`16`), and the analytic estimate (`crossover_reuse_analytic`) matches that exactly only for `A|512` (`4`): it reads `2` for `A|4096`, `B` and `C|4096` and `9` for `C|512`, so it is never later than measured but lands a grid step early in four of the five cells — the analytic form assumes the build totals are free of mode noise, which they are not.
Below the crossover the split *loses* — at one turn by 15–34 % (`0.66`–`0.85`) — because it pays `trace + split_by_inputs + two stagings` (`split_total_ms`) where the unsplit program pays one (`unsplit_total_ms`):

| Cell | `split_transform_ms` | `split_total_ms` | `unsplit_total_ms` | `hand_written_total_ms` |
|---|---|---|---|---|
| `A|512` | 0.263 | 1.150 | 0.797 | 0.933 |
| `A|4096` | 0.263 | 1.118 | 0.768 | 0.904 |
| `B|16384` | 0.138 | 0.504 | 0.293 | 0.406 |
| `C|512` | 0.175 | 0.630 | 0.366 | 0.492 |
| `C|4096` | 0.193 | 0.642 | 0.360 | 0.493 |

The split's own transform cost is 0.138–0.263 ms and the extra staging hop is the rest: on the numpy backend the split costs 1.44–1.78× the unsplit *build* and 1.23–1.30× the hand-written build (`A|4096` 1.118 vs 0.904).
That build premium is what a low reuse count has to amortize, and it is the whole reason the crossover is not 1.

## What the boundary costs per turn

`run` medians, and the per-turn cost of crossing the boundary (`prepare` + `apply` − one `unsplit` run):

| Cell | `unsplit` | `prepare` | `apply` | `prepare+apply` | `boundary_overhead_ms_one_turn` | boundary bytes |
|---|---|---|---|---|---|---|
| `A|512` | 0.212 | 0.202 | 0.036 | 0.238 | 0.026 | 131072 |
| `A|4096` | 1.393 | 1.339 | 0.093 | 1.432 | 0.039 | 1048576 |
| `B|16384` | 0.827 | 0.221 | 0.599 | 0.820 | −0.007 | 4194304 |
| `C|512` | 0.188 | 0.059 | 0.149 | 0.208 | 0.020 | 147456 |
| `C|4096` | 1.356 | 0.350 | 1.034 | 1.384 | 0.028 | 1064960 |

Reading them:

- **A is the shape the transform is for.** `prepare` costs 1.339 ms once and removes it from every turn: `apply` is `0.093 ms`, so each turn saves `1.300 ms` for a boundary of `1048576 B` — the saving is ~33× the measured per-turn crossing overhead (`prepare` + `apply` − `unsplit` = `0.039 ms`; ~14× the per-turn `apply` run of `0.093 ms` on its own), and even at 4 turns the split is `2.25×` ahead.
- **B is the boundary-dominated ceiling.** The fixed part is cheap and the split *moves work into the boundary*: `prepare` writes `4194304 B` (`0.221 ms`) and `apply` still runs the reduce (`0.599 ms`). Per turn only `0.228 ms` is saved, so the ratio saturates at `1.37×` however many turns are run — the split cannot win more than the reduce it left in `apply`, and the 4 MiB the boundary materializes is paid in `prepare`. A boundary this size against this little work is not worth splitting for its own sake.
- **C shows the pass-through case.** The boundary carries **two** values although `prepare` is a single `dot`: `W1` crosses as a fixed *input* (`is_input: true`, 16384 B) because `apply` still needs it. `prepare` removes the state-independent `dataset @ W1` (`0.350 ms` of the `1.356 ms` turn) and the ratio tops out at `1.30×`. At 512 rows that product costs so little (`prepare` 0.059 ms) that 16 turns are needed before the build premium is repaid.

## Is the transform free against the hand-written pair?

Yes — within noise, and the graph is the same (`ops_match_split: true` everywhere).

| Cell | split `prepare` / hand `prepare` | split `apply` / hand `apply` | `speedup_unsplit_over_hand_written` (1/4/16/64, `amortized_per_repetition_ratio`) |
|---|---|---|---|
| `A|512` | 0.202 / 0.198 | 0.036 / 0.032 | 0.87 / 1.31 / 2.54 / 4.53 |
| `A|4096` | 1.339 / 1.327 | 0.093 / 0.087 | 0.93 / 2.45 / 6.35 / 11.54 |
| `B|16384` | 0.221 / 0.211 | 0.599 / 0.594 | 0.93 / 1.20 / 1.33 / 1.38 |
| `C|512` | 0.059 / 0.056 | 0.149 / 0.147 | 0.80 / 0.99 / 1.17 / 1.25 |
| `C|4096` | 0.350 / 0.350 | 1.034 / 1.030 | 0.92 / 1.17 / 1.28 / 1.31 |

The split's `apply` tracks the hand-written one to within 0.4–12.5 % (i.e. ≤0.006 ms absolute, on runs as small as `0.032 ms`), and its per-generation ratio (`speedup_unsplit_over_hand_written` in `amortized_per_repetition_ratio`) differs from the hand-written pair's by at most `0.55` absolute at 64 turns (`3.98` vs `4.53` for `A|512`, `11.05` vs `11.54` for `A|4096`, `1.37` vs `1.38` for `B`, `1.22` vs `1.25` for `C|512` and `1.30` vs `1.31` for `C|4096` — because the split's slightly dearer build is amortized, not its runs).
So: the automatic split costs nothing recurring over writing the pair by hand, and costs `0.138`–`0.263 ms` (`split_transform_ms`) of one-off transform work at trace time.

## The competing strategy: batch (what `vmap` expands to) vs reuse

Workload A evaluates 64 candidate `W2` vectors. Three ways, from `batched_vmap_stand_in.totals_for_candidates` (the batched graph is a hand-written `h @ W2_matrix` graph validated against 64 separate applies; `_meta.vmap` records that `vmap` itself is unavailable in this worktree because the Rust extension is not built and every builtin transform delegates to it with no Python fallback):

| Cell | one batched run (64 candidates) | 64 `apply` runs | 64 unsplit runs | batched vs applies | batched vs 64 unsplit |
|---|---|---|---|---|---|
| `A|512` | 0.997 | 3.656 | 14.365 | 3.67× | 14.41× |
| `A|4096` | 2.466 | 8.409 | 89.920 | 3.41× | 36.46× |

Batching is **3.4–3.7× cheaper than 64 applies and 14–36× cheaper than 64 unsplit runs**, so whenever the candidate axis is static and all candidates are known at once, the batched graph is the strategy — not prepare/apply.
That is a different axis of the same fixed data: `vmap` amortizes *across candidates within one turn*, `split_by_inputs` amortizes *across turns over one candidate at a time*. They compose (a batched `apply`), but the batched comparison above is against 64 separate `apply` runs, which is the case prepare/apply is for.

## When to use which

| Situation | Reach for | Measured basis |
|---|---|---|
| All candidates known at once, static candidate axis | batched graph (`vmap`-expanded) | 3.41–3.67× over 64 applies, 14.41–36.46× over 64 unsplit runs |
| Candidates arrive one at a time (streaming / online loop), many turns over the same fixed data | `split_by_inputs` → prepare once, apply per turn | 3.98–11.05× at 64 turns for a precompute-shaped graph (A) |
| One to three turns over the fixed data | don't split — run unsplit | 0.66–0.85 at 1 turn, crossover at 4 turns (16 for a tiny prepare) |
| The fixed part is cheap and its result is large | don't split — the boundary is the work | B: 4194304 B boundary, 1.37× ceiling at any reuse |
| Split removes a single op and the graph is small | expect little | `C|512`: 1.22× at 64 turns, crossover 16 |
| A fixed matrix is still needed after the boundary | expect it to cross as a pass-through input | C: 2 boundary values, 16384 B of them the fixed `W1` itself |

## What this does NOT claim

- **numpy interpreter only, host memory only.** Every tensor is `Device('cpu', 0)`; nothing here measures a device-resident boundary, where the prepared values would be uploaded once instead of being re-read from host memory, or the explicit `.to(cpu)` a device-side apply would need to hand them on.
- **Single process, shared host.** One timing process at a time, but the machine is shared: the same call spread from `1.271 ms` to `35.856 ms` inside one cell in this session, and an earlier session of the same probe was unusable for absolute values. The ratios are reported three ways for that reason; the absolute milliseconds are indicative, not isolated-hardware measurements.
- **No `native`/`iree`/`xla`/`tvm` leg.** `native_extension: null` in `_meta.native_extension` — the Rust CPU engine and the compiler adapters are not measured here, and the staging premium measured above is the numpy interpreter's serialized-IR path.
- **No memory measurement.** Boundary size is recorded in bytes (`boundary_bytes`, per-value in `boundary`), but nothing here measures peak RSS or allocator behaviour, which is the other half of what a large boundary costs.
- **Three workload shapes.** A, B and C are representative of a fixed precompute, a boundary-dominated split and a state-variable step; they are not a survey, and the crossover reuse counts are properties of these graphs on this host.
