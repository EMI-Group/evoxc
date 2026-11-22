//! Unit tests for the strided-run driver (`super`): the run decomposition, the
//! two loop shapes, the ternary arms and the materializing fill, each pinned
//! against the general `walk` driver it replaces.
use super::*;
use crate::buffer::{
    apply2 as walk2, apply3 as walk3, broadcast_shape, broadcast_to, Buffer, Storage,
};
use evoxc_core::DType;

/// A deterministic value generator (no `rand` dependency).
fn values(count: usize, seed: i32) -> Vec<i32> {
    let mut state = seed as u64 | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as i32) % 200 - 100
        })
        .collect()
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The binary contract: the run driver and the general `walk` driver agree
/// element for element, and both see the shapes they accept.
fn assert_apply2_agrees(out_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) {
    let count = product(out_shape);
    let a = values(product(a_shape), 7);
    let b = values(product(b_shape), 11);
    let f = |x: i32, y: i32| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(7));
    let mut runs = vec![0i32; count];
    apply2(&mut runs, out_shape, &a, a_shape, &b, b_shape, f);
    let mut walked = vec![0i32; count];
    walk2(&mut walked, out_shape, &a, a_shape, &b, b_shape, f);
    assert_eq!(
        runs, walked,
        "out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: the run driver must match `walk`"
    );
}

#[test]
fn the_run_driver_agrees_with_walk_on_every_layout() {
    // The task's shapes first, then every pattern combination the
    // classification can produce.
    let cases: &[(&[usize], &[usize], &[usize])] = &[
        // a: row scalar (streamed, constant) with one block per outer step.
        (&[500, 100], &[500, 100], &[500, 1]),
        // b: trailing bias (streamed over the whole run, constant).
        (&[500, 50, 4], &[500, 50, 4], &[4]),
        // c: per-row bias (streamed, streamed with a short block).
        (&[500, 50, 4], &[500, 50, 4], &[500, 1, 4]),
        // d: both operands broadcast on different axes (the result is the
        // wider one: `a` is constant along the middle axis, `b` streams it).
        (&[500, 4, 50], &[500, 1, 50], &[500, 4, 50]),
        // A rank-1 operand repeating inside the run (period < span).
        (&[2, 3, 4], &[3, 4], &[2, 1, 1]),
        // Mixed repeat + streaming.
        (&[2, 3, 4], &[2, 1, 4], &[2, 3, 1]),
        // The classic outer product shape.
        (&[3, 4], &[3, 1], &[1, 4]),
        // Flat, unit axes and rank-0 operands.
        (&[7, 5], &[7, 5], &[7, 5]),
        (&[5, 1, 3], &[5, 1, 3], &[1]),
        (&[6], &[6], &[]),
        (&[6], &[], &[6]),
        (&[], &[], &[]),
        (&[4], &[1], &[1]),
        // Unit axes inside the run.
        (&[2, 1, 3], &[2, 1, 3], &[1, 1, 3]),
        (&[2, 1, 3], &[2, 1, 1], &[1, 1, 3]),
    ];
    for (out_shape, a_shape, b_shape) in cases {
        assert_apply2_agrees(out_shape, a_shape, b_shape);
    }
}

#[test]
fn every_combination_of_the_two_patterns_is_exercised() {
    // (streamed, streamed) — both contiguous over the run.
    let plan = Plan::build(&[2, 3], &[&[2, 3], &[2, 3]], &[6, 6]).unwrap();
    assert_eq!(plan.patterns[0], Pattern::Streamed { period: 6 });
    assert_eq!(plan.patterns[1], Pattern::Streamed { period: 6 });
    assert_eq!((plan.block, plan.span), (6, 6));
    // (streamed, constant) — a row scalar against full rows.
    let plan = Plan::build(&[2, 3], &[&[2, 3], &[2, 1]], &[6, 2]).unwrap();
    assert_eq!(plan.patterns[0], Pattern::Streamed { period: 3 });
    assert_eq!(plan.patterns[1], Pattern::Constant);
    // A repeat inside the run: a trailing bias is periodic with the
    // width, so the block is the width and the run repeats it.
    let plan = Plan::build(&[2, 3, 4], &[&[3, 4], &[4]], &[12, 4]).unwrap();
    assert_eq!(plan.patterns[0], Pattern::Streamed { period: 12 });
    assert_eq!(plan.patterns[1], Pattern::Streamed { period: 4 });
    assert_eq!((plan.block, plan.span), (4, 24));
    // (constant, constant) — nothing can be narrowed further.
    let plan = Plan::build(&[2, 3], &[&[2, 1], &[1, 3]], &[2, 3]).unwrap();
    assert_eq!(plan.patterns[0], Pattern::Constant);
    // The left operand is constant here; the right one streams its 3 rows.
    assert_eq!(plan.patterns[1], Pattern::Streamed { period: 3 });
}

#[test]
fn one_contiguous_pass_per_outer_step_and_repeated_blocks_both_agree() {
    // The two loop shapes the driver picks between, and the shapes that put
    // it on each: `(n, m) + (n, 1)` is one `m`-element pass per row
    // (`block == span`, `run2_once`), a trailing bias repeats a 4-element
    // block through one long run (`block < span`, `run2_repeats`).
    let plan = Plan::build(&[500, 100], &[&[500, 100], &[500, 1]], &[50000, 500]).unwrap();
    assert_eq!((plan.block, plan.span), (100, 100));
    let plan = Plan::build(&[500, 50, 4], &[&[500, 50, 4], &[4]], &[100000, 4]).unwrap();
    assert_eq!((plan.block, plan.span), (4, 100000));
    let plan =
        Plan::build(&[500, 4, 50], &[&[500, 1, 50], &[500, 4, 50]], &[25000, 100000]).unwrap();
    assert_eq!((plan.block, plan.span), (50, 200));
    for (out_shape, a_shape, b_shape) in [
        (&[500usize, 100][..], &[500usize, 100][..], &[500usize, 1][..]),
        (&[500, 50, 4][..], &[500, 50, 4][..], &[4][..]),
        (&[500, 50, 4][..], &[500, 50, 4][..], &[500, 1, 4][..]),
        (&[500, 4, 50][..], &[500, 1, 50][..], &[500, 4, 50][..]),
    ] {
        assert_apply2_agrees(out_shape, a_shape, b_shape);
    }
}

#[test]
fn element_i_reads_element_i_of_each_broadcast_operand() {
    // Distinct ranges make any reordering visible: `a` counts rows, `b`
    // counts columns, so a transposed or shifted read cannot pass.
    let out_shape = [2usize, 3];
    let a = [1i32, 2];
    let b = [10i32, 20, 30];
    let mut out = [0i32; 6];
    apply2(&mut out, &out_shape, &a, &[2, 1], &b, &[3], |x, y| x * 100 + y);
    assert_eq!(out, [110, 120, 130, 210, 220, 230]);

    // The same for a per-axis vector against a 3-D result: a `(2, 1)`
    // operand is right-aligned, so it rides out's MIDDLE axis and the
    // result repeats once the middle index rolls over.
    let out_shape = [2usize, 2, 2];
    let a = [1i32, 2];
    let mut out = [0i32; 8];
    apply2(&mut out, &out_shape, &a, &[2, 1], &b[..2], &[2], |x, y| x * 100 + y);
    assert_eq!(out, [110, 120, 210, 220, 110, 120, 210, 220]);
}

#[test]
fn a_rank_zero_operand_and_a_rank_zero_result_take_the_fallback() {
    // Everything is one element and nothing is accepted into the run, so
    // both shapes go through `walk` — with the same result.
    assert_apply2_agrees(&[3, 2], &[1, 1], &[1, 1]);
    assert_apply2_agrees(&[], &[], &[]);
    let mut out = [0i32; 1];
    apply2(&mut out, &[], &[4], &[], &[5], &[], |x, y| x * y);
    assert_eq!(out, [20]);
}

#[test]
fn an_empty_result_writes_nothing() {
    // A zero-length axis needs no work at all (and indexes no operand).
    let mut out: Vec<i32> = Vec::new();
    apply2(&mut out, &[2, 0, 3], &[2, 0, 3], &[2, 0, 3], &[3], &[3], |x, y| x + y);
    assert!(out.is_empty());
}

#[test]
fn nan_signed_zeros_and_wrapping_ride_through_untouched() {
    // The closure owns the semantics; the driver may not reorder, drop or
    // canonicalize a single element. `np.maximum` propagates NaN and keeps
    // the sign of a zero, and i32 adds wrap.
    let maximum = |x: f32, y: f32| {
        if x.is_nan() || y.is_nan() {
            f32::NAN
        } else if x >= y {
            x
        } else {
            y
        }
    };
    // A row scalar (streamed, constant) with a NaN in the scalar and a NaN
    // plus signed zeros in the rows.
    let rows = [f32::NAN, -0.0, 1.0, 0.0, -1.0, 2.0];
    let mut out = [0.0f32; 6];
    apply2(&mut out, &[2, 3], &rows, &[2, 3], &[f32::NAN], &[], maximum);
    assert!(out.iter().all(|value| value.is_nan()));
    let mut out = [0.0f32; 6];
    apply2(&mut out, &[2, 3], &rows, &[2, 3], &[-0.0f32], &[], maximum);
    assert!(out[0].is_nan());
    assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
    // i32 wraparound on both the streaming and the repeat arm.
    let mut out = [0i32; 4];
    apply2(&mut out, &[2, 2], &[i32::MAX, 1], &[2, 1], &[1, 2], &[2], |x, y| {
        x.wrapping_add(y)
    });
    assert_eq!(out, [i32::MIN, i32::MIN + 1, 2, 3]);
}

/// The ternary contract: the run driver and `apply3` agree elementwise.
fn assert_apply3_agrees(
    out_shape: &[usize],
    p_shape: &[usize],
    a_shape: &[usize],
    b_shape: &[usize],
) {
    let count = product(out_shape);
    let p: Vec<bool> = values(product(p_shape), 3).iter().map(|v| *v > 0).collect();
    let a = values(product(a_shape), 5);
    let b = values(product(b_shape), 9);
    let f = |pick: bool, x: i32, y: i32| if pick { x } else { y };
    let mut runs = vec![0i32; count];
    apply3(
        &mut runs, out_shape, &p, p_shape, &a, a_shape, &b, b_shape, f,
    );
    let mut walked = vec![0i32; count];
    walk3(
        &mut walked, out_shape, &p, p_shape, &a, a_shape, &b, b_shape, f,
    );
    assert_eq!(
        runs, walked,
        "out {out_shape:?}, pred {p_shape:?}, a {a_shape:?}, b {b_shape:?}"
    );
}

#[test]
fn the_ternary_driver_agrees_with_walk_on_every_layout() {
    // The select shapes a real graph produces, then each of the eight
    // pattern combinations.
    // (out, predicate, on-true, on-false)
    type TernaryCase<'a> = (&'a [usize], &'a [usize], &'a [usize], &'a [usize]);
    let cases: &[TernaryCase<'_>] = &[
        (&[500, 50, 4], &[500, 1, 1], &[500, 50, 4], &[500, 50, 4]),
        (&[500, 50, 4], &[500, 50, 4], &[500, 50, 4], &[500, 50, 4]),
        (&[2, 3], &[2, 1], &[2, 3], &[2, 3]),
        (&[2, 3], &[2, 1], &[2, 1], &[2, 3]),
        (&[2, 3], &[2, 1], &[2, 3], &[1, 3]),
        (&[2, 3], &[1, 3], &[2, 1], &[2, 3]),
        (&[2, 3], &[2, 3], &[2, 1], &[1, 3]),
        (&[2, 3, 4], &[2, 1, 4], &[2, 3, 4], &[2, 3, 4]),
        (&[2, 3, 4], &[3, 1], &[2, 3, 4], &[2, 1, 4]),
        (&[2, 3], &[1, 1], &[2, 1], &[2, 3]),
        (&[4], &[], &[4], &[4]),
        (&[4], &[4], &[1], &[4]),
    ];
    for (out_shape, p_shape, a_shape, b_shape) in cases {
        assert_apply3_agrees(out_shape, p_shape, a_shape, b_shape);
    }
}

#[test]
fn the_materializing_fill_agrees_with_broadcast_to() {
    // `fill_broadcast` is the `broadcast` op's copy: every shape the op
    // accepts must materialize bit-identically to `broadcast_to`.
    let cases: &[(&[usize], &[usize])] = &[
        (&[500, 1, 50], &[500, 4, 50]),
        (&[4], &[500, 50, 4]),
        (&[500, 1], &[500, 100]),
        (&[2, 1, 4], &[2, 3, 4]),
        (&[3, 4], &[2, 3, 4]),
        (&[], &[2, 3]),
        (&[2, 3], &[2, 3]),
        (&[1, 1, 1], &[5, 1, 4]),
    ];
    for (shape, out_shape) in cases {
        let source = values(product(shape), 13);
        let buffer = Buffer::new(
            DType::Int32,
            shape.to_vec(),
            Storage::I32(source.clone()),
        )
        .unwrap();
        let expected = broadcast_to(&buffer, out_shape).unwrap();
        let filled = fill_broadcast(&source, shape, out_shape);
        assert_eq!(
            Storage::I32(filled),
            expected.storage,
            "shape {shape:?} -> {out_shape:?}"
        );
        // And the shape really is the broadcast one, so nothing silently
        // stops early.
        assert_eq!(product(out_shape), expected.storage.len());
    }
}

#[test]
fn an_empty_target_materializes_nothing() {
    let filled: Vec<i32> = fill_broadcast(&[1, 2], &[2], &[0, 2]);
    assert!(filled.is_empty());
}

#[test]
fn a_block_length_never_splits_a_period() {
    // `block` divides every streamed operand's period, which is what keeps
    // the inner loop contiguous — the property the whole driver rests on.
    for (out_shape, a_shape, b_shape) in [
        (&[2usize, 3, 4][..], &[3usize, 4][..], &[2usize, 1, 1][..]),
        (&[2, 3, 4], &[2, 1, 4], &[2, 3, 1]),
        (&[4, 3], &[4, 3], &[1, 3]),
        (&[3, 5, 2], &[3, 1, 2], &[5, 2]),
    ] {
        assert_eq!(
            broadcast_shape(&[a_shape, b_shape]).unwrap(),
            out_shape.to_vec()
        );
        if let Some(plan) = Plan::build(out_shape, &[a_shape, b_shape], &[
            product(a_shape),
            product(b_shape),
        ]) {
            assert_eq!(plan.span % plan.block, 0);
            for pattern in plan.patterns.iter().take(2) {
                if let Pattern::Streamed { period } = pattern {
                    assert_eq!(period % plan.block, 0, "{out_shape:?} {pattern:?}");
                }
            }
        }
        assert_apply2_agrees(out_shape, a_shape, b_shape);
    }
}

#[test]
fn a_row_predicate_select_takes_one_pass_per_outer_step() {
    // `select` with a predicate broadcast on the leading axes (`(n, 1, 1)`
    // against `(n, m, k)`) has every streamed period equal to the run, so the
    // ternary driver takes its `block == span` single-pass arm; a predicate
    // that repeats inside the run (`(n, 1)`) keeps the repeat arm. Both arms
    // must agree with `walk`.
    let plan = Plan::build(
        &[500, 50, 4],
        &[&[500, 1, 1], &[500, 50, 4], &[500, 50, 4]],
        &[500, 100000, 100000],
    )
    .unwrap();
    assert_eq!(plan.patterns[0], Pattern::Constant);
    assert_eq!((plan.block, plan.span), (200, 200));
    let plan = Plan::build(
        &[500, 50],
        &[&[500, 1], &[500, 50], &[500, 50]],
        &[500, 25000, 25000],
    )
    .unwrap();
    assert_eq!(plan.patterns[0], Pattern::Constant);
    assert_eq!((plan.block, plan.span), (50, 50));
    // A predicate streamed over the run's own trailing axis: one pass as well.
    let plan = Plan::build(
        &[500, 50],
        &[&[500, 50], &[500, 50], &[500, 50]],
        &[25000, 25000, 25000],
    )
    .unwrap();
    assert_eq!(plan.patterns[0], Pattern::Streamed { period: 25000 });
    assert_eq!((plan.block, plan.span), (25000, 25000));
    for (out_shape, p_shape, a_shape, b_shape) in [
        (&[500usize, 50, 4][..], &[500usize, 1, 1][..], &[500, 50, 4][..], &[500, 50, 4][..]),
        (&[500, 50][..], &[500, 1][..], &[500, 50][..], &[500, 50][..]),
        (&[500, 50][..], &[500, 50][..], &[500, 50][..], &[500, 50][..]),
        (&[2, 3][..], &[1, 3][..], &[2, 1][..], &[2, 3][..]),
        (&[2, 3][..], &[2, 1][..], &[2, 3][..], &[1, 1][..]),
        (&[2, 3][..], &[2, 1][..], &[1, 3][..], &[2, 1][..]),
    ] {
        assert_apply3_agrees(out_shape, p_shape, a_shape, b_shape);
    }
}

#[test]
fn a_short_operand_falls_back_instead_of_reading_past_the_slice() {
    // The general driver is what reports a mismatch; the run plan refuses
    // to build, so the run driver never guesses an element count.
    let out_shape = [2usize, 3];
    assert!(Plan::build(&out_shape, &[&[2, 3], &[3]], &[5, 3]).is_none());
    assert!(Plan::build(&out_shape, &[&[2, 3], &[3]], &[6, 2]).is_none());
    assert!(Plan::build(&out_shape, &[&[2, 3], &[3]], &[6, 3]).is_some());
}

#[test]
fn the_periodic_block_loop_covers_every_specialised_width() {
    // A period inside the const-generic set takes the unrolled tile loop, and
    // the run length leaves a tail that does not divide by `4 * period`, so
    // both the four-tile body and its scalar tail must agree with `walk`.
    for period in [2usize, 3, 4, 5, 6, 8] {
        let out_shape = [2, 3, period];
        let a_shape = [2, 3, period];
        let b_shape = [period];
        let plan = Plan::build(&out_shape, &[&a_shape, &b_shape], &[6 * period, period]).unwrap();
        assert_eq!(plan.block, period);
        assert_eq!(plan.span, 6 * period);
        assert_eq!(plan.patterns[1], Pattern::Streamed { period });
        assert_apply2_agrees(&out_shape, &a_shape, &b_shape);
        // The same period with the tile on the LEFT and a wider run: the tail
        // is `6p - 4p = 2p` here, the widest tail the unroll can leave.
        assert_apply2_agrees(
            &[2, 3, period],
            &[period],
            &[2, 3, period],
        );
    }
}
#[test]
fn a_short_period_reads_element_i_modulo_the_period_across_the_whole_run() {
    // Distinct ranges make any reordering or a stale tile visible: the left
    // operand counts up, the tile counts by a different stride, so a run that
    // restarted its tile or reused the previous step's would not match.
    let out_shape = [3usize, 4];
    let left: Vec<i32> = (0..12).collect();
    let tile = [7i32, 1000, -3, 55];
    let mut out = [0i32; 12];
    // The right operand is right-aligned, so a `(4,)` tile rides the trailing
    // axis and repeats three times inside the run of twelve.
    let plan = Plan::build(&out_shape, &[&out_shape, &[4]], &[12, 4]).unwrap();
    assert_eq!((plan.block, plan.span), (4, 12));
    apply2(&mut out, &out_shape, &left, &out_shape, &tile, &[4], |x, y| {
        x * 10 + y
    });
    let expected: Vec<i32> = (0..12).map(|i| i as i32 * 10 + tile[i % 4]).collect();
    assert_eq!(out.to_vec(), expected);
    // A one-element run and a period equal to the run are the two edges of the
    // tile path (the latter falls back to the contiguous one).
    assert_apply2_agrees(&[3], &[3], &[3]);
    assert_apply2_agrees(&[5], &[1], &[5]);
    assert_apply2_agrees(&[7, 6], &[7, 6], &[6]);
}
#[test]
fn a_short_period_fill_matches_broadcast_to() {
    // The exponential-doubling fill covers spans that are not a power-of-two
    // multiple of the period and periods that do not divide the source length.
    let cases: &[(&[usize], &[usize])] = &[
        (&[5], &[7, 5]),
        (&[3], &[499, 3]),
        (&[1], &[9, 4, 1]),
        (&[6], &[2, 3, 6]),
    ];
    for (shape, out_shape) in cases {
        let source = values(product(shape), 17);
        let buffer =
            Buffer::new(DType::Int32, shape.to_vec(), Storage::I32(source.clone())).unwrap();
        let expected = broadcast_to(&buffer, out_shape).unwrap();
        let filled = fill_broadcast(&source, shape, out_shape);
        assert_eq!(Storage::I32(filled), expected.storage, "{shape:?} -> {out_shape:?}");
    }
}
#[test]
fn a_long_run_over_many_outer_steps_agrees_with_walk() {
    // A run wide enough to matter (2^18 elements, the flat driver's split
    // threshold) with one short block per outer step, both with the broadcast
    // operand on the trailing axis and as a per-row scalar.
    assert_apply2_agrees(&[4096, 64], &[4096, 64], &[4096, 1]);
    assert_apply2_agrees(&[1000, 1000], &[1000, 1000], &[1000]);
    assert_apply3_agrees(
        &[1024, 256],
        &[1024, 1],
        &[1024, 256],
        &[256],
    );
}
