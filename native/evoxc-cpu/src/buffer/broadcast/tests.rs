//! Tests for the broadcasting substrate: the numpy shape/stride rules, the
//! ndarray layout drivers (`walk` / `apply1..3`) and `broadcast_to` /
//! [`fill_broadcast`](super::fill_broadcast).

use super::*;
use crate::buffer::test_support::{switch, switch_guard};
use crate::buffer::{Buffer, Storage};
use evoxc_core::DType;

/// A deterministic value generator (no `rand` dependency).
fn values(count: usize, seed: i32) -> Vec<i32> {
    let mut state = seed as u64 | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as i32) % 200 - 100
        })
        .collect()
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Independent right-aligned C-order broadcast reference: for flat result
/// index `flat`, the operand element it reads.
///
/// Deliberately NOT the implementation under test: it walks the result
/// coordinates one axis at a time (no ndarray views, no strides).
fn reference_slot(out_shape: &[usize], operand_shape: &[usize], flat: usize) -> usize {
    let rank = out_shape.len();
    let offset = rank - operand_shape.len();
    let mut pos = vec![0usize; rank];
    let mut rest = flat;
    for axis in (0..rank).rev() {
        pos[axis] = rest % out_shape[axis];
        rest /= out_shape[axis];
    }
    let (mut slot, mut stride) = (0usize, 1usize);
    for axis in (0..operand_shape.len()).rev() {
        let p = if operand_shape[axis] == 1 {
            0
        } else {
            pos[offset + axis]
        };
        slot += p * stride;
        stride *= operand_shape[axis];
    }
    slot
}

/// `apply2` against the independent reference, for every element.
fn check_apply2_i32(out_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) {
    let count = product(out_shape);
    let a = values(product(a_shape), 7);
    let b = values(product(b_shape), 11);
    let f = |x: i32, y: i32| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(7));
    let mut out = vec![0i32; count];
    apply2(&mut out, out_shape, &a, a_shape, &b, b_shape, f);
    for (flat, produced) in out.iter().enumerate() {
        let expected = f(
            a[reference_slot(out_shape, a_shape, flat)],
            b[reference_slot(out_shape, b_shape, flat)],
        );
        assert_eq!(
            *produced, expected,
            "i32 out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
        );
    }
}

/// The f32 counterpart (bit comparison, so signed zeros stay visible).
fn check_apply2_f32(out_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) {
    let count = product(out_shape);
    let a: Vec<f32> = values(product(a_shape), 5).iter().map(|v| *v as f32 * 0.5).collect();
    let b: Vec<f32> = values(product(b_shape), 9).iter().map(|v| *v as f32 * 0.25).collect();
    let f = |x: f32, y: f32| x * 1.5 - y;
    let mut out = vec![f32::NAN; count];
    apply2(&mut out, out_shape, &a, a_shape, &b, b_shape, f);
    for (flat, produced) in out.iter().enumerate() {
        let expected = f(
            a[reference_slot(out_shape, a_shape, flat)],
            b[reference_slot(out_shape, b_shape, flat)],
        );
        assert_eq!(
            produced.to_bits(),
            expected.to_bits(),
            "f32 out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
        );
    }
}

/// `apply3` against the independent reference (a row-predicate `select`).
fn check_apply3_select(
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
    let mut out = vec![0i32; count];
    apply3(
        &mut out, out_shape, &p, p_shape, &a, a_shape, &b, b_shape, f,
    );
    for (flat, produced) in out.iter().enumerate() {
        let expected = f(
            p[reference_slot(out_shape, p_shape, flat)],
            a[reference_slot(out_shape, a_shape, flat)],
            b[reference_slot(out_shape, b_shape, flat)],
        );
        assert_eq!(
            *produced, expected,
            "select out {out_shape:?}, pred {p_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
        );
    }
}

/// `fill_broadcast` against the independent reference.
fn check_fill(shape: &[usize], out_shape: &[usize]) {
    let source = values(product(shape), 13);
    let filled = fill_broadcast(&source, shape, out_shape);
    assert_eq!(filled.len(), product(out_shape), "{shape:?} -> {out_shape:?}");
    for (flat, produced) in filled.iter().enumerate() {
        let expected = source[reference_slot(out_shape, shape, flat)];
        assert_eq!(*produced, expected, "{shape:?} -> {out_shape:?}: element {flat}");
    }
}

#[test]
fn broadcast_strides_align_right() {
    assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
    assert_eq!(broadcast_strides(&[1, 3], &[2, 3]), vec![0, 1]);
    assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
}

#[test]
fn apply2_broadcasts_both_sides() {
    let mut out = vec![0i32; 6];
    apply2(&mut out, &[2, 3], &[1, 2, 3], &[1, 3], &[1, 2], &[2, 1], |a, b| a + b);
    assert_eq!(out, vec![2, 3, 4, 3, 4, 5]);
}

#[test]
fn broadcast_to_rejects_what_it_always_rejected() {
    let buffer = Buffer::new(DType::Int32, vec![3], Storage::I32(vec![1, 2, 3])).unwrap();
    // Right-aligned only: a length-3 axis cannot serve a length-2 target.
    let error = broadcast_to(&buffer, &[2]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "ShapeError: operand shapes cannot broadcast: (3) vs (2)"
    );
    // A target that the source would widen into is refused by the strict
    // equality check (the broadcast of `(1, 3)` is `(1, 3)`, not `(3,)`).
    let row = Buffer::new(DType::Int32, vec![1, 3], Storage::I32(vec![1, 2, 3])).unwrap();
    let error = broadcast_to(&row, &[3]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "ShapeError: operand shape (1, 3) cannot be broadcast to target shape (3)"
    );
    // An equal shape short-circuits to a clone, values untouched.
    let same = broadcast_to(&buffer, &[3]).unwrap();
    assert_eq!(same.storage, Storage::I32(vec![1, 2, 3]));
}

/// Element `i` of a broadcast result is element `slots[i]` of the source.
fn same_sequence<A: Copy, B: Copy>(
    out: &[A],
    source: &[B],
    slots: &[usize],
    eq: impl Fn(A, B) -> bool,
) {
    assert_eq!(out.len(), slots.len());
    for (index, slot) in slots.iter().enumerate() {
        assert!(eq(out[index], source[*slot]), "element {index} (slot {slot})");
    }
}

#[test]
fn every_dtype_materializes_through_the_block_copy() {
    // The pooled copy serves all 14 storages; bool and the complex and
    // float pairs are the ones a per-dtype arm could silently drop.
    let cases: [(Storage, Vec<usize>, Vec<usize>); 3] = [
        (Storage::Bool(vec![true, false]), vec![2], vec![3, 2]),
        (
            Storage::C64(vec![
                num_complex::Complex::new(1.0, 2.0),
                num_complex::Complex::new(-0.0, 0.5),
            ]),
            vec![1, 2],
            vec![2, 3, 2],
        ),
        (Storage::F64(vec![1.5, -0.0, f64::NAN]), vec![3, 1], vec![3, 4]),
    ];
    for (storage, shape, out_shape) in cases {
        let dtype = storage.dtype();
        let buffer = Buffer::new(dtype, shape.clone(), storage).unwrap();
        let expanded = broadcast_to(&buffer, &out_shape).unwrap();
        assert_eq!(expanded.shape, out_shape);
        // The general odometer is the reference: same slot per element.
        let mut slots = Vec::new();
        let strides = broadcast_strides(&shape, &out_shape);
        walk(out_shape.len(), &out_shape, &[strides], |offsets| slots.push(offsets[0]));
        match (&expanded.storage, &buffer.storage) {
            (Storage::Bool(out), Storage::Bool(source)) => {
                same_sequence(out, source, &slots, |a, b| a == b)
            }
            (Storage::C64(out), Storage::C64(source)) => {
                same_sequence(out, source, &slots, |a, b| a == b)
            }
            (Storage::F64(out), Storage::F64(source)) => {
                same_sequence(out, source, &slots, |a, b| a.to_bits() == b.to_bits())
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}

#[test]
fn scalar_buffer_broadcasts() {
    let scalar = Buffer::new(DType::Int32, vec![], Storage::I32(vec![7])).unwrap();
    let out = broadcast_to(&scalar, &[2, 2]).unwrap();
    assert_eq!(out.storage, Storage::I32(vec![7, 7, 7, 7]));
}

#[test]
fn apply2_agrees_with_the_reference_on_every_layout() {
    // (out, left, right): the broadcast shapes a real graph produces and
    // every alignment family the right-aligned rule has.
    let cases: &[(&[usize], &[usize], &[usize])] = &[
        (&[2, 3], &[1, 3], &[2, 1]),
        (&[2, 2, 2], &[2, 1], &[2]),
        (&[4, 3, 5], &[4, 3, 5], &[5]),
        (&[4, 3, 5], &[1, 1, 5], &[1, 1, 1]),
        (&[4096, 64], &[4096, 64], &[4096, 1]),
        (&[1000, 1000], &[1000, 1000], &[1000]),
        (&[3], &[1], &[3]),
        (&[7, 6], &[7, 6], &[6]),
        (&[5], &[5], &[5]),
        // A per-row scalar, a short trailing tile and a rank-0 operand.
        (&[500, 100], &[500, 100], &[500, 1]),
        (&[500, 50, 4], &[500, 50, 4], &[4]),
        (&[2, 3, 4], &[3, 4], &[2, 1, 1]),
        (&[3, 4], &[3, 1], &[1, 4]),
        (&[6], &[6], &[]),
        (&[], &[], &[]),
    ];
    for (out_shape, a_shape, b_shape) in cases {
        check_apply2_i32(out_shape, a_shape, b_shape);
        check_apply2_f32(out_shape, a_shape, b_shape);
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
fn a_rank_zero_operand_and_a_rank_zero_result_are_one_element() {
    // Rank-0 views broadcast to each other and visit exactly once; the
    // rank-1 result with a rank-0 operand agrees element for element.
    check_apply2_i32(&[3, 2], &[1, 1], &[1, 1]);
    check_apply2_i32(&[], &[], &[]);
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
    let mut out: Vec<i32> = Vec::new();
    let pred = [true, false, true];
    apply3(
        &mut out,
        &[2, 0, 3],
        &pred,
        &[3],
        &[2, 0, 3],
        &[2, 0, 3],
        &[3],
        &[3],
        |p, x, y| if p { x } else { y },
    );
    assert!(out.is_empty());
    let filled: Vec<i32> = fill_broadcast(&[1, 2], &[2], &[0, 3]);
    assert!(filled.is_empty());
    let filled: Vec<i32> = fill_broadcast(&[1, 2], &[2], &[0, 2]);
    assert!(filled.is_empty());
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
    // A rank-0 operand holding a NaN: every element is NaN.
    let rows = [f32::NAN, -0.0, 1.0, 0.0, -1.0, 2.0];
    let mut out = [0.0f32; 6];
    apply2(&mut out, &[2, 3], &rows, &[2, 3], &[f32::NAN], &[], maximum);
    assert!(out.iter().all(|value| value.is_nan()));
    // A signed zero against rows: NaN stays NaN, and -0.0 / 0.0 keep
    // their signs where they win.
    let mut out = [0.0f32; 6];
    apply2(&mut out, &[2, 3], &rows, &[2, 3], &[-0.0f32], &[], maximum);
    assert!(out[0].is_nan());
    assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
    // i32 wraparound on both the row-scalar and the trailing-tile layout.
    let mut out = [0i32; 4];
    apply2(&mut out, &[2, 2], &[i32::MAX, 1], &[2, 1], &[1, 2], &[2], |x, y| {
        x.wrapping_add(y)
    });
    assert_eq!(out, [i32::MIN, i32::MIN + 1, 2, 3]);
}

#[test]
fn apply3_agrees_with_the_reference_on_every_select_layout() {
    // (out, predicate, on-true, on-false)
    type TernaryCase<'a> = (&'a [usize], &'a [usize], &'a [usize], &'a [usize]);
    let cases: &[TernaryCase<'_>] = &[
        (&[2, 3], &[3], &[2, 3], &[1]),
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
        (&[500, 50, 4], &[500, 1, 1], &[500, 50, 4], &[500, 50, 4]),
        (&[1024, 256], &[1024, 1], &[1024, 256], &[256]),
    ];
    for (out_shape, p_shape, a_shape, b_shape) in cases {
        check_apply3_select(out_shape, p_shape, a_shape, b_shape);
    }
}

#[test]
fn a_size_one_operand_is_a_captured_scalar() {
    // A rank-0 operand, a `[1, 1]` operand and two size-1 operands at once
    // all broadcast to a constant; the drivers capture that value and drop
    // the operand from the `Zip`. Every case is still checked against the
    // independent `reference_slot` reading, so a captured scalar that landed
    // in the wrong slot would show up.
    let cases: &[(&[usize], &[usize], &[usize])] = &[
        (&[2, 3], &[], &[2, 3]),             // rank-0 left
        (&[2, 3], &[2, 3], &[1, 1]),         // `[1, 1]` right
        (&[5], &[], &[]),                    // both size-1, rank-0
        (&[2, 3, 4], &[1, 1], &[1, 1, 1]),   // both size-1, shaped
        (&[500, 50, 4], &[1, 1, 1], &[4]),   // the xor_p500 bias shapes
        (&[], &[], &[]),                     // one element in total
    ];
    for (out_shape, a_shape, b_shape) in cases {
        check_apply2_i32(out_shape, a_shape, b_shape);
        check_apply2_f32(out_shape, a_shape, b_shape);
    }
    // Argument order is `f(left, right)` in every arm, so a captured scalar
    // cannot silently swap sides.
    let mut out = [0i32; 4];
    apply2(&mut out, &[2, 2], &[7], &[], &[1, 2], &[2], |x, y| x * 10 + y);
    assert_eq!(out, [71, 72, 71, 72]);
    let mut out = [0i32; 4];
    apply2(&mut out, &[2, 2], &[1, 2], &[2], &[9], &[], |x, y| x * 10 + y);
    assert_eq!(out, [19, 29, 19, 29]);
    let mut out = [0i32; 3];
    apply2(&mut out, &[3], &[4], &[1, 1], &[5], &[], |x, y| x * 10 + y);
    assert_eq!(out, [45, 45, 45]);
}

#[test]
fn a_size_one_operand_is_a_captured_scalar_in_select() {
    // (out, predicate, on-true, on-false): a rank-0 predicate, a `[1, 1]`
    // on-true, a rank-0 on-false, and all three size-1 at once.
    type SelectCase<'a> = (&'a [usize], &'a [usize], &'a [usize], &'a [usize]);
    let cases: &[SelectCase<'_>] = &[
        (&[2, 3], &[], &[2, 3], &[2, 3]),
        (&[2, 3], &[2, 1], &[1, 1], &[2, 3]),
        (&[2, 3], &[2, 1], &[2, 3], &[]),
        (&[2, 3], &[], &[2, 3], &[]),
        (&[2, 3, 4], &[1, 1], &[1, 1, 1], &[]),
        (&[5, 5], &[1, 1], &[1, 1], &[1, 1]),
    ];
    for (out_shape, p_shape, a_shape, b_shape) in cases {
        check_apply3_select(out_shape, p_shape, a_shape, b_shape);
    }
    // The predicate is operand 0 and the branches keep their order:
    // `true` picks the on-true side, `false` the captured on-false scalar.
    let mut out = [0i32; 3];
    apply3(
        &mut out,
        &[3],
        &[true],
        &[],
        &[1, 2, 3],
        &[3],
        &[9],
        &[1, 1],
        |pick, x, y| if pick { x } else { y },
    );
    assert_eq!(out, [1, 2, 3]);
    let mut out = [0i32; 3];
    apply3(
        &mut out,
        &[3],
        &[false],
        &[1, 1],
        &[1, 2, 3],
        &[3],
        &[9],
        &[1, 1],
        |pick, x, y| if pick { x } else { y },
    );
    assert_eq!(out, [9, 9, 9]);
}

#[test]
fn fill_broadcast_from_a_size_one_source_is_one_fill() {
    // A rank-0 source and a shaped-but-size-1 source both broadcast to a
    // constant; the pooled destination is still written in full.  Each
    // source shape is paired only with targets of at least its own rank
    // (a broadcast never lowers the rank, and `reference_slot` right-aligns
    // the source).
    let cases: &[(&[usize], &[usize])] = &[
        (&[], &[7]),
        (&[], &[2, 3]),
        (&[], &[5, 1, 4]),
        (&[], &[2000, 50, 4]),
        (&[1], &[7]),
        (&[1], &[2, 3]),
        (&[1], &[5, 1, 4]),
        (&[1, 1], &[2, 3]),
        (&[1, 1], &[5, 1, 4]),
        (&[1, 1], &[2000, 50, 4]),
        (&[1, 1, 1], &[5, 1, 4]),
        (&[1, 1, 1], &[2000, 50, 4]),
    ];
    for (shape, out_shape) in cases {
        check_fill(shape, out_shape);
    }
    // The value is the source's single element, not a zero fill.
    let filled = fill_broadcast(&[2.5f32], &[1, 1], &[3, 2]);
    assert_eq!(filled, vec![2.5f32; 6]);
    let filled = fill_broadcast(&[-7i64], &[], &[2, 2]);
    assert_eq!(filled, vec![-7i64; 4]);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn a_one_element_operand_whose_shape_claims_two_is_not_a_scalar() {
    // `[9]` with shape `[2]` is a MALFORMED operand, not a scalar: the
    // size-1 arms validate the shape's element count too, so this keeps the
    // pre-existing fallback (which reads the second element the slice does
    // not have) instead of silently filling both result elements with `9`.
    // A fast path keyed on the slice length alone would make this call
    // succeed and this test fail with "test did not panic".
    let mut out = [0i32; 2];
    apply2(&mut out, &[2], &[9], &[2], &[10, 20], &[2], |a, b| a + b);
}

#[test]
fn fill_broadcast_agrees_with_the_reference() {
    // `fill_broadcast` consults the A/B switch through `copy_runs`, so no
    // instrument (or sibling test) may flip it mid-run.
    let _serialized = switch_guard();
    // Not compared against `broadcast_to`: that is `fill_broadcast` now, so
    // the reference has to be independent.
    let cases: &[(&[usize], &[usize])] = &[
        (&[5], &[7, 5]),
        (&[3], &[499, 3]),
        (&[1], &[9, 4, 1]),
        (&[6], &[2, 3, 6]),
        (&[4], &[2000, 50, 4]),
        (&[], &[2, 3]),
        (&[2, 3], &[2, 3]),
        (&[500, 1, 50], &[500, 4, 50]),
        (&[500, 1], &[500, 100]),
        (&[1, 1, 1], &[5, 1, 4]),
        (&[3, 4], &[2, 3, 4]),
    ];
    for (shape, out_shape) in cases {
        check_fill(shape, out_shape);
    }
}

#[test]
fn a_pooled_broadcast_result_is_fully_overwritten() {
    // `map2_runs` consults both A/B switches, so no instrument may hold one
    // while this runs.
    let _serialized = switch_guard();
    // A parked garbage stock of the result's exact size must not leak a
    // single NaN into the result: the broadcast map writes every element.
    let out_shape = [2000usize, 50, 4];
    let count = product(&out_shape);
    let left: Vec<f32> = (0..count).map(|index| (index % 97) as f32 * 0.5).collect();
    let right = [0.5f32, -1.5, 2.0, 3.25];
    crate::buffer::test_support::garbage_f32(count);
    let result = crate::buffer::map2_runs(
        DType::Float32,
        &out_shape,
        &left,
        &out_shape,
        &right,
        &[4],
        |a, b| a + b,
    )
    .unwrap();
    match &result.storage {
        Storage::F32(produced) => {
            assert_eq!(produced.len(), count);
            for (flat, value) in produced.iter().enumerate() {
                let expected = left[flat] + right[flat % 4];
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "element {flat} was not rewritten"
                );
            }
        }
        other => panic!("unexpected storage {other:?}"),
    }
}

#[test]
fn the_view_path_agrees_with_the_flat_fast_path() {
    // `map2`/`map3` are the flat typed-slice builders and `apply2`/`apply3`
    // are the ndarray layout drivers: over full-shape operands both must
    // produce the identical storage.
    let out_shape = [3usize, 4];
    let count = product(&out_shape);
    let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 3.0).collect();
    let right: Vec<f32> = (0..count).map(|index| ((index * 7) % 11) as f32).collect();
    let f2 = |a: f32, b: f32| a * 2.0 - b;
    let flat = crate::buffer::map2(DType::Float32, &out_shape, &left, &right, f2).unwrap();
    let mut driven = vec![0.0f32; count];
    apply2(&mut driven, &out_shape, &left, &out_shape, &right, &out_shape, f2);
    assert_eq!(flat.storage, Storage::F32(driven));

    let pred: Vec<bool> = (0..count).map(|index| index % 3 == 0).collect();
    let f3 = |p: bool, x: f32, y: f32| if p { x } else { y };
    let flat3 = crate::buffer::map3(
        DType::Float32, &out_shape, &pred, &left, &right, f3,
    )
    .unwrap();
    let mut driven3 = vec![0.0f32; count];
    apply3(
        &mut driven3,
        &out_shape,
        &pred,
        &out_shape,
        &left,
        &out_shape,
        &right,
        &out_shape,
        f3,
    );
    assert_eq!(flat3.storage, Storage::F32(driven3));
}


// ---------------------------------------------------------------------------
// The repeating-run specialization
// ---------------------------------------------------------------------------

/// The layout cases the repeating-run driver must cover: an innermost `(n,)`
/// bias, a `(1, ..., 1, n)` operand, a whole-storage repeat, two operands with
/// different short periods, and a rank-1 result.  Every one of them is checked
/// against the independent [`reference_slot`] reading AND against the ndarray
/// `Zip` the specialization replaces, so both the values and the agreement
/// with the reference driver are pinned.
#[test]
fn the_repeating_run_driver_covers_the_innermost_broadcast_family() {
    let _serialized = switch_guard();
    type Case<'a> = (&'a [usize], &'a [usize], &'a [usize]);
    let cases: &[Case<'_>] = &[
        // The xor_p500 shape: a `(4,)` bias beside a (pop, m, n) result.
        (&[2000, 50, 4], &[2000, 50, 4], &[4]),
        (&[64, 8, 4], &[64, 8, 4], &[4]),
        // A shaped size-1-prefixed tile and a whole-storage repeat.
        (&[32, 8, 4], &[32, 8, 4], &[1, 1, 4]),
        (&[32, 8, 4], &[32, 8, 4], &[1, 8, 4]),
        (&[32, 8, 4], &[32, 8, 4], &[8, 4]),
        // A shorter innermost extent, and a period covering the whole result
        // (read straight out of the operand instead of a tile).
        (&[64, 2, 4], &[64, 2, 4], &[1, 2, 4]),
        (&[16, 64], &[16, 64], &[16, 64]),
        // Rank 1 on both sides, a rank-2 result with a rank-1 tile, and a
        // tile on the LEFT.
        (&[128, 8], &[128, 8], &[8]),
        (&[4096], &[4096], &[4096]),
        (&[16, 64], &[16, 64], &[64]),
        (&[16, 64], &[1, 64], &[16, 64]),
        (&[16, 4, 8], &[8], &[4, 8]),
        // A rank-0 result beside rank-0 operands (one element: the driver
        // declines and the reference answers).
        (&[], &[], &[]),
    ];
    for (out_shape, a_shape, b_shape) in cases {
        let count = product(out_shape);
        let a: Vec<i32> = values(product(a_shape), 7);
        let b: Vec<i32> = values(product(b_shape), 11);
        let f = |x: i32, y: i32| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(7));

        // The reference driver's answer first, with the specialization
        // disabled, and then the specialized answer.
        let mut generic = vec![0i32; count];
        {
            let _generic = switch(&RUNS_DISABLED, true);
            apply2(&mut generic, out_shape, &a, a_shape, &b, b_shape, f);
        }
        let mut specialized = vec![0i32; count];
        apply2(&mut specialized, out_shape, &a, a_shape, &b, b_shape, f);

        assert_eq!(
            specialized, generic,
            "out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: the specialization disagrees with the ndarray Zip"
        );
        for (flat, produced) in specialized.iter().enumerate() {
            let expected = f(
                a[reference_slot(out_shape, a_shape, flat)],
                b[reference_slot(out_shape, b_shape, flat)],
            );
            assert_eq!(
                *produced, expected,
                "out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
            );
        }
    }
}

/// The layouts the driver must DECLINE keep the ndarray `Zip` — and the values
/// it produces must still agree with the independent reference.
#[test]
fn the_repeating_run_driver_declines_what_it_does_not_cover() {
    let _serialized = switch_guard();
    // A middle-axis broadcast is not a repetition of the operand's storage, and
    // a per-row scalar's period is past the tile cap.
    type Case<'a> = (&'a [usize], &'a [usize], &'a [usize]);
    let cases: &[Case<'_>] = &[
        (&[64, 3, 4], &[64, 3, 4], &[3, 1]),
        (&[64, 6, 4], &[64, 6, 4], &[1, 6, 1]),
        (&[512, 16], &[512, 16], &[512, 1]),
    ];
    for (out_shape, a_shape, b_shape) in cases {
        let count = product(out_shape);
        let a: Vec<i32> = values(product(a_shape), 5);
        let b: Vec<i32> = values(product(b_shape), 9);
        let f = |x: i32, y: i32| x.wrapping_sub(y);
        let mut out = vec![0i32; count];
        assert!(
            !runs2(&mut out, out_shape, &a, a_shape, &b, b_shape, &f),
            "out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: the driver should decline"
        );
        apply2(&mut out, out_shape, &a, a_shape, &b, b_shape, f);
        for (flat, produced) in out.iter().enumerate() {
            let expected = f(
                a[reference_slot(out_shape, a_shape, flat)],
                b[reference_slot(out_shape, b_shape, flat)],
            );
            assert_eq!(
                *produced, expected,
                "out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
            );
        }
    }

    // The classifiers themselves: what counts as a repeat, and how long a
    // block the driver picks.
    assert!(repeat_view(&[0.0f32; 4], &[4], &[2, 3, 4]).is_some());
    assert!(repeat_view(&[0.0f32; 1], &[1, 1], &[64, 16]).is_some());
    assert!(repeat_view(&[0.0f32; 24], &[1, 6, 4], &[64, 6, 4]).is_some());
    assert!(repeat_view(&[0.0f32; 6], &[1, 6, 1], &[64, 6]).is_none()); // rank above
    assert!(repeat_view(&[0.0f32; 7], &[6], &[64, 6]).is_none()); // malformed slice
    assert!(repeat_view(&[0.0f32; 64], &[64, 1], &[64, 16]).is_none()); // per-row scalar
    assert!(repeat_view(&[0.0f32; 3], &[3], &[64, 3, 4]).is_none()); // middle axis

    assert_eq!(run_block(&[4], 4000), Some(RUN_BLOCK_TARGET));
    assert_eq!(run_block(&[4, 16], 4000), Some(RUN_BLOCK_TARGET));
    assert_eq!(run_block(&[3], 4000), Some(66));
    assert_eq!(run_block(&[200], 4000), Some(200));
    assert_eq!(run_block(&[1], 4000), None); // scalars keep the Zip
    assert_eq!(run_block(&[4000], 4000), None); // nothing is tiled
    assert_eq!(run_block(&[300], 4000), None); // past the tile cap
    assert_eq!(run_block(&[200, 128], 4000), None); // common block too long
}

/// `select` (`apply3`) takes the same specialization, with the argument order
/// pinned to `(predicate, on-true, on-false)`.
#[test]
fn the_repeating_run_driver_covers_select() {
    let _serialized = switch_guard();
    let out_shape = [64usize, 8, 4];
    let count = product(&out_shape);
    let pred: Vec<bool> = (0..count).map(|index| index % 5 < 2).collect();
    let tile: Vec<bool> = (0..4).map(|index| index % 2 == 0).collect();
    let a: Vec<i32> = values(count, 3);
    let b: Vec<i32> = values(4, 13);
    let f = |pick: bool, x: i32, y: i32| if pick { x } else { y };

    let mut generic = vec![0i32; count];
    {
        let _generic = switch(&RUNS_DISABLED, true);
        apply3(
            &mut generic,
            &out_shape,
            &pred,
            &out_shape,
            &a,
            &out_shape,
            &b,
            &[4],
            f,
        );
    }
    let mut specialized = vec![0i32; count];
    assert!(runs3(
        &mut specialized,
        &out_shape,
        &pred,
        &out_shape,
        &a,
        &out_shape,
        &b,
        &[4],
        &f
    ));
    assert_eq!(specialized, generic);

    // The predicate itself may be the tiled operand.
    let mut from_predicate = vec![0i32; count];
    let _enabled = switch(&RUNS_DISABLED, false);
    apply3(
        &mut from_predicate,
        &out_shape,
        &tile,
        &[4],
        &a,
        &out_shape,
        &b,
        &[4],
        f,
    );
    for (flat, produced) in from_predicate.iter().enumerate() {
        let expected = f(tile[flat % 4], a[flat], b[flat % 4]);
        assert_eq!(*produced, expected, "element {flat}");
    }
}

/// One value / one combination per dtype, so the driver's dtype coverage is a
/// generic property (it holds no dtype-specific code) rather than a claim.
trait Slot: Copy + PartialEq + std::fmt::Debug {
    fn at(index: usize) -> Self;
    fn combine(left: Self, right: Self) -> Self;
}

macro_rules! integer_slot {
    ($($ty:ty),*) => {$(
        impl Slot for $ty {
            fn at(index: usize) -> Self {
                (index as i64 % 97 - 48) as $ty
            }
            fn combine(left: Self, right: Self) -> Self {
                left.wrapping_add(right)
            }
        }
    )*};
}

macro_rules! float_slot {
    ($($ty:ty),*) => {$(
        impl Slot for $ty {
            fn at(index: usize) -> Self {
                (index % 97) as $ty * 0.5 - 3.0
            }
            fn combine(left: Self, right: Self) -> Self {
                left + right
            }
        }
    )*};
}

integer_slot!(i8, i16, i32, i64, u8, u16, u32, u64);
float_slot!(f32, f64);

impl Slot for half::f16 {
    fn at(index: usize) -> Self {
        half::f16::from_f32((index % 97) as f32 * 0.5 - 3.0)
    }
    fn combine(left: Self, right: Self) -> Self {
        left + right
    }
}

impl Slot for num_complex::Complex<f32> {
    fn at(index: usize) -> Self {
        num_complex::Complex::new(index as f32 * 0.25, (index % 7) as f32 - 3.0)
    }
    fn combine(left: Self, right: Self) -> Self {
        left + right
    }
}

impl Slot for num_complex::Complex<f64> {
    fn at(index: usize) -> Self {
        num_complex::Complex::new(index as f64 * 0.25, (index % 7) as f64 - 3.0)
    }
    fn combine(left: Self, right: Self) -> Self {
        left + right
    }
}

impl Slot for bool {
    fn at(index: usize) -> Self {
        index % 3 == 0
    }
    fn combine(left: Self, right: Self) -> Self {
        left | right
    }
}

fn check_repeating_run_dtype<T: Slot + Send + Sync>() {
    let out_shape = [32usize, 8, 4];
    let count = product(&out_shape);
    let left: Vec<T> = (0..count).map(T::at).collect();
    let bias: Vec<T> = (0..4).map(|index| T::at(index * 5 + 1)).collect();
    let mut specialized = vec![T::at(0); count];
    assert!(
        runs2(
            &mut specialized,
            &out_shape,
            &left,
            &out_shape,
            &bias,
            &[4],
            &T::combine
        ),
        "the (4,) bias beside a (32, 8, 4) result must take the specialization ({})",
        std::any::type_name::<T>()
    );
    for (flat, produced) in specialized.iter().enumerate() {
        assert_eq!(
            *produced,
            T::combine(left[flat], bias[flat % 4]),
            "element {flat} of {:?}",
            std::any::type_name::<T>()
        );
    }
    // And the ndarray `Zip` reference produces exactly the same elements.
    let mut generic = vec![T::at(0); count];
    {
        let _generic = switch(&RUNS_DISABLED, true);
        apply2(
            &mut generic,
            &out_shape,
            &left,
            &out_shape,
            &bias,
            &[4],
            T::combine,
        );
    }
    assert_eq!(specialized, generic);
}

#[test]
fn the_repeating_run_driver_is_dtype_generic() {
    let _serialized = switch_guard();
    // All 14 evoxc dtypes through the same driver: it addresses elements and
    // never touches a value, so every dtype rides the same code.
    check_repeating_run_dtype::<bool>();
    check_repeating_run_dtype::<i8>();
    check_repeating_run_dtype::<i16>();
    check_repeating_run_dtype::<i32>();
    check_repeating_run_dtype::<i64>();
    check_repeating_run_dtype::<u8>();
    check_repeating_run_dtype::<u16>();
    check_repeating_run_dtype::<u32>();
    check_repeating_run_dtype::<u64>();
    check_repeating_run_dtype::<half::f16>();
    check_repeating_run_dtype::<f32>();
    check_repeating_run_dtype::<f64>();
    check_repeating_run_dtype::<num_complex::Complex<f32>>();
    check_repeating_run_dtype::<num_complex::Complex<f64>>();
}

/// NaN and signed zeros ride through the specialization untouched, bit for bit
/// (the closure owns every semantic, the driver only supplies addresses).
#[test]
fn the_repeating_run_driver_keeps_nan_and_signed_zeros() {
    let _serialized = switch_guard();
    let out_shape = [16usize, 8, 4];
    let count = product(&out_shape);
    let rows: Vec<f32> = (0..count)
        .map(|index| match index % 6 {
            0 => -0.0,
            1 => 0.0,
            2 => f32::NAN,
            3 => f32::INFINITY,
            _ => (index % 11) as f32 - 5.0,
        })
        .collect();
    let bias = [1.0f32, -1.0, f32::NAN, 2.0];
    let f = |x: f32, y: f32| if x.is_nan() || y.is_nan() { f32::NAN } else { x * y };

    let mut generic = vec![0.0f32; count];
    {
        let _generic = switch(&RUNS_DISABLED, true);
        apply2(&mut generic, &out_shape, &rows, &out_shape, &bias, &[4], f);
    }
    let mut specialized = vec![0.0f32; count];
    assert!(runs2(
        &mut specialized,
        &out_shape,
        &rows,
        &out_shape,
        &bias,
        &[4],
        &f
    ));
    for (flat, (produced, reference)) in specialized.iter().zip(&generic).enumerate() {
        let expected = f(rows[flat], bias[flat % 4]);
        assert_eq!(produced.to_bits(), reference.to_bits(), "element {flat} vs the Zip");
        assert_eq!(produced.to_bits(), expected.to_bits(), "element {flat} vs the closure");
    }
    // A signed zero really is visible in the result (not just "equal" to 0.0).
    assert!(specialized.iter().any(|value| value.to_bits() == (-0.0f32).to_bits()));
}

/// The repeating copy behind `fill_broadcast` (`broadcast_to`): whole blocks out
/// of one tile, and a whole-storage source is one plain copy.
#[test]
fn the_repeating_copy_covers_fill_broadcast() {
    let _serialized = switch_guard();
    // A `(4,)` source tiled across a (32, 8, 4) target.
    let source = [1i64, -2, 3, 4];
    let filled = fill_broadcast(&source, &[4], &[32, 8, 4]);
    assert_eq!(filled.len(), 32 * 8 * 4);
    for (flat, produced) in filled.iter().enumerate() {
        assert_eq!(*produced, source[flat % 4], "element {flat}");
    }
    // A source that IS the target is one plain copy of its own storage.
    let whole: Vec<u16> = (0..32 * 8 * 4).map(|index| index as u16).collect();
    let filled = fill_broadcast(&whole, &[32, 8, 4], &[32, 8, 4]);
    assert_eq!(filled, whole);
    // And a layout that is not a repetition keeps the ndarray `assign`.
    let middle: Vec<u8> = (0..3).map(|index| index as u8 + 1).collect();
    let filled = fill_broadcast(&middle, &[3, 1], &[8, 3, 4]);
    for (flat, produced) in filled.iter().enumerate() {
        assert_eq!(
            *produced,
            middle[reference_slot(&[8, 3, 4], &[3, 1], flat) % 3],
            "element {flat}"
        );
    }
    assert!(!copy_runs(&mut [0i32; 32], &repeat_view(&[0i32; 1], &[1], &[32]).unwrap()));
}
