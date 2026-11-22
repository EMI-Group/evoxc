//! Linear-algebra kernels: `dot`, `conv`, `solve`, `diagonal`, `tril`/`triu`.
//!
//! Semantics source of truth: `evoxc/backends/numpy/kernels/linalg.py`
//! (`dot`/`conv`/`solve`/`diagonal`) and
//! `evoxc/backends/numpy/kernels/indexing.py` (`tril`/`triu`).  The IR's
//! declared result type is authoritative: every kernel evaluates the result
//! shape from `util::result_shape`, allocates exactly that shape, computes in
//! the declared dtype and cross-checks the operand-derived geometry against it
//! before touching an index — a malformed program is an explicit `ShapeError`,
//! never a panic or a silent coercion.
//!
//! `dot`/`conv` accumulate in the DECLARED (promoted) dtype with the numpy
//! matmul/tensordot rules (integer arithmetic wraps, bool is `&&`/`||`).
//! `solve` runs the LU factorization in f64/complex128 work space (the declared
//! dtype — f16/f32/complex64 included — is what the result is stored in, which
//! only removes rounding, never changes the contract).  `diagonal`, `tril` and
//! `triu` copy the operand dtype unchanged.
//!
//! `conv` supports `strides`, `input_dilation`, `kernel_dilation`, `padding`
//! (`"VALID"`/`"SAME"` or explicit per-spatial `[lo, hi]` pairs),
//! `feature_group_count` and `batch_group_count` for rank >= 3 (N-D batched)
//! inputs, i.e. exactly the numpy kernel's surface, with the same explicit
//! rejections.
//!
//! Each op family lives in its own submodule: [`matmul`] owns `dot` (with the
//! f32/f64 BLAS-3 fast path and its threaded row split), [`conv`] the N-D
//! cross-correlation, [`solve`] the LU-based system solve, [`structural`] the
//! `diagonal`/`tril`/`triu` index kernels and [`common`] the attribute, arity
//! and index helpers they share.

use crate::kernels::KernelMap;

mod common;
mod conv;
mod matmul;
mod solve;
mod structural;

pub fn register(map: &mut KernelMap) {
    matmul::register(map);
    conv::register(map);
    solve::register(map);
    structural::register(map);
}

#[cfg(test)]
mod test_util {
    use crate::buffer::{Buffer, Storage};
    use crate::exec::Value;
    use crate::plan::OpSpec;
    use evoxc_core::{Attrs, Dim, DType, ValueType};

    pub(super) fn spec(name: &str, dtype: DType, shape: &[usize], attrs: Attrs) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs,
            operand_types: Vec::new(),
            result_types: vec![ValueType::of(
                dtype,
                shape.iter().map(|dim| Dim::int(*dim as i64)),
            )],
            regions: Vec::new(),
            location: None,
        }
    }

    pub(super) fn value(dtype: DType, shape: &[usize], storage: Storage) -> Value {
        std::rc::Rc::new(Buffer::new(dtype, shape.to_vec(), storage).unwrap())
    }

    pub(super) fn floats(buffer: &Buffer) -> Vec<f64> {
        match &buffer.storage {
            Storage::F64(values) => values.clone(),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    pub(super) fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch: {actual:?} vs {expected:?}");
        for (got, want) in actual.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "got {actual:?}, expected {expected:?}"
            );
        }
    }
}
