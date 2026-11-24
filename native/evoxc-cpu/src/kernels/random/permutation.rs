//! The index-shaped samplers: `random_permutation` (stable argsort of the word
//! stream) and `random_multinomial` (inverse-CDF sampling).
use evoxc_core::{Error, Result};

use crate::exec::{single, RunCtx, Value};
use crate::kernels::util::numel;
use crate::plan::OpSpec;
use crate::scalar::ScalarValue;

use super::bits::stream_words;
use super::key::{extract_key, Words};
use super::sampling::word_uniforms;
use super::{
    algorithm_of, as_f64, as_i64, build, int_attr, require_kind, result_dtype, result_shape,
    scalar_values, SALT_MULTINOMIAL, SALT_PERMUTATION,
};

/// `random_permutation`: stable argsort of the salted word stream.
pub(super) fn permutation_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, false)?;
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    let n_values = scalar_values(&operands[1]);
    let Some(n_value) = n_values.first() else {
        return Err(Error::shape(format!(
            "op '{}': random.permutation needs a rank-0 `n` operand, got an empty tensor",
            spec.name
        )));
    };
    let n = as_i64(*n_value);
    if n < 0 {
        return Err(Error::backend(format!(
            "op '{}': random.permutation n must be >= 0, got {n}",
            spec.name
        )));
    }
    let n = n as usize;
    // A runtime-dynamic (unchecked) declared dim takes its extent from the
    // `n` operand; a static one must agree.
    let declared = &spec.result_types[0].shape;
    if declared.len() != 1 {
        return Err(Error::shape(format!(
            "op '{}': random.permutation result must be rank 1, the declared shape \
             is {declared:?}",
            spec.name
        )));
    }
    let shape = if declared[0].is_dynamic() {
        vec![n]
    } else {
        let shape = result_shape(ctx, spec)?;
        if shape != vec![n] {
            return Err(Error::shape(format!(
                "op '{}': random.permutation produced {n} draws but the declared \
                 result shape is {shape:?}",
                spec.name
            )));
        }
        shape
    };
    let mut order: Vec<usize> = (0..n).collect();
    // `sort_by_key` is stable, matching numpy's `argsort(kind="stable")`:
    // colliding words resolve by index order.
    match stream_words(algorithm, &key, SALT_PERMUTATION, n)? {
        Words::W64(words) => order.sort_by_key(|index| words[*index]),
        Words::W32(words) => order.sort_by_key(|index| words[*index]),
    }
    let values: Vec<ScalarValue> = order
        .iter()
        .map(|index| ScalarValue::I(*index as i64))
        .collect();
    single(build(dtype, shape, &values)?)
}

/// `random_multinomial`: inverse-CDF sampling from a 1-D probability vector.
pub(super) fn multinomial_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, false)?;
    let num_samples = int_attr(spec, "num_samples")?;
    if num_samples < 0 {
        return Err(Error::backend(format!(
            "op '{}': random.multinomial num_samples must be >= 0, got {num_samples}",
            spec.name
        )));
    }
    let shape = result_shape(ctx, spec)?;
    if shape != vec![num_samples as usize] {
        return Err(Error::shape(format!(
            "op '{}': random.multinomial draws {num_samples} sample(s) but the \
             declared result shape is {shape:?}",
            spec.name
        )));
    }
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    let probabilities = &operands[1];
    if probabilities.rank() != 1 {
        return Err(Error::shape(format!(
            "op '{}': random.multinomial input must be 1-D, got shape {:?}",
            spec.name, probabilities.shape
        )));
    }
    let m = probabilities.shape[0];
    if m == 0 {
        return Err(Error::backend(format!(
            "op '{}': random.multinomial input must be non-empty",
            spec.name
        )));
    }
    let p: Vec<f64> = scalar_values(probabilities)
        .into_iter()
        .map(as_f64)
        .collect();
    let mut total = 0.0f64;
    for probability in &p {
        if *probability < 0.0 {
            return Err(Error::backend(format!(
                "op '{}': random.multinomial input probabilities must be non-negative",
                spec.name
            )));
        }
        total += *probability;
    }
    if (total - 1.0).abs() > 1e-6 {
        return Err(Error::backend(format!(
            "op '{}': random.multinomial input probabilities must sum to 1 (got {total})",
            spec.name
        )));
    }
    let count = numel(&shape);
    let u = word_uniforms(&stream_words(algorithm, &key, SALT_MULTINOMIAL, count)?);
    // Sequential cumulative sum (the `np.cumsum` order).
    let mut cumulative = Vec::with_capacity(m);
    let mut running = 0.0f64;
    for probability in &p {
        running += *probability;
        cumulative.push(running);
    }
    let mut values = Vec::with_capacity(count);
    for draw in &u {
        // `np.searchsorted(cum, u)` (side="left"): first index with cum >= u,
        // clipped defensively to the last bin.
        let mut index = 0usize;
        while index < m && cumulative[index] < *draw {
            index += 1;
        }
        values.push(ScalarValue::I(index.min(m - 1) as i64));
    }
    single(build(dtype, shape, &values)?)
}

#[cfg(test)]
mod tests {
    //! Hard-coded numpy reference vectors for the index-shaped samplers.
    use super::*;
    use crate::kernels::random::test_util::*;
    use evoxc_core::{AttrValue, DType, Program};

    #[test]
    fn permutation_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        let spec_splitmix = spec(
            "random_permutation",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![8]),
        );
        let out = permutation_kernel(&mut ctx, &spec_splitmix, &[key_splitmix(3), scalar_i64(8)])
            .expect("permutation");
        assert_eq!(i64_of(&out), vec![6, 2, 7, 0, 3, 1, 4, 5]);

        let spec_threefry = spec(
            "random_permutation",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Int64)),
            ],
            (DType::Int64, vec![9]),
        );
        let out = permutation_kernel(
            &mut ctx,
            &spec_threefry,
            &[key_words(&[4, 5]), scalar_i64(9)],
        )
        .expect("permutation");
        assert_eq!(i64_of(&out), vec![0, 2, 3, 5, 4, 8, 6, 1, 7]);

        let spec_philox = spec(
            "random_permutation",
            &[
                algorithm("philox4x32_10"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![10]),
        );
        let out = permutation_kernel(
            &mut ctx,
            &spec_philox,
            &[key_words(&[1, 2, 3, 4]), scalar_i64(10)],
        )
        .expect("permutation");
        assert_eq!(i64_of(&out), vec![8, 2, 0, 6, 3, 9, 5, 7, 4, 1]);
    }

    #[test]
    fn multinomial_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);
        let probabilities = value(
            DType::Float64,
            vec![4],
            &[0.1, 0.2, 0.3, 0.4]
                .iter()
                .map(|p| ScalarValue::F(*p))
                .collect::<Vec<_>>(),
        );
        let spec_splitmix = spec(
            "random_multinomial",
            &[
                algorithm("splitmix64"),
                ("num_samples", AttrValue::Int(5)),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![5]),
        );
        let out = multinomial_kernel(&mut ctx, &spec_splitmix, &[key_splitmix(11), probabilities])
            .expect("multinomial");
        assert_eq!(i64_of(&out), vec![1, 3, 2, 3, 2]);

        let probabilities = value(
            DType::Float32,
            vec![3],
            &[0.25, 0.25, 0.5]
                .iter()
                .map(|p| ScalarValue::F(*p))
                .collect::<Vec<_>>(),
        );
        let spec_threefry = spec(
            "random_multinomial",
            &[
                algorithm("threefry2x32"),
                ("num_samples", AttrValue::Int(6)),
                ("dtype", AttrValue::DType(DType::Int64)),
            ],
            (DType::Int64, vec![6]),
        );
        let out = multinomial_kernel(
            &mut ctx,
            &spec_threefry,
            &[key_words(&[2, 3]), probabilities],
        )
        .expect("multinomial");
        assert_eq!(i64_of(&out), vec![2, 2, 0, 1, 2, 1]);
    }
}
