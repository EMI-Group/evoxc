//! `conv` — N-D cross-correlation in NCHW layout, with `strides`,
//! `input_dilation`, `kernel_dilation`, `padding` (`"VALID"`/`"SAME"` or
//! explicit per-spatial `[lo, hi]` pairs), `feature_group_count` and
//! `batch_group_count` for rank >= 3 (N-D batched) inputs: exactly the numpy
//! kernel's surface, with the same explicit rejections.

use evoxc_core::{AttrValue, Attrs, DType, Error, Result};
use serde_json::Value as Json;

use crate::buffer::Buffer;
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util::{self, c_strides};
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::Num;
use crate::with_same_type;

use super::common::{attr_json, check_result_shape, each_index, expect_arity};

pub(super) fn register(map: &mut KernelMap) {
    map.insert("conv", conv_kernel);
}

// ---------------------------------------------------------------------------
// conv — N-D cross-correlation, NCHW
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Pad {
    Same,
    Pair(i64, i64),
}

/// Per-batch-group conv geometry (all spatial entries are concrete extents).
struct ConvGeom {
    n: usize,
    c_in: usize,
    c_out: usize,
    gin: usize,
    gout: usize,
    batch_groups: usize,
    in_spatial: Vec<usize>,
    kernel_spatial: Vec<usize>,
    out_spatial: Vec<usize>,
    strides: Vec<i64>,
    input_dilation: Vec<i64>,
    kernel_dilation: Vec<i64>,
    /// Lower padding offset per spatial dim (negative = crop the dilated input).
    pad_lo: Vec<i64>,
}

/// A conv `strides`/`input_dilation`/`kernel_dilation` attribute: absent or
/// `null` is all-`1`, a bare int replicates, a sequence needs one entry per
/// spatial dim.
fn conv_spatial_param(attrs: &Attrs, name: &str, n_spatial: usize) -> Result<Vec<i64>> {
    let raw = match attrs.get(name) {
        None => Json::Null,
        Some(value) => attr_json(value),
    };
    if raw.is_null() {
        return Ok(vec![1; n_spatial]);
    }
    if let Some(value) = raw.as_i64() {
        return Ok(vec![value; n_spatial]);
    }
    let entries = raw.as_array().ok_or_else(|| {
        Error::shape(format!(
            "conv: {name} must be an int or a tuple of {n_spatial} ints, got {raw}"
        ))
    })?;
    if entries.len() != n_spatial {
        return Err(Error::shape(format!(
            "conv: {name} must have {n_spatial} entries, got {}",
            entries.len()
        )));
    }
    entries
        .iter()
        .map(|entry| {
            entry.as_i64().ok_or_else(|| {
                Error::shape(format!(
                    "conv: {name} entries must be positive ints, got {entry}"
                ))
            })
        })
        .collect()
}

/// The `padding` attribute normalized to per-spatial entries (`"VALID"` → 0,
/// `"SAME"` kept as a mode, an int replicates symmetrically, `[lo, hi]` pairs
/// pass through).
fn conv_padding(attrs: &Attrs, n_spatial: usize) -> Result<Vec<Pad>> {
    let raw = match attrs.get("padding") {
        None => Json::String("VALID".to_string()),
        Some(value) => attr_json(value),
    };
    if let Some(mode) = raw.as_str() {
        return match mode {
            "VALID" => Ok(vec![Pad::Pair(0, 0); n_spatial]),
            "SAME" => Ok(vec![Pad::Same; n_spatial]),
            other => Err(Error::shape(format!("conv: unknown padding mode {other:?}"))),
        };
    }
    if let Some(value) = raw.as_i64() {
        if value < 0 {
            return Err(Error::shape("conv: negative padding"));
        }
        return Ok(vec![Pad::Pair(value, value); n_spatial]);
    }
    let entries = raw.as_array().ok_or_else(|| {
        Error::shape(format!("conv: invalid padding {raw}"))
    })?;
    if entries.len() != n_spatial {
        return Err(Error::shape(format!(
            "conv: expected {n_spatial} padding entries, got {}",
            entries.len()
        )));
    }
    entries.iter().map(conv_pad_entry).collect()
}

fn conv_pad_entry(entry: &Json) -> Result<Pad> {
    if let Some(value) = entry.as_i64() {
        if value < 0 {
            return Err(Error::shape("conv: negative padding"));
        }
        return Ok(Pad::Pair(value, value));
    }
    if let Some(pair) = entry.as_array() {
        if pair.len() == 2 {
            if let (Some(lo), Some(hi)) = (pair[0].as_i64(), pair[1].as_i64()) {
                if lo < 0 || hi < 0 {
                    return Err(Error::shape("conv: negative padding"));
                }
                return Ok(Pad::Pair(lo, hi));
            }
        }
    }
    Err(Error::shape(format!("conv: invalid padding entry {entry}")))
}

/// A conv group count (`feature_group_count`/`batch_group_count`): positive.
fn conv_group_count(attrs: &Attrs, name: &str) -> Result<usize> {
    let raw = match attrs.get(name) {
        None | Some(AttrValue::Null) => return Ok(1),
        Some(value) => attr_json(value),
    };
    match raw.as_i64() {
        Some(value) if value > 0 => Ok(value as usize),
        _ => Err(Error::shape(format!(
            "conv: {name} must be a positive int, got {raw}"
        ))),
    }
}

/// Effective spatial sizes, per-dim padding offsets and output extents.
///
/// Port of the numpy kernel's step 1-2: dilate (effective size
/// `(d - 1) * dil + 1`), then pad — `"SAME"` (TF) derives `ceil(d / stride)` and
/// may need a NEGATIVE total (cropping) when `input_dilation > 1`.
fn conv_geometry(
    in_spatial: &[usize],
    kernel_spatial: &[usize],
    strides: &[i64],
    in_dil: &[i64],
    k_dil: &[i64],
    pads: &[Pad],
) -> Result<(Vec<i64>, Vec<usize>)> {
    let n_spatial = in_spatial.len();
    let mut pad_lo = Vec::with_capacity(n_spatial);
    let mut out_spatial = Vec::with_capacity(n_spatial);
    for i in 0..n_spatial {
        // Saturating arithmetic: the dilation/stride attributes are user data,
        // so an absurd value must become an explicit extent error, not a panic.
        let d = in_spatial[i] as i64;
        let eff_d = d.saturating_sub(1).saturating_mul(in_dil[i]).saturating_add(1);
        let eff_k = (kernel_spatial[i] as i64)
            .saturating_sub(1)
            .saturating_mul(k_dil[i])
            .saturating_add(1);
        let (lo, hi) = match pads[i] {
            Pad::Same => {
                let out_i = d
                    .saturating_add(strides[i])
                    .saturating_sub(1)
                    .div_euclid(strides[i]);
                let total = out_i
                    .saturating_sub(1)
                    .saturating_mul(strides[i])
                    .saturating_add(eff_k)
                    .saturating_sub(eff_d);
                let lo = total.div_euclid(2);
                (lo, total.saturating_sub(lo))
            }
            Pad::Pair(lo, hi) => (lo, hi),
        };
        let out_i = eff_d
            .saturating_add(lo)
            .saturating_add(hi)
            .saturating_sub(eff_k)
            .div_euclid(strides[i])
            .saturating_add(1);
        if out_i <= 0 {
            return Err(Error::shape(format!(
                "conv: spatial dim {i} output size {out_i} <= 0 (effective kernel larger than the padded input?)"
            )));
        }
        pad_lo.push(lo);
        out_spatial.push(out_i as usize);
    }
    Ok((pad_lo, out_spatial))
}

fn conv_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    conv_impl(operands, spec, out_shape)
}

fn conv_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("conv", operands, 2)?;
    let x: &Buffer = &operands[0];
    let w: &Buffer = &operands[1];
    let x_rank = x.shape.len();
    let w_rank = w.shape.len();
    if x_rank != w_rank {
        return Err(Error::shape(format!(
            "conv: input and kernel ranks differ: {x_rank} vs {w_rank}"
        )));
    }
    if x_rank < 3 {
        return Err(Error::shape(format!(
            "conv: rank must be >= 3 (N, C, *spatial), got {x_rank}"
        )));
    }
    let n_spatial = x_rank - 2;
    let strides = conv_spatial_param(&spec.attrs, "strides", n_spatial)?;
    let in_dil = conv_spatial_param(&spec.attrs, "input_dilation", n_spatial)?;
    let k_dil = conv_spatial_param(&spec.attrs, "kernel_dilation", n_spatial)?;
    for (name, values) in [
        ("strides", &strides),
        ("input_dilation", &in_dil),
        ("kernel_dilation", &k_dil),
    ] {
        for value in values {
            if *value <= 0 {
                return Err(Error::shape(format!(
                    "conv: {name} entries must be positive ints, got {value}"
                )));
            }
        }
    }
    let pads = conv_padding(&spec.attrs, n_spatial)?;
    let feature_groups = conv_group_count(&spec.attrs, "feature_group_count")?;
    let batch_groups = conv_group_count(&spec.attrs, "batch_group_count")?;
    let n = x.shape[0];
    let c_in = x.shape[1];
    let c_out = w.shape[0];
    if c_in % feature_groups != 0 {
        return Err(Error::shape(format!(
            "conv: in_channels {c_in} not divisible by feature_group_count {feature_groups}"
        )));
    }
    if c_out % feature_groups != 0 {
        return Err(Error::shape(format!(
            "conv: out_channels {c_out} not divisible by feature_group_count {feature_groups}"
        )));
    }
    let gin = c_in / feature_groups;
    let gout = c_out / feature_groups;
    if w.shape[1] != gin {
        return Err(Error::shape(format!(
            "conv: kernel in-channels {} != in_channels {c_in} // feature_group_count {feature_groups}",
            w.shape[1]
        )));
    }
    if n % batch_groups != 0 {
        return Err(Error::shape(format!(
            "conv: batch dim {n} not divisible by batch_group_count {batch_groups}"
        )));
    }
    let in_spatial = x.shape[2..].to_vec();
    let kernel_spatial = w.shape[2..].to_vec();
    let (pad_lo, out_spatial) =
        conv_geometry(&in_spatial, &kernel_spatial, &strides, &in_dil, &k_dil, &pads)?;
    let mut expected = vec![n, c_out * batch_groups];
    expected.extend(out_spatial.iter().copied());
    check_result_shape("conv", &out_shape, &expected)?;

    let dtype = spec.result_types[0].dtype;
    let left = x.cast(dtype)?;
    let right = w.cast(dtype)?;
    let geometry = ConvGeom {
        n,
        c_in,
        c_out,
        gin,
        gout,
        batch_groups,
        in_spatial,
        kernel_spatial,
        out_spatial,
        strides,
        input_dilation: in_dil,
        kernel_dilation: k_dil,
        pad_lo,
    };
    let buffer = with_same_type!(left, right, |x_values, w_values| conv_run(
        x_values,
        w_values,
        &geometry,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

fn conv_run<T: Num>(
    x: &[T],
    w: &[T],
    g: &ConvGeom,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let out_spatial_count = util::numel(&g.out_spatial);
    let in_spatial_count = util::numel(&g.in_spatial);
    let kernel_spatial_count = util::numel(&g.kernel_spatial);
    let out_channels = g.c_out * g.batch_groups;
    let per_group_n = g.n / g.batch_groups;
    let mut out = vec![T::zero(); out_spatial_count * out_channels * g.n];
    let in_strides = c_strides(&g.in_spatial);
    let k_strides = c_strides(&g.kernel_spatial);
    // (kernel spatial offset, input spatial offset) pairs that contribute.
    let mut patch: Vec<(usize, usize)> = Vec::new();
    for batch in 0..g.n {
        let mut position = 0usize;
        each_index(&g.out_spatial, |out_index| {
            patch.clear();
            each_index(&g.kernel_spatial, |kernel_index| {
                let mut input_offset = 0usize;
                let mut kernel_offset = 0usize;
                for axis in 0..g.in_spatial.len() {
                    let coordinate = (out_index[axis] as i64)
                        .saturating_mul(g.strides[axis])
                        .saturating_add(
                            (kernel_index[axis] as i64).saturating_mul(g.kernel_dilation[axis]),
                        )
                        .saturating_sub(g.pad_lo[axis]);
                    let rate = g.input_dilation[axis];
                    if coordinate < 0 || coordinate % rate != 0 {
                        return Ok(());
                    }
                    let source = coordinate / rate;
                    if source >= g.in_spatial[axis] as i64 {
                        return Ok(());
                    }
                    input_offset += source as usize * in_strides[axis];
                    kernel_offset += kernel_index[axis] * k_strides[axis];
                }
                patch.push((kernel_offset, input_offset));
                Ok(())
            })?;
            for channel in 0..out_channels {
                let batch_group = channel / g.c_out;
                let kernel_channel = channel % g.c_out;
                let feature_group = kernel_channel / g.gout;
                let in_batch = batch_group * per_group_n + batch / g.batch_groups;
                let x_channel_base = in_batch * g.c_in * in_spatial_count;
                let w_channel_base = kernel_channel * g.gin * kernel_spatial_count;
                let mut acc = T::zero();
                for ci in 0..g.gin {
                    let x_base = x_channel_base + (feature_group * g.gin + ci) * in_spatial_count;
                    let w_base = w_channel_base + ci * kernel_spatial_count;
                    for (kernel_offset, input_offset) in &patch {
                        acc = acc.add(x[x_base + input_offset].mul(w[w_base + kernel_offset]));
                    }
                }
                out[(batch * out_channels + channel) * out_spatial_count + position] = acc;
            }
            position += 1;
            Ok(())
        })?;
    }
    Buffer::new(dtype, out_shape, T::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{assert_close, floats, spec, value};
    use super::*;
    use crate::buffer::Storage;
    use evoxc_core::{AttrValue, Attrs, DType};
    use serde_json::Value as Json;

    // --- conv --------------------------------------------------------------

    fn conv_op(out_shape: &[usize], attrs: Attrs) -> OpSpec {
        spec("conv", DType::Float64, out_shape, attrs)
    }

    fn ramp(rows: usize, columns: usize) -> Value {
        let values: Vec<f64> = (0..rows * columns).map(|i| (i + 1) as f64).collect();
        value(DType::Float64, &[1, 1, rows, columns], Storage::F64(values))
    }

    #[test]
    fn conv_valid_matches_the_hand_computed_correlation() {
        let x = ramp(3, 3);
        // Cross-correlation kernel [[1, 0], [0, 1]] (not flipped).
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let op = conv_op(&[1, 1, 2, 2], Attrs::new());
        let out = conv_impl(&[x, w], &op, vec![1, 1, 2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![1, 1, 2, 2]);
        assert_close(&floats(&out[0]), &[6., 8., 12., 14.]);
    }

    #[test]
    fn conv_same_pads_the_trailing_side() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1., 1., 1., 1.]));
        let attrs = Attrs::from_pairs([("padding", AttrValue::Any(Json::String("SAME".into())))]);
        let op = conv_op(&[1, 1, 3, 3], attrs);
        let out = conv_impl(&[x, w], &op, vec![1, 1, 3, 3]).unwrap();
        assert_close(
            &floats(&out[0]),
            &[12., 16., 9., 24., 28., 15., 15., 17., 9.],
        );
    }

    #[test]
    fn conv_honours_kernel_and_input_dilation() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let attrs = Attrs::from_pairs([("kernel_dilation", AttrValue::Ints(vec![2, 2]))]);
        let op = conv_op(&[1, 1, 1, 1], attrs);
        let out = conv_impl(&[x.clone(), w], &op, vec![1, 1, 1, 1]).unwrap();
        assert_close(&floats(&out[0]), &[10.]);

        let ones = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("input_dilation", AttrValue::Ints(vec![2, 2]))]);
        let op = conv_op(&[1, 1, 4, 4], attrs);
        let out = conv_impl(&[x, ones], &op, vec![1, 1, 4, 4]).unwrap();
        // The 5x5 dilated grid is nonzero only on even coordinates, so each
        // output keeps at most one of the four window corners.
        assert_close(
            &floats(&out[0]),
            &[1., 2., 2., 3., 4., 5., 5., 6., 4., 5., 5., 6., 7., 8., 8., 9.],
        );
    }

    #[test]
    fn conv_applies_strides_and_explicit_padding_pairs() {
        let x = ramp(4, 4);
        let ones = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("strides", AttrValue::Ints(vec![2, 2]))]);
        let op = conv_op(&[1, 1, 2, 2], attrs);
        let out = conv_impl(&[x, ones], &op, vec![1, 1, 2, 2]).unwrap();
        assert_close(&floats(&out[0]), &[14., 22., 46., 54.]);
    }

    #[test]
    fn conv_feature_and_batch_groups() {
        let x = value(
            DType::Float64,
            &[1, 2, 2, 2],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8.]),
        );
        let w = value(DType::Float64, &[2, 1, 1, 1], Storage::F64(vec![1., 2.]));
        let attrs = Attrs::from_pairs([("feature_group_count", AttrValue::Int(2))]);
        let op = conv_op(&[1, 2, 2, 2], attrs);
        let out = conv_impl(&[x, w], &op, vec![1, 2, 2, 2]).unwrap();
        assert_close(&floats(&out[0]), &[1., 2., 3., 4., 10., 12., 14., 16.]);

        // batch_group_count = 2: each batch group is convolved with the full
        // kernel, then the per-group rows are replicated along the channels.
        let x = value(
            DType::Float64,
            &[2, 1, 1, 2],
            Storage::F64(vec![1., 2., 3., 4.]),
        );
        let w = value(DType::Float64, &[1, 1, 1, 1], Storage::F64(vec![1.]));
        let attrs = Attrs::from_pairs([("batch_group_count", AttrValue::Int(2))]);
        let op = conv_op(&[2, 2, 1, 2], attrs);
        let out = conv_impl(&[x, w], &op, vec![2, 2, 1, 2]).unwrap();
        assert_close(&floats(&out[0]), &[1., 2., 3., 4., 1., 2., 3., 4.]);
    }

    #[test]
    fn conv_explicit_padding_pairs_offset_both_sides() {
        let x = ramp(2, 2);
        let w = value(DType::Float64, &[1, 1, 1, 1], Storage::F64(vec![1.]));
        let padding = Json::Array(vec![
            Json::Array(vec![Json::from(1), Json::from(0)]),
            Json::Array(vec![Json::from(0), Json::from(1)]),
        ]);
        let attrs = Attrs::from_pairs([("padding", AttrValue::Any(padding))]);
        let op = conv_op(&[1, 1, 3, 3], attrs);
        let out = conv_impl(&[x, w], &op, vec![1, 1, 3, 3]).unwrap();
        assert_close(&floats(&out[0]), &[0., 0., 0., 1., 2., 0., 3., 4., 0.]);
    }

    #[test]
    fn conv_extreme_dilation_is_an_explicit_error() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("kernel_dilation", AttrValue::Ints(vec![i64::MAX, 1]))]);
        let op = conv_op(&[1, 1, 1, 1], attrs);
        let err = conv_impl(&[x, w], &op, vec![1, 1, 1, 1]).unwrap_err();
        assert!(err.message.contains("output size"), "{}", err.message);
    }

    #[test]
    fn conv_rejects_the_numpy_kernel_rejections() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("feature_group_count", AttrValue::Int(3))]);
        let op = conv_op(&[1, 1, 2, 2], attrs);
        let err = conv_impl(&[x.clone(), w.clone()], &op, vec![1, 1, 2, 2]).unwrap_err();
        assert!(err.message.contains("not divisible by feature_group_count 3"), "{}", err.message);

        let attrs = Attrs::from_pairs([("padding", AttrValue::Any(Json::String("FULL".into())))]);
        let op = conv_op(&[1, 1, 2, 2], attrs);
        let err = conv_impl(&[x, w], &op, vec![1, 1, 2, 2]).unwrap_err();
        assert!(err.message.contains("unknown padding mode"), "{}", err.message);
    }
}
