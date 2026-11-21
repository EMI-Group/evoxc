//! The neutral rule surface: [`RuleCtx`] (emission into the active builder) and
//! [`RuleOp`] (one op presented to a batching / JVP / VJP rule).
//!
//! Builtin rules never touch a `Program` directly: they read `RuleOp` (the op's
//! attributes, its original types and its rewritten operands/results) and call
//! `RuleCtx` helpers, which re-validate and re-infer through the native schema
//! table exactly like the frontend does.

use evoxc_core::attr::{ArrayPayload, AttrValue, Attrs};
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Location, ValueId};
use evoxc_core::value_type::ValueType;

use crate::builder::Builder;

/// Emission context handed to every builtin rule.
pub struct RuleCtx<'a> {
    /// The builder the rule appends to (the transform's output module).
    pub builder: &'a mut Builder,
    /// The pass's fresh batch `Dim` name (`vectorize`/`vmap`); `None` in AD.
    pub batch_dim: Option<String>,
}

impl<'a> RuleCtx<'a> {
    /// A context over `builder` with no batch dim (AD rules).
    pub fn new(builder: &'a mut Builder) -> RuleCtx<'a> {
        RuleCtx {
            builder,
            batch_dim: None,
        }
    }

    /// A context over `builder` carrying the pass's batch dim name.
    pub fn with_batch_dim(builder: &'a mut Builder, batch_dim: Option<String>) -> RuleCtx<'a> {
        RuleCtx { builder, batch_dim }
    }

    // ---- types -----------------------------------------------------------

    /// The type of a value of the output module.
    pub fn ty(&self, id: ValueId) -> Result<ValueType> {
        self.builder.value_type(id)
    }

    /// The dtype of a value of the output module.
    pub fn dtype(&self, id: ValueId) -> Result<DType> {
        Ok(self.ty(id)?.dtype)
    }

    /// The rank of a value of the output module.
    pub fn rank(&self, id: ValueId) -> Result<usize> {
        Ok(self.ty(id)?.rank())
    }

    /// The shape of a value of the output module.
    pub fn shape(&self, id: ValueId) -> Result<Vec<Dim>> {
        Ok(self.ty(id)?.shape)
    }

    // ---- emission --------------------------------------------------------

    /// Emit `name(operands)` with native inference; all results.
    pub fn op(&mut self, name: &str, operands: &[ValueId], attrs: &Attrs) -> Result<Vec<ValueId>> {
        self.builder.emit(name, operands, attrs, None, None)
    }

    /// Emit `name(operands)` expecting exactly one result.
    pub fn op1(&mut self, name: &str, operands: &[ValueId], attrs: &Attrs) -> Result<ValueId> {
        self.builder.emit_typed_one(name, operands, attrs)
    }

    /// Emit a no-attribute one-result op.
    pub fn call(&mut self, name: &str, operands: &[ValueId]) -> Result<ValueId> {
        self.op1(name, operands, &Attrs::new())
    }

    /// Emit a no-attribute unary op.
    pub fn unary(&mut self, name: &str, operand: ValueId) -> Result<ValueId> {
        self.call(name, &[operand])
    }

    /// Emit a no-attribute binary op.
    pub fn binary(&mut self, name: &str, left: ValueId, right: ValueId) -> Result<ValueId> {
        self.call(name, &[left, right])
    }

    /// Emit a no-attribute ternary op (`select`).
    pub fn ternary(
        &mut self,
        name: &str,
        first: ValueId,
        second: ValueId,
        third: ValueId,
    ) -> Result<ValueId> {
        self.call(name, &[first, second, third])
    }

    /// Emit a `constant` carrying `payload`.
    pub fn constant(&mut self, payload: ArrayPayload) -> Result<ValueId> {
        self.builder.constant(payload)
    }

    /// Emit a rank-0 constant of `dtype` holding `value`.
    pub fn scalar(&mut self, dtype: DType, value: f64) -> Result<ValueId> {
        self.constant(scalar_payload(dtype, value)?)
    }

    /// Emit a rank-0 integer constant (a `bool`/`int` dtype gets its own bytes).
    pub fn scalar_i64(&mut self, dtype: DType, value: i64) -> Result<ValueId> {
        self.scalar(dtype, value as f64)
    }

    /// Emit an int64 `arange(start, stop, step)` constant (gather indices).
    pub fn arange_i64(&mut self, start: i64, stop: i64, step: i64) -> Result<ValueId> {
        if step == 0 {
            return Err(Error::transform("arange: step must be non-zero"));
        }
        let mut values = Vec::new();
        let mut cursor = start;
        while (step > 0 && cursor < stop) || (step < 0 && cursor > stop) {
            values.push(cursor);
            cursor += step;
        }
        self.constant(i64_vector_payload(&values))
    }

    // ---- structural ops --------------------------------------------------

    /// `reshape` to `dims`.
    pub fn reshape(&mut self, value: ValueId, dims: &[Dim]) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("shape", AttrValue::Shape(dims.to_vec()))]);
        self.op1("reshape", &[value], &attrs)
    }

    /// `broadcast` to `dims`.
    pub fn broadcast(&mut self, value: ValueId, dims: &[Dim]) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("shape", AttrValue::Shape(dims.to_vec()))]);
        self.op1("broadcast", &[value], &attrs)
    }

    /// `transpose` with an optional permutation (absent = reverse).
    pub fn transpose(&mut self, value: ValueId, permutation: Option<&[i64]>) -> Result<ValueId> {
        let mut attrs = Attrs::new();
        if let Some(permutation) = permutation {
            attrs.insert("permutation", AttrValue::Ints(permutation.to_vec()));
        }
        self.op1("transpose", &[value], &attrs)
    }

    /// `slice` from `starts` to `limits` with optional `strides`.
    pub fn slice(
        &mut self,
        value: ValueId,
        starts: &[i64],
        limits: &[i64],
        strides: Option<&[i64]>,
    ) -> Result<ValueId> {
        let mut attrs = Attrs::from_pairs([
            ("start_indices", AttrValue::Ints(starts.to_vec())),
            ("limit_indices", AttrValue::Ints(limits.to_vec())),
        ]);
        if let Some(strides) = strides {
            attrs.insert("strides", AttrValue::Ints(strides.to_vec()));
        }
        self.op1("slice", &[value], &attrs)
    }

    /// `pad` with one `(low, high)` pair per axis and a fill `value`.
    pub fn pad(&mut self, value: ValueId, config: &[(i64, i64)], fill: f64) -> Result<ValueId> {
        let rows: Vec<Vec<i64>> = config.iter().map(|(lo, hi)| vec![*lo, *hi]).collect();
        let attrs = Attrs::from_pairs([
            ("padding_config", AttrValue::NestedInts(rows)),
            ("value", AttrValue::Float(fill)),
        ]);
        self.op1("pad", &[value], &attrs)
    }

    /// `concatenate(values, axis=axis)`.
    pub fn concatenate(&mut self, values: &[ValueId], axis: i64) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("axis", AttrValue::Int(axis))]);
        self.op1("concatenate", values, &attrs)
    }

    /// `gather(x, indices, axes=axes)`.
    pub fn gather(
        &mut self,
        x: ValueId,
        indices: ValueId,
        axes: &[i64],
    ) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("axes", AttrValue::Ints(axes.to_vec()))]);
        self.op1("gather", &[x, indices], &attrs)
    }

    /// `scatter(x, indices, updates, axis=axis)`.
    pub fn scatter(
        &mut self,
        x: ValueId,
        indices: ValueId,
        updates: ValueId,
        axis: i64,
    ) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("axis", AttrValue::Int(axis))]);
        self.op1("scatter", &[x, indices, updates], &attrs)
    }

    // ---- reductions ------------------------------------------------------

    /// `reduce_<kind>(value, axes, keepdims)`; `kind` is `sum`, `prod`, `mean`,
    /// `max` or `min` (the IR op name is `reduce_<kind>`).
    pub fn reduce(
        &mut self,
        kind: &str,
        value: ValueId,
        axes: &[i64],
        keepdims: bool,
    ) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([
            ("axes", AttrValue::Ints(axes.to_vec())),
            ("keepdims", AttrValue::Bool(keepdims)),
            ("reduce_op", AttrValue::Str(kind.to_string())),
        ]);
        self.op1(&format!("reduce_{kind}"), &[value], &attrs)
    }

    /// `reduce_sum`.
    pub fn reduce_sum(&mut self, value: ValueId, axes: &[i64], keepdims: bool) -> Result<ValueId> {
        self.reduce("sum", value, axes, keepdims)
    }

    /// `reduce_prod`.
    pub fn reduce_prod(&mut self, value: ValueId, axes: &[i64], keepdims: bool) -> Result<ValueId> {
        self.reduce("prod", value, axes, keepdims)
    }

    /// `reduce_mean`.
    pub fn reduce_mean(&mut self, value: ValueId, axes: &[i64], keepdims: bool) -> Result<ValueId> {
        self.reduce("mean", value, axes, keepdims)
    }

    /// `reduce_max`.
    pub fn reduce_max(&mut self, value: ValueId, axes: &[i64], keepdims: bool) -> Result<ValueId> {
        self.reduce("max", value, axes, keepdims)
    }

    /// `reduce_min`.
    pub fn reduce_min(&mut self, value: ValueId, axes: &[i64], keepdims: bool) -> Result<ValueId> {
        self.reduce("min", value, axes, keepdims)
    }

    /// `argmax(value, axis, keepdims)`.
    pub fn argmax(&mut self, value: ValueId, axis: Option<i64>, keepdims: bool) -> Result<ValueId> {
        self.arg_op("argmax", value, axis, keepdims)
    }

    /// `argmin(value, axis, keepdims)`.
    pub fn argmin(&mut self, value: ValueId, axis: Option<i64>, keepdims: bool) -> Result<ValueId> {
        self.arg_op("argmin", value, axis, keepdims)
    }

    fn arg_op(
        &mut self,
        name: &str,
        value: ValueId,
        axis: Option<i64>,
        keepdims: bool,
    ) -> Result<ValueId> {
        let mut attrs = Attrs::from_pairs([("keepdims", AttrValue::Bool(keepdims))]);
        if let Some(axis) = axis {
            attrs.insert("axis", AttrValue::Int(axis));
        }
        self.op1(name, &[value], &attrs)
    }

    // ---- numeric / linalg ------------------------------------------------

    /// `cast` to `dtype`.
    pub fn cast(&mut self, value: ValueId, dtype: DType) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("dtype", AttrValue::DType(dtype))]);
        self.op1("cast", &[value], &attrs)
    }

    /// `select(condition, on_true, on_false)`.
    pub fn select(
        &mut self,
        condition: ValueId,
        on_true: ValueId,
        on_false: ValueId,
    ) -> Result<ValueId> {
        self.call("select", &[condition, on_true, on_false])
    }

    /// `dot(lhs, rhs)`.
    pub fn dot(&mut self, lhs: ValueId, rhs: ValueId) -> Result<ValueId> {
        self.call("dot", &[lhs, rhs])
    }

    /// `conv(x, kernel)` with the source op's validated conv parameters.
    pub fn conv(&mut self, x: ValueId, kernel: ValueId, params: &ConvParams) -> Result<ValueId> {
        self.op1("conv", &[x, kernel], &params.to_attrs())
    }

    /// `solve(a, b)` (left side only in v1).
    pub fn solve(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.call("solve", &[a, b])
    }

    /// `tril(value, k)`.
    pub fn tril(&mut self, value: ValueId, k: i64) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("k", AttrValue::Int(k))]);
        self.op1("tril", &[value], &attrs)
    }

    /// `triu(value, k)`.
    pub fn triu(&mut self, value: ValueId, k: i64) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([("k", AttrValue::Int(k))]);
        self.op1("triu", &[value], &attrs)
    }

    /// `cumsum(value, axis, reverse)`.
    pub fn cumsum(&mut self, value: ValueId, axis: i64, reverse: bool) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([
            ("axis", AttrValue::Int(axis)),
            ("reverse", AttrValue::Bool(reverse)),
        ]);
        self.op1("cumsum", &[value], &attrs)
    }

    /// `cumprod(value, axis, reverse)`.
    pub fn cumprod(&mut self, value: ValueId, axis: i64, reverse: bool) -> Result<ValueId> {
        let attrs = Attrs::from_pairs([
            ("axis", AttrValue::Int(axis)),
            ("reverse", AttrValue::Bool(reverse)),
        ]);
        self.op1("cumprod", &[value], &attrs)
    }

    /// `stop_gradient(value)`.
    pub fn stop_gradient(&mut self, value: ValueId) -> Result<ValueId> {
        self.call("stop_gradient", &[value])
    }
}

/// One op as a builtin rule sees it.
///
/// `original_*_types` are the types of the source program's operands/results;
/// the batched (`batching`) or rebuilt (`jvp`/`vjp`) values are `operands` /
/// `results` and their `*_types` mirrors.  Rules never follow `Value::owner`.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleOp {
    pub name: String,
    pub attributes: Attrs,
    pub original_operand_types: Vec<ValueType>,
    pub original_result_types: Vec<ValueType>,
    pub operands: Vec<ValueId>,
    pub operand_types: Vec<ValueType>,
    pub results: Vec<ValueId>,
    pub result_types: Vec<ValueType>,
    pub location: Option<Location>,
}

impl RuleOp {
    /// Capture an op's name / attributes / original types from `program`.
    pub fn from_source(program: &evoxc_core::graph::Program, op: &evoxc_core::graph::Op) -> Result<RuleOp> {
        let original_operand_types = op
            .operands
            .iter()
            .map(|id| Ok(program.value(*id)?.ty.clone()))
            .collect::<Result<Vec<_>>>()?;
        let original_result_types = op
            .results
            .iter()
            .map(|id| Ok(program.value(*id)?.ty.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(RuleOp {
            name: op.name.clone(),
            attributes: op.attributes.clone(),
            original_operand_types,
            original_result_types,
            operands: Vec::new(),
            operand_types: Vec::new(),
            results: Vec::new(),
            result_types: Vec::new(),
            location: op.location.clone(),
        })
    }

    /// Attach the rewrite's operand values (their types come from `builder`).
    pub fn bind_operands(mut self, builder: &Builder, operands: &[ValueId]) -> Result<RuleOp> {
        if operands.len() != self.original_operand_types.len() {
            return Err(Error::transform(format!(
                "op '{}': {} rewritten operand(s) for {} source operand(s)",
                self.name,
                operands.len(),
                self.original_operand_types.len()
            )));
        }
        self.operands = operands.to_vec();
        self.operand_types = builder.value_types(operands)?;
        Ok(self)
    }
    /// Attach the rewritten result values (their types come from `builder`).
    pub fn bind_results(mut self, builder: &Builder, results: &[ValueId]) -> Result<RuleOp> {
        self.results = results.to_vec();
        self.result_types = builder.value_types(results)?;
        Ok(self)
    }
    /// The key a rule is registered under (`block:<name>`, `external:<name>`, or
    /// the op name).
    pub fn rule_key(&self) -> String {
        crate::batching::rule_key(&self.name, &self.attributes)
    }

    /// The display name used in diagnostics.
    pub fn display(&self) -> String {
        self.rule_key()
    }

    /// The source program's first operand type.
    pub fn original_operand(&self, index: usize) -> Result<&ValueType> {
        self.original_operand_types.get(index).ok_or_else(|| {
            Error::transform(format!(
                "op '{}': rule read original operand {index} of {}",
                self.name,
                self.original_operand_types.len()
            ))
        })
    }

    /// The source program's first result type.
    pub fn original_result(&self, index: usize) -> Result<&ValueType> {
        self.original_result_types.get(index).ok_or_else(|| {
            Error::transform(format!(
                "op '{}': rule read original result {index} of {}",
                self.name,
                self.original_result_types.len()
            ))
        })
    }

    // ---- attribute accessors --------------------------------------------

    /// A required int attribute.
    pub fn int(&self, name: &str) -> Result<i64> {
        self.attributes.i64(name)
    }

    /// An optional int attribute (`Null` → `None`).
    pub fn opt_int(&self, name: &str) -> Result<Option<i64>> {
        self.attributes.opt_i64(name)
    }

    /// A required int-list attribute.
    pub fn ints(&self, name: &str) -> Result<&[i64]> {
        self.attributes.ints(name)
    }

    /// An optional int-list attribute (`Null` → `None`).
    pub fn opt_ints(&self, name: &str) -> Result<Option<&[i64]>> {
        match self.attributes.get(name) {
            None | Some(AttrValue::Null) => Ok(None),
            Some(_) => Ok(Some(self.attributes.ints(name)?)),
        }
    }

    /// A required bool attribute.
    pub fn bool(&self, name: &str) -> Result<bool> {
        self.attributes.bool(name)
    }

    /// A required str attribute.
    pub fn str(&self, name: &str) -> Result<&str> {
        self.attributes.str(name)
    }

    /// A required dtype attribute.
    pub fn dtype(&self, name: &str) -> Result<DType> {
        self.attributes.dtype(name)
    }

    /// A required shape attribute.
    pub fn shape_attr(&self, name: &str) -> Result<&[Dim]> {
        Ok(self.attributes.shape(name)?.as_slice())
    }

    /// A required float attribute.
    pub fn float(&self, name: &str) -> Result<f64> {
        self.attributes.f64(name)
    }

    /// A required nested-int attribute.
    pub fn nested_ints(&self, name: &str) -> Result<&[Vec<i64>]> {
        self.attributes.nested_ints(name)
    }

    /// A free-form (`any`) attribute.
    pub fn any(&self, name: &str) -> Result<&serde_json::Value> {
        self.attributes.any(name)
    }
}

/// The validated conv parameters of a source `conv` op.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvParams {
    pub strides: Option<Vec<i64>>,
    pub padding: AttrValue,
    pub input_dilation: Option<Vec<i64>>,
    pub kernel_dilation: Option<Vec<i64>>,
    pub feature_group_count: i64,
    pub batch_group_count: i64,
}

impl ConvParams {
    /// Read and validate a source conv op's attributes.
    pub fn from_op(op: &RuleOp) -> Result<ConvParams> {
        let batch_group_count = op.attributes.opt_i64("batch_group_count")?.unwrap_or(1);
        if batch_group_count != 1 {
            return Err(Error::transform(format!(
                "vectorize/jvp: cannot transform 'conv' with batch_group_count {batch_group_count} (v1 gap: batch-grouped conv)"
            )));
        }
        Ok(ConvParams {
            strides: op.opt_ints("strides")?.map(|value| value.to_vec()),
            padding: padding_attr(op),
            input_dilation: op.opt_ints("input_dilation")?.map(|value| value.to_vec()),
            kernel_dilation: op.opt_ints("kernel_dilation")?.map(|value| value.to_vec()),
            feature_group_count: op.attributes.opt_i64("feature_group_count")?.unwrap_or(1),
            batch_group_count,
        })
    }

    /// Rebuild the IR attribute set.
    pub fn to_attrs(&self) -> Attrs {
        let mut attrs = Attrs::new();
        if let Some(strides) = &self.strides {
            attrs.insert("strides", AttrValue::Ints(strides.clone()));
        }
        attrs.insert("padding", self.padding.clone());
        if let Some(dilation) = &self.input_dilation {
            attrs.insert("input_dilation", AttrValue::Ints(dilation.clone()));
        }
        if let Some(dilation) = &self.kernel_dilation {
            attrs.insert("kernel_dilation", AttrValue::Ints(dilation.clone()));
        }
        attrs.insert("feature_group_count", AttrValue::Int(self.feature_group_count));
        attrs.insert("batch_group_count", AttrValue::Int(self.batch_group_count));
        attrs
    }
}

/// The source op's `padding` attribute in its canonical form.
///
/// `padding` is declared `any`: a decoded/schema-defaulted value is an
/// `AttrValue::Any` wrapping a plain string ("VALID"/"SAME"), which the
/// emitted attribute renders as that same string.  A bare string is
/// canonicalized to `AttrValue::Str` so `to_attrs` writes a plain string (the
/// Python frontend's form); lists and ints stay `Any`.
fn padding_attr(op: &RuleOp) -> AttrValue {
    match op.attributes.get("padding") {
        Some(AttrValue::Any(serde_json::Value::String(text))) => {
            AttrValue::Str(text.clone())
        }
        Some(value) => value.clone(),
        None => AttrValue::Str("VALID".to_string()),
    }
}

/// `[1, 2, 3]` — int literals as shape dims.
pub fn dims_i64(values: &[i64]) -> Vec<Dim> {    values.iter().map(|value| Dim::Int(*value)).collect()
}

/// A rank-0 payload of `dtype` holding `value` (raw little-endian bytes).
pub fn scalar_payload(dtype: DType, value: f64) -> Result<ArrayPayload> {
    let bytes: Vec<u8> = match dtype.name() {
        "bool" => vec![u8::from(value != 0.0)],
        "int8" => (value as i8).to_le_bytes().to_vec(),
        "int16" => (value as i16).to_le_bytes().to_vec(),
        "int32" => (value as i32).to_le_bytes().to_vec(),
        "int64" => (value as i64).to_le_bytes().to_vec(),
        "uint8" => (value as u8).to_le_bytes().to_vec(),
        "uint16" => (value as u16).to_le_bytes().to_vec(),
        "uint32" => (value as u32).to_le_bytes().to_vec(),
        "uint64" => (value as u64).to_le_bytes().to_vec(),
        "float16" => f32_to_f16_bits(value as f32).to_le_bytes().to_vec(),
        "float32" => (value as f32).to_le_bytes().to_vec(),
        "float64" => value.to_le_bytes().to_vec(),
        "complex64" => {
            let mut bytes = (value as f32).to_le_bytes().to_vec();
            bytes.extend(0f32.to_le_bytes());
            bytes
        }
        "complex128" => {
            let mut bytes = value.to_le_bytes().to_vec();
            bytes.extend(0f64.to_le_bytes());
            bytes
        }
        other => {
            return Err(Error::transform(format!(
                "cannot build a scalar constant of unsupported dtype '{other}'"
            )))
        }
    };
    Ok(ArrayPayload {
        dtype,
        shape: Vec::new(),
        data: bytes,
    })
}

/// IEEE-754 binary16 bits for an `f32` (round-to-nearest-even).
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let mut exponent16 = exponent - 127 + 15;
    if exponent16 >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent16 <= 0 {
        if exponent16 < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - exponent16) as u32;
        let mut half = (mantissa >> shift) as u16;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    exponent16 = exponent16.max(0);
    let mut half = ((exponent16 as u16) << 10) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    sign | half
}

/// An int64 vector payload (gather / arange indices).
pub fn i64_vector_payload(values: &[i64]) -> ArrayPayload {
    let mut data = Vec::with_capacity(values.len() * 8);
    for value in values {
        data.extend(value.to_le_bytes());
    }
    ArrayPayload {
        dtype: DType::parse("int64").expect("int64 is a registered dtype"),
        shape: vec![values.len() as i64],
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_matrix(rows: i64, cols: i64) -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            [Dim::Int(rows), Dim::Int(cols)],
        )
    }

    #[test]
    fn scalar_payloads_cover_every_dtype() {
        for (name, len) in [
            ("bool", 1usize),
            ("int8", 1),
            ("int16", 2),
            ("int32", 4),
            ("int64", 8),
            ("uint8", 1),
            ("uint16", 2),
            ("uint32", 4),
            ("uint64", 8),
            ("float16", 2),
            ("float32", 4),
            ("float64", 8),
            ("complex64", 8),
            ("complex128", 16),
        ] {
            let payload = scalar_payload(DType::parse(name).unwrap(), 1.0).unwrap();
            assert_eq!(payload.data.len(), len, "{name}");
            assert!(payload.shape.is_empty());
            assert_eq!(payload.element_count().unwrap(), 1);
        }
        let one = scalar_payload(DType::parse("float16").unwrap(), 1.0).unwrap();
        assert_eq!(u16::from_le_bytes([one.data[0], one.data[1]]), 0x3c00);
    }

    #[test]
    fn f16_rounding_is_nearest_even() {
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);
        // 0.1 is not exact; it must round to the nearest f16 value.
        assert_eq!(f32_to_f16_bits(0.1), 0x2e66);
    }

    #[test]
    fn helpers_emit_expected_shapes() {
        let mut builder = crate::Builder::new("main");
        builder
            .open_function("main", &[f32_matrix(2, 3)])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let reshaped = ctx.reshape(args[0], &dims_i64(&[6])).unwrap();
        assert_eq!(ctx.rank(reshaped).unwrap(), 1);
        let transposed = ctx.transpose(reshaped, Some(&[0])).unwrap();
        assert_eq!(ctx.rank(transposed).unwrap(), 1);
        let reduced = ctx.reduce_sum(args[0], &[1], false).unwrap();
        assert_eq!(ctx.shape(reduced).unwrap(), vec![Dim::Int(2)]);
        let mean = ctx.reduce_mean(args[0], &[0, 1], true).unwrap();
        assert_eq!(ctx.rank(mean).unwrap(), 2);
        let padded = ctx.pad(args[0], &[(1, 0), (0, 2)], 0.0).unwrap();
        assert_eq!(
            ctx.shape(padded).unwrap(),
            vec![Dim::Int(3), Dim::Int(5)]
        );
        let idx = ctx.arange_i64(0, 3, 1).unwrap();
        assert_eq!(ctx.shape(idx).unwrap(), vec![Dim::Int(3)]);
        let gathered = ctx.gather(args[0], idx, &[1]).unwrap();
        assert_eq!(ctx.shape(gathered).unwrap(), vec![Dim::Int(2), Dim::Int(3)]);
        let one = ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap();
        assert_eq!(ctx.rank(one).unwrap(), 0);
        let condition = ctx.scalar(DType::parse("bool").unwrap(), 1.0).unwrap();
        let picked = ctx.select(condition, args[0], args[0]).unwrap();
        assert_eq!(ctx.shape(picked).unwrap(), vec![Dim::Int(2), Dim::Int(3)]);
        ctx.argmax(args[0], Some(1), false).unwrap();
        ctx.argmin(args[0], None, true).unwrap();
        ctx.slice(args[0], &[0, 1], &[2, 3], Some(&[1, 1])).unwrap();
        ctx.concatenate(&[args[0], args[0]], 1).unwrap();
        let _ = ctx.scatter(args[0], idx, args[0], 0).is_err();
        ctx.cumsum(args[0], 1, false).unwrap();
        ctx.cumprod(args[0], 1, true).unwrap();
        ctx.tril(args[0], 0).unwrap();
        ctx.triu(args[0], 0).unwrap();
        let _ = ctx.solve(args[0], args[0]).is_err();
    }

    #[test]
    fn rule_op_accessors_and_conv_params() {
        let mut builder = crate::Builder::new("rule");
        builder
            .open_function(
                "rule",
                &[
                    ValueType::of(
                        DType::parse("float32").unwrap(),
                        [Dim::Int(1), Dim::Int(4), Dim::Int(8), Dim::Int(8)],
                    ),
                    ValueType::of(
                        DType::parse("float32").unwrap(),
                        [Dim::Int(1), Dim::Int(4), Dim::Int(3), Dim::Int(3)],
                    ),
                ],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let out = builder.emit_typed_one("conv", &args, &Attrs::new()).unwrap();
        let op_id = *builder.block_op_ids().unwrap().last().unwrap();
        assert_eq!(builder.value_type(out).unwrap().rank(), 4);
        let mut rule_op = {
            let op = builder.program().op(op_id).unwrap();
            RuleOp::from_source(builder.program(), op).unwrap()
        };
        rule_op.operands = args.clone();
        rule_op.results = vec![out];
        let params = ConvParams::from_op(&rule_op).unwrap();
        assert_eq!(params.batch_group_count, 1);
        assert_eq!(params.feature_group_count, 1);
        assert!(params.strides.is_none());
        assert_eq!(params.padding, AttrValue::Str("VALID".to_string()));
        assert_eq!(rule_op.rule_key(), "conv");
        assert_eq!(rule_op.original_operand(0).unwrap().rank(), 4);
        assert!(rule_op.original_operand(2).is_err());
        let mut attrs = rule_op.attributes.clone();
        attrs.insert("batch_group_count", AttrValue::Int(2));
        rule_op.attributes = attrs;
        let err = ConvParams::from_op(&rule_op).unwrap_err();
        assert!(err.message.contains("batch-grouped conv"));
        assert_eq!(params.to_attrs().i64("feature_group_count").unwrap(), 1);
    }
}
