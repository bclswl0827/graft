use std::collections::{HashMap, HashSet};
use std::path::{Component, Path};
use std::sync::Arc;

use anyhow::{Context, ensure, format_err};
use prost::Message;
use tract_linalg::multithread::Executor;
use tract_onnx::data_resolver::ModelDataResolver;
use tract_onnx::pb;
use tract_onnx::prelude::*;

use crate::error::{ErrorCode, Result, RuntimeError};
use crate::tensor::{DataType, Decoder};

const PACKAGE_MAGIC: u32 = u32::from_le_bytes(*b"ONXP");
const PACKAGE_VERSION: u16 = 1;
const METADATA_MAGIC: u32 = u32::from_le_bytes(*b"ONXM");
const METADATA_VERSION: u16 = 1;

type ExternalData = HashMap<String, Vec<u8>>;

pub struct Model {
    runnable: Arc<TypedRunnableModel>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
}

impl Model {
    pub fn load(bytes: &[u8], executor: &Executor) -> Result<Self> {
        Self::load_parts(bytes, HashMap::new(), executor)
    }

    pub fn load_package(bytes: &[u8], executor: &Executor) -> Result<Self> {
        let (model, external_data) = decode_package(bytes)?;
        Self::load_parts(&model, external_data, executor)
    }

    fn load_parts(bytes: &[u8], external_data: ExternalData, executor: &Executor) -> Result<Self> {
        let proto = pb::ModelProto::decode(bytes).map_err(|error| {
            RuntimeError::new(
                ErrorCode::ModelParse,
                format!("decoding ONNX protobuf: {error}"),
            )
        })?;
        let (inputs, outputs) = metadata_from_proto(&proto)?;

        let mut framework = tract_onnx::onnx();
        let has_external_data = !external_data.is_empty();
        if has_external_data {
            framework.provider = Arc::new(MemoryDataResolver {
                files: external_data,
            });
        }
        let parse_result = framework
            .parse(&proto, has_external_data.then_some("."))
            .map_err(|error| {
                RuntimeError::new(
                    ErrorCode::ModelParse,
                    format!("translating ONNX graph: {error:#}"),
                )
            })?;
        if !parse_result.unresolved_inputs.is_empty() {
            return Err(RuntimeError::new(
                ErrorCode::ModelParse,
                format!(
                    "unresolved graph inputs: {:?}",
                    parse_result.unresolved_inputs
                ),
            ));
        }
        let typed = parse_result.model.into_optimized().map_err(|error| {
            RuntimeError::new(
                ErrorCode::ModelOptimization,
                format!("optimizing ONNX graph: {error:#}"),
            )
        })?;
        let options = RunOptions {
            executor: Some(executor.clone()),
            ..RunOptions::default()
        };
        let runnable = TypedSimplePlan::new_with_options(typed, &options).map_err(|error| {
            RuntimeError::new(
                ErrorCode::ModelOptimization,
                format!("creating runnable ONNX graph: {error:#}"),
            )
        })?;
        Ok(Self {
            runnable,
            inputs,
            outputs,
        })
    }

    pub fn run(&self, tensors: Vec<Tensor>) -> Result<TVec<TValue>> {
        if tensors.len() != self.inputs.len() {
            return Err(RuntimeError::new(
                ErrorCode::InvalidTensor,
                format!(
                    "model expects {} inputs, received {}",
                    self.inputs.len(),
                    tensors.len()
                ),
            ));
        }
        let inputs: TVec<TValue> = tensors.into_iter().map(Into::into).collect();
        self.runnable.run(inputs).map_err(|error| {
            RuntimeError::new(
                ErrorCode::Inference,
                format!("running ONNX graph: {error:#}"),
            )
        })
    }

    pub fn encode_metadata(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, METADATA_MAGIC);
        push_u16(&mut bytes, METADATA_VERSION);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, checked_u32(self.inputs.len(), "input count")?);
        push_u32(&mut bytes, checked_u32(self.outputs.len(), "output count")?);
        for value in self.inputs.iter().chain(&self.outputs) {
            value.encode(&mut bytes)?;
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug)]
struct ValueInfo {
    name: String,
    data_type: DataType,
    rank_known: bool,
    dimensions: Vec<Dimension>,
}

impl ValueInfo {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<()> {
        push_u32(bytes, checked_u32(self.name.len(), "value name length")?);
        bytes.extend_from_slice(self.name.as_bytes());
        bytes.push(self.data_type as u8);
        bytes.push(u8::from(self.rank_known));
        push_u16(bytes, 0);
        push_u32(bytes, checked_u32(self.dimensions.len(), "tensor rank")?);
        for dimension in &self.dimensions {
            match dimension {
                Dimension::Known(value) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&[0; 3]);
                    push_i64(bytes, *value);
                    push_u32(bytes, 0);
                }
                Dimension::Symbol(symbol) => {
                    bytes.push(2);
                    bytes.extend_from_slice(&[0; 3]);
                    push_i64(bytes, 0);
                    push_u32(bytes, checked_u32(symbol.len(), "dimension symbol length")?);
                    bytes.extend_from_slice(symbol.as_bytes());
                }
                Dimension::Dynamic => {
                    bytes.push(3);
                    bytes.extend_from_slice(&[0; 3]);
                    push_i64(bytes, 0);
                    push_u32(bytes, 0);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum Dimension {
    Known(i64),
    Symbol(String),
    Dynamic,
}

fn metadata_from_proto(proto: &pb::ModelProto) -> Result<(Vec<ValueInfo>, Vec<ValueInfo>)> {
    let graph = proto.graph.as_ref().ok_or_else(|| {
        RuntimeError::new(ErrorCode::ModelParse, "ONNX model does not contain a graph")
    })?;
    let initializers: HashSet<&str> = graph
        .initializer
        .iter()
        .map(|value| value.name.as_str())
        .collect();
    let inputs = graph
        .input
        .iter()
        .filter(|value| !initializers.contains(value.name.as_str()))
        .map(value_info_from_proto)
        .collect::<Result<Vec<_>>>()?;
    let outputs = graph
        .output
        .iter()
        .map(value_info_from_proto)
        .collect::<Result<Vec<_>>>()?;
    Ok((inputs, outputs))
}

fn value_info_from_proto(value: &pb::ValueInfoProto) -> Result<ValueInfo> {
    let tensor_type = value
        .r#type
        .as_ref()
        .and_then(|value_type| value_type.value.as_ref())
        .map(|value| match value {
            pb::type_proto::Value::TensorType(tensor) => tensor,
        })
        .ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::ModelParse,
                format!("value {:?} is missing a tensor type", value.name),
            )
        })?;
    let data_type = match pb::tensor_proto::DataType::try_from(tensor_type.elem_type) {
        Ok(pb::tensor_proto::DataType::Float) => DataType::Float32,
        Ok(pb::tensor_proto::DataType::Double) => DataType::Float64,
        Ok(pb::tensor_proto::DataType::Int32) => DataType::Int32,
        Ok(pb::tensor_proto::DataType::Int64) => DataType::Int64,
        Ok(pb::tensor_proto::DataType::Uint8) => DataType::Uint8,
        Ok(pb::tensor_proto::DataType::Bool) => DataType::Bool,
        Ok(other) => {
            return Err(RuntimeError::new(
                ErrorCode::UnsupportedDataType,
                format!(
                    "value {:?} uses unsupported ONNX data type {other:?}",
                    value.name
                ),
            ));
        }
        Err(_) => {
            return Err(RuntimeError::new(
                ErrorCode::UnsupportedDataType,
                format!(
                    "value {:?} uses unknown ONNX data type {}",
                    value.name, tensor_type.elem_type
                ),
            ));
        }
    };
    let (rank_known, dimensions) = match &tensor_type.shape {
        Some(shape) => {
            let dimensions = shape
                .dim
                .iter()
                .map(|dimension| match &dimension.value {
                    Some(pb::tensor_shape_proto::dimension::Value::DimValue(value))
                        if *value >= 0 =>
                    {
                        Dimension::Known(*value)
                    }
                    Some(pb::tensor_shape_proto::dimension::Value::DimParam(symbol))
                        if !symbol.is_empty() =>
                    {
                        Dimension::Symbol(symbol.clone())
                    }
                    _ => Dimension::Dynamic,
                })
                .collect();
            (true, dimensions)
        }
        None => (false, Vec::new()),
    };
    Ok(ValueInfo {
        name: value.name.clone(),
        data_type,
        rank_known,
        dimensions,
    })
}

fn decode_package(bytes: &[u8]) -> Result<(Vec<u8>, ExternalData)> {
    let mut decoder = Decoder::new(bytes);
    if decoder.u32()? != PACKAGE_MAGIC {
        return Err(RuntimeError::new(
            ErrorCode::Serialization,
            "invalid model package magic",
        ));
    }
    if decoder.u16()? != PACKAGE_VERSION {
        return Err(RuntimeError::new(
            ErrorCode::Serialization,
            "unsupported model package version",
        ));
    }
    decoder.skip(2)?;
    let model_length = usize::try_from(decoder.u64()?)
        .map_err(|_| RuntimeError::new(ErrorCode::Serialization, "model length overflows usize"))?;
    let external_count = decoder.u32()? as usize;
    decoder.skip(4)?;
    if external_count > 4096 {
        return Err(RuntimeError::new(
            ErrorCode::ResourceLimit,
            "external data file count exceeds 4096",
        ));
    }
    let model = decoder.bytes(model_length)?.to_vec();
    let mut external_data = HashMap::with_capacity(external_count);
    for _ in 0..external_count {
        let name_length = decoder.u32()? as usize;
        if name_length == 0 || name_length > 4096 {
            return Err(RuntimeError::new(
                ErrorCode::Serialization,
                "invalid external data name length",
            ));
        }
        let name = std::str::from_utf8(decoder.bytes(name_length)?)
            .map_err(|_| {
                RuntimeError::new(ErrorCode::Serialization, "external data name is not UTF-8")
            })?
            .to_owned();
        validate_external_name(&name)?;
        let data_length = usize::try_from(decoder.u64()?).map_err(|_| {
            RuntimeError::new(
                ErrorCode::Serialization,
                "external data length overflows usize",
            )
        })?;
        let data = decoder.bytes(data_length)?.to_vec();
        if external_data.insert(name.clone(), data).is_some() {
            return Err(RuntimeError::new(
                ErrorCode::Serialization,
                format!("duplicate external data name {name:?}"),
            ));
        }
    }
    if !decoder.is_finished() {
        return Err(RuntimeError::new(
            ErrorCode::Serialization,
            "trailing model package data",
        ));
    }
    Ok((model, external_data))
}

fn validate_external_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(RuntimeError::new(
            ErrorCode::InvalidArgument,
            format!("external data name must be a relative normalized path, got {name:?}"),
        ));
    }
    Ok(())
}

struct MemoryDataResolver {
    files: HashMap<String, Vec<u8>>,
}

impl ModelDataResolver for MemoryDataResolver {
    fn read_bytes_from_path(
        &self,
        buffer: &mut Vec<u8>,
        path: &Path,
        offset: usize,
        length: Option<usize>,
    ) -> TractResult<()> {
        let normalized = path
            .components()
            .filter_map(|component| match component {
                Component::Normal(value) => value.to_str(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/");
        let data = self
            .files
            .get(&normalized)
            .ok_or_else(|| format_err!("external ONNX data {normalized:?} was not supplied"))?;
        ensure!(
            offset <= data.len(),
            "external data offset is beyond end of file"
        );
        let end = match length {
            Some(length) => offset
                .checked_add(length)
                .context("external data range overflows")?,
            None => data.len(),
        };
        ensure!(
            end <= data.len(),
            "external data range is beyond end of file"
        );
        buffer.extend_from_slice(&data[offset..end]);
        Ok(())
    }
}

fn checked_u32(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| RuntimeError::new(ErrorCode::ResourceLimit, format!("{field} exceeds u32")))
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_parent_external_path() {
        let error = validate_external_name("../weights.bin").expect_err("path must be rejected");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
}
