use tract_onnx::prelude::*;

use crate::error::{ErrorCode, Result, RuntimeError};

pub const TENSOR_MAGIC: u32 = u32::from_le_bytes(*b"ONXR");
pub const PROTOCOL_VERSION: u16 = 1;
#[cfg(test)]
const HEADER_SIZE: usize = 20;
#[cfg(test)]
const TENSOR_HEADER_SIZE: usize = 16;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataType {
    Float32 = 1,
    Float64 = 2,
    Int32 = 3,
    Int64 = 4,
    Uint8 = 5,
    Bool = 6,
}

impl TryFrom<u8> for DataType {
    type Error = RuntimeError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Float32),
            2 => Ok(Self::Float64),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Int64),
            5 => Ok(Self::Uint8),
            6 => Ok(Self::Bool),
            _ => Err(RuntimeError::new(
                ErrorCode::UnsupportedDataType,
                format!("unsupported tensor data type {value}"),
            )),
        }
    }
}

impl DataType {
    pub fn size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
            Self::Uint8 | Self::Bool => 1,
        }
    }

    pub fn from_datum_type(value: DatumType) -> Result<Self> {
        match value {
            DatumType::F32 => Ok(Self::Float32),
            DatumType::F64 => Ok(Self::Float64),
            DatumType::I32 => Ok(Self::Int32),
            DatumType::I64 => Ok(Self::Int64),
            DatumType::U8 => Ok(Self::Uint8),
            DatumType::Bool => Ok(Self::Bool),
            _ => Err(RuntimeError::new(
                ErrorCode::UnsupportedDataType,
                format!("tract data type {value:?} is not supported by ABI v2"),
            )),
        }
    }

    fn datum_type(self) -> DatumType {
        match self {
            Self::Float32 => DatumType::F32,
            Self::Float64 => DatumType::F64,
            Self::Int32 => DatumType::I32,
            Self::Int64 => DatumType::I64,
            Self::Uint8 => DatumType::U8,
            Self::Bool => DatumType::Bool,
        }
    }
}

#[derive(Debug)]
pub struct TensorData<'a> {
    pub data_type: DataType,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

impl TensorData<'_> {
    pub fn into_tensor(self) -> Result<Tensor> {
        if self.data_type == DataType::Bool {
            let values: Vec<bool> = self.bytes.iter().map(|value| *value != 0).collect();
            return Tensor::from_shape(&self.shape, &values).map_err(|error| {
                RuntimeError::new(
                    ErrorCode::InvalidTensor,
                    format!("creating tensor: {error:#}"),
                )
            });
        }

        // wasm32 is little-endian, as is the ABI. All bit patterns of the
        // supported numeric datum types are valid, and decode_request already
        // checked the exact byte length. Bool is handled above because arbitrary
        // bytes are not valid Rust bool representations.
        unsafe { Tensor::from_raw_dt(self.data_type.datum_type(), &self.shape, self.bytes) }
            .map_err(|error| {
                RuntimeError::new(
                    ErrorCode::InvalidTensor,
                    format!("creating tensor: {error:#}"),
                )
            })
    }
}

#[derive(Debug)]
pub struct TensorRequest<'a> {
    pub tensors: Vec<TensorData<'a>>,
    pub max_output_bytes: u64,
}

pub fn decode_request(bytes: &[u8]) -> Result<TensorRequest<'_>> {
    let mut decoder = Decoder::new(bytes);
    if decoder.u32()? != TENSOR_MAGIC {
        return Err(RuntimeError::new(
            ErrorCode::Serialization,
            "invalid tensor protocol magic",
        ));
    }
    if decoder.u16()? != PROTOCOL_VERSION {
        return Err(RuntimeError::new(
            ErrorCode::Serialization,
            "unsupported tensor protocol version",
        ));
    }
    decoder.skip(2)?;
    let count = decoder.u32()? as usize;
    let max_output_bytes = decoder.u64()?;
    if count > 1024 {
        return Err(RuntimeError::new(
            ErrorCode::ResourceLimit,
            "tensor count exceeds 1024",
        ));
    }

    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let data_type = DataType::try_from(decoder.u8()?)?;
        decoder.skip(3)?;
        let rank = decoder.u32()? as usize;
        let byte_length = usize::try_from(decoder.u64()?).map_err(|_| {
            RuntimeError::new(
                ErrorCode::Serialization,
                "tensor byte length overflows usize",
            )
        })?;
        if rank > 64 {
            return Err(RuntimeError::new(
                ErrorCode::ResourceLimit,
                "tensor rank exceeds 64",
            ));
        }
        let mut shape = Vec::with_capacity(rank);
        let mut elements = 1_usize;
        for _ in 0..rank {
            let dimension = decoder.i64()?;
            if dimension < 0 {
                return Err(RuntimeError::new(
                    ErrorCode::InvalidTensor,
                    "negative tensor dimension",
                ));
            }
            let dimension = usize::try_from(dimension).map_err(|_| {
                RuntimeError::new(ErrorCode::InvalidTensor, "tensor dimension overflows usize")
            })?;
            elements = elements.checked_mul(dimension).ok_or_else(|| {
                RuntimeError::new(
                    ErrorCode::InvalidTensor,
                    "tensor element count overflows usize",
                )
            })?;
            shape.push(dimension);
        }
        let expected = elements.checked_mul(data_type.size()).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::InvalidTensor,
                "tensor byte length overflows usize",
            )
        })?;
        if expected != byte_length {
            return Err(RuntimeError::new(
                ErrorCode::InvalidTensor,
                format!("tensor data has {byte_length} bytes, expected {expected}"),
            ));
        }
        let data = decoder.bytes(byte_length)?;
        tensors.push(TensorData {
            data_type,
            shape,
            bytes: data,
        });
    }
    if !decoder.is_finished() {
        return Err(RuntimeError::new(
            ErrorCode::Serialization,
            "trailing tensor protocol data",
        ));
    }
    Ok(TensorRequest {
        tensors,
        max_output_bytes,
    })
}

pub fn encode_response(outputs: &[TValue], max_output_bytes: u64) -> Result<Vec<u8>> {
    let count = u32::try_from(outputs.len())
        .map_err(|_| RuntimeError::new(ErrorCode::ResourceLimit, "too many output tensors"))?;
    let mut encoder = Encoder::with_limit(max_output_bytes);
    encoder.u32(TENSOR_MAGIC)?;
    encoder.u16(PROTOCOL_VERSION)?;
    encoder.u16(0)?;
    encoder.u32(count)?;
    encoder.u64(0)?;
    for output in outputs {
        let tensor: &Tensor = output;
        let data_type = DataType::from_datum_type(tensor.datum_type())?;
        encoder.u8(data_type as u8)?;
        encoder.bytes(&[0; 3])?;
        encoder.u32(u32::try_from(tensor.rank()).map_err(|_| {
            RuntimeError::new(ErrorCode::ResourceLimit, "output tensor rank exceeds u32")
        })?)?;
        let byte_length = tensor.len().checked_mul(data_type.size()).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::ResourceLimit,
                "output tensor byte length overflows",
            )
        })?;
        encoder.u64(byte_length as u64)?;
        for &dimension in tensor.shape() {
            encoder.i64(i64::try_from(dimension).map_err(|_| {
                RuntimeError::new(ErrorCode::ResourceLimit, "output dimension exceeds i64")
            })?)?;
        }
        let raw = tensor.as_bytes();
        if raw.len() != byte_length {
            return Err(RuntimeError::new(
                ErrorCode::Inference,
                "output tensor storage length does not match its shape",
            ));
        }
        encoder.bytes(raw)?;
    }
    Ok(encoder.finish())
}

pub struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    pub fn bytes(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self.offset.checked_add(length).ok_or_else(|| {
            RuntimeError::new(ErrorCode::Serialization, "binary protocol offset overflow")
        })?;
        let value = self.bytes.get(self.offset..end).ok_or_else(|| {
            RuntimeError::new(ErrorCode::Serialization, "truncated binary protocol")
        })?;
        self.offset = end;
        Ok(value)
    }

    pub fn skip(&mut self, length: usize) -> Result<()> {
        self.bytes(length).map(|_| ())
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        let bytes: [u8; 2] = self
            .bytes(2)?
            .try_into()
            .map_err(|_| RuntimeError::new(ErrorCode::Serialization, "invalid u16 field"))?;
        Ok(u16::from_le_bytes(bytes))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self
            .bytes(4)?
            .try_into()
            .map_err(|_| RuntimeError::new(ErrorCode::Serialization, "invalid u32 field"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let bytes: [u8; 8] = self
            .bytes(8)?
            .try_into()
            .map_err(|_| RuntimeError::new(ErrorCode::Serialization, "invalid u64 field"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn i64(&mut self) -> Result<i64> {
        let bytes: [u8; 8] = self
            .bytes(8)?
            .try_into()
            .map_err(|_| RuntimeError::new(ErrorCode::Serialization, "invalid i64 field"))?;
        Ok(i64::from_le_bytes(bytes))
    }
}

struct Encoder {
    bytes: Vec<u8>,
    limit: usize,
}

impl Encoder {
    fn with_limit(limit: u64) -> Self {
        Self {
            bytes: Vec::new(),
            limit: usize::try_from(limit).unwrap_or(usize::MAX),
        }
    }

    fn ensure(&self, additional: usize) -> Result<()> {
        let length = self.bytes.len().checked_add(additional).ok_or_else(|| {
            RuntimeError::new(ErrorCode::ResourceLimit, "response size overflows usize")
        })?;
        if length > self.limit {
            return Err(RuntimeError::new(
                ErrorCode::ResourceLimit,
                format!(
                    "response would exceed the {0}-byte output limit",
                    self.limit
                ),
            ));
        }
        Ok(())
    }

    fn bytes(&mut self, value: &[u8]) -> Result<()> {
        self.ensure(value.len())?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<()> {
        self.bytes(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn i64(&mut self, value: i64) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_request() {
        let error = decode_request(b"ONXR").expect_err("request must be rejected");
        assert_eq!(error.code, ErrorCode::Serialization);
    }

    #[test]
    fn rejects_bad_tensor_length() {
        let mut request = Vec::new();
        request.extend_from_slice(&TENSOR_MAGIC.to_le_bytes());
        request.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        request.extend_from_slice(&0_u16.to_le_bytes());
        request.extend_from_slice(&1_u32.to_le_bytes());
        request.extend_from_slice(&1024_u64.to_le_bytes());
        request.push(DataType::Float32 as u8);
        request.extend_from_slice(&[0; 3]);
        request.extend_from_slice(&1_u32.to_le_bytes());
        request.extend_from_slice(&3_u64.to_le_bytes());
        request.extend_from_slice(&1_i64.to_le_bytes());
        request.extend_from_slice(&[0; 3]);
        let error = decode_request(&request).expect_err("request must be rejected");
        assert_eq!(error.code, ErrorCode::InvalidTensor);
    }

    #[test]
    fn protocol_constants_have_expected_sizes() {
        assert_eq!(HEADER_SIZE, 20);
        assert_eq!(TENSOR_HEADER_SIZE, 16);
    }
}
