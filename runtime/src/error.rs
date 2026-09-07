use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    Ok = 0,
    InvalidArgument = 1,
    InvalidHandle = 2,
    ModelParse = 3,
    ModelOptimization = 4,
    Inference = 5,
    UnsupportedDataType = 6,
    InvalidTensor = 7,
    BufferTooSmall = 8,
    Serialization = 9,
    Internal = 10,
    ResourceLimit = 11,
}

#[derive(Debug)]
pub struct RuntimeError {
    pub code: ErrorCode,
    pub message: String,
}

impl RuntimeError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeError {}
