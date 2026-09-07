use std::collections::HashMap;

use crate::error::{ErrorCode, Result, RuntimeError};

#[derive(Default)]
pub struct Allocations {
    buffers: HashMap<u32, Vec<u8>>,
}

impl Allocations {
    pub fn allocate(&mut self, size: u32) -> Result<u32> {
        if size == 0 {
            return Err(RuntimeError::new(
                ErrorCode::InvalidArgument,
                "allocation size must be greater than zero",
            ));
        }
        self.insert(vec![0_u8; size as usize])
    }

    pub fn register(&mut self, buffer: Vec<u8>) -> Result<(u32, u32)> {
        let length = u32::try_from(buffer.len()).map_err(|_| {
            RuntimeError::new(
                ErrorCode::ResourceLimit,
                "allocation exceeds the WASM32 address space",
            )
        })?;
        if length == 0 {
            return Err(RuntimeError::new(
                ErrorCode::InvalidArgument,
                "allocation size must be greater than zero",
            ));
        }
        let pointer = self.insert(buffer)?;
        Ok((pointer, length))
    }

    fn insert(&mut self, mut buffer: Vec<u8>) -> Result<u32> {
        let pointer = buffer.as_mut_ptr() as usize;
        let pointer = u32::try_from(pointer).map_err(|_| {
            RuntimeError::new(
                ErrorCode::Internal,
                "allocation pointer exceeds the WASM32 address space",
            )
        })?;
        if pointer == 0 || self.buffers.contains_key(&pointer) {
            return Err(RuntimeError::new(
                ErrorCode::Internal,
                "invalid or duplicate allocation pointer",
            ));
        }
        self.buffers.insert(pointer, buffer);
        Ok(pointer)
    }

    pub fn free(&mut self, pointer: u32, size: u32) -> Result<()> {
        let actual = self
            .buffers
            .get(&pointer)
            .ok_or_else(|| RuntimeError::new(ErrorCode::InvalidArgument, "unknown allocation"))?
            .len();
        if actual != size as usize {
            return Err(RuntimeError::new(
                ErrorCode::InvalidArgument,
                format!("allocation size mismatch: allocated {actual}, freeing {size}"),
            ));
        }
        self.buffers.remove(&pointer);
        Ok(())
    }

    pub fn read(&self, pointer: u32, length: u32) -> Result<&[u8]> {
        let buffer = self.buffers.get(&pointer).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::InvalidArgument,
                "buffer is not runtime allocated",
            )
        })?;
        if length as usize > buffer.len() {
            return Err(RuntimeError::new(
                ErrorCode::BufferTooSmall,
                "read exceeds allocation",
            ));
        }
        Ok(&buffer[..length as usize])
    }

    pub fn write(&mut self, pointer: u32, bytes: &[u8]) -> Result<()> {
        let buffer = self.buffers.get_mut(&pointer).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::InvalidArgument,
                "buffer is not runtime allocated",
            )
        })?;
        if bytes.len() > buffer.len() {
            return Err(RuntimeError::new(
                ErrorCode::BufferTooSmall,
                "write exceeds allocation",
            ));
        }
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn write_u32(&mut self, pointer: u32, value: u32) -> Result<()> {
        self.write(pointer, &value.to_le_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn register_transfers_vec_storage() {
        let mut allocations = Allocations::default();
        let buffer = vec![1_u8, 2, 3, 4];
        let original_pointer = buffer.as_ptr() as u32;
        let (pointer, length) = allocations.register(buffer).expect("register buffer");
        assert_eq!(pointer, original_pointer);
        assert_eq!(length, 4);
        assert_eq!(allocations.read(pointer, length).unwrap(), [1, 2, 3, 4]);
        allocations.free(pointer, length).unwrap();
        assert!(allocations.read(pointer, length).is_err());
    }

    #[test]
    fn register_rejects_empty_buffer() {
        let error = Allocations::default()
            .register(Vec::new())
            .expect_err("empty buffer must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
    }
}
