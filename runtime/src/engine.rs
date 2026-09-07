use std::collections::HashMap;
use std::sync::Arc;

use rayon::ThreadPoolBuilder;
use tract_linalg::multithread::Executor;

use crate::error::{ErrorCode, Result, RuntimeError};
use crate::model::Model;
use crate::tensor::{decode_request, encode_response};

pub struct Engine {
    models: HashMap<u32, Model>,
    next_handle: u32,
    executor: Executor,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_handle: 1,
            executor: Executor::SingleThread,
        }
    }

    pub fn set_num_threads(&mut self, count: u32) -> Result<()> {
        if count == 0 {
            return Err(RuntimeError::new(
                ErrorCode::InvalidArgument,
                "thread count must be greater than zero",
            ));
        }
        if !self.models.is_empty() {
            return Err(RuntimeError::new(
                ErrorCode::InvalidArgument,
                "thread count cannot change after a model is loaded",
            ));
        }
        self.executor = if count == 1 {
            Executor::SingleThread
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(count as usize)
                .thread_name(|index| format!("graft-{index}"))
                .build()
                .map_err(|error| {
                    RuntimeError::new(
                        ErrorCode::Internal,
                        format!("creating inference thread pool: {error}"),
                    )
                })?;
            Executor::MultiThread(Arc::new(pool))
        };
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.models.clear();
        // Dropping the Rayon pool asks every WASI worker to exit and joins it.
        self.executor = Executor::SingleThread;
    }

    pub fn load(&mut self, bytes: &[u8], packaged: bool) -> Result<u32> {
        let handle = self.next_handle;
        if handle == 0 || handle == u32::MAX {
            return Err(RuntimeError::new(
                ErrorCode::ResourceLimit,
                "model handle space exhausted",
            ));
        }
        let model = if packaged {
            Model::load_package(bytes, &self.executor)?
        } else {
            Model::load(bytes, &self.executor)?
        };
        self.next_handle += 1;
        self.models.insert(handle, model);
        Ok(handle)
    }

    pub fn unload(&mut self, handle: u32) -> Result<()> {
        self.models.remove(&handle).map(|_| ()).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::InvalidHandle,
                format!("unknown model handle {handle}"),
            )
        })
    }

    pub fn run(&self, handle: u32, request: &[u8]) -> Result<Vec<u8>> {
        let model = self.models.get(&handle).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::InvalidHandle,
                format!("unknown model handle {handle}"),
            )
        })?;
        let request = decode_request(request)?;
        let tensors = request
            .tensors
            .into_iter()
            .map(|tensor| tensor.into_tensor())
            .collect::<Result<Vec<_>>>()?;
        let outputs = model.run(tensors)?;
        encode_response(&outputs, request.max_output_bytes)
    }

    pub fn metadata(&self, handle: u32) -> Result<Vec<u8>> {
        self.models
            .get(&handle)
            .ok_or_else(|| {
                RuntimeError::new(
                    ErrorCode::InvalidHandle,
                    format!("unknown model handle {handle}"),
                )
            })?
            .encode_metadata()
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_handle() {
        let error = Engine::new()
            .unload(7)
            .expect_err("unknown handle must fail");
        assert_eq!(error.code, ErrorCode::InvalidHandle);
    }

    #[test]
    fn rejects_exhausted_handle_space_before_parsing() {
        let mut engine = Engine::new();
        engine.next_handle = u32::MAX;
        let error = engine
            .load(b"not a model", false)
            .expect_err("handle overflow must fail");
        assert_eq!(error.code, ErrorCode::ResourceLimit);
    }

    #[test]
    fn validates_and_replaces_executor() {
        let mut engine = Engine::new();
        let error = engine
            .set_num_threads(0)
            .expect_err("zero workers must fail");
        assert_eq!(error.code, ErrorCode::InvalidArgument);
        engine.set_num_threads(2).expect("create thread pool");
        engine.shutdown();
        engine.set_num_threads(1).expect("select serial executor");
    }
}
