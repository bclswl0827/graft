use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::engine::Engine;
use crate::error::{ErrorCode, Result, RuntimeError};
use crate::memory::Allocations;

pub const ABI_VERSION: u32 = 2;

#[derive(Default)]
struct State {
    engine: Engine,
    allocations: Allocations,
    last_error: String,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_set_num_threads(count: u32) -> i32 {
    status_call(|state| state.engine.set_num_threads(count))
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_shutdown() -> i32 {
    status_call(|state| {
        state.engine.shutdown();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_alloc(size: u32) -> u32 {
    value_call_preserve_error(0, |state| state.allocations.allocate(size))
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_free(pointer: u32, size: u32) {
    let _ = status_call(|state| state.allocations.free(pointer, size));
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_model_load(model_pointer: u32, model_length: u32) -> u32 {
    value_call(0, |state| {
        let model = state.allocations.read(model_pointer, model_length)?;
        state.engine.load(model, false)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_model_load_package(package_pointer: u32, package_length: u32) -> u32 {
    value_call(0, |state| {
        let package = state.allocations.read(package_pointer, package_length)?;
        state.engine.load(package, true)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_model_unload(model_handle: u32) -> i32 {
    status_call(|state| state.engine.unload(model_handle))
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_model_run(
    model_handle: u32,
    request_pointer: u32,
    request_length: u32,
    response_pointer_pointer: u32,
    response_length_pointer: u32,
) -> i32 {
    status_call(|state| {
        validate_output_slots(state, response_pointer_pointer, response_length_pointer)?;
        let request = state.allocations.read(request_pointer, request_length)?;
        let response = state.engine.run(model_handle, request)?;
        return_buffer(
            state,
            response,
            response_pointer_pointer,
            response_length_pointer,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_model_metadata(
    model_handle: u32,
    response_pointer_pointer: u32,
    response_length_pointer: u32,
) -> i32 {
    status_call(|state| {
        validate_output_slots(state, response_pointer_pointer, response_length_pointer)?;
        let response = state.engine.metadata(model_handle)?;
        return_buffer(
            state,
            response,
            response_pointer_pointer,
            response_length_pointer,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_last_error_len() -> u32 {
    STATE.with(|state| u32::try_from(state.borrow().last_error.len()).unwrap_or(u32::MAX))
}

#[unsafe(no_mangle)]
pub extern "C" fn onnx_last_error(pointer: u32, length: u32) -> i32 {
    status_call_preserve_error(|state| {
        let error = state.last_error.as_bytes();
        if error.len() > length as usize {
            return Err(RuntimeError::new(
                ErrorCode::BufferTooSmall,
                "last-error buffer is too small",
            ));
        }
        let error = error.to_vec();
        state.allocations.write(pointer, &error)
    })
}

fn validate_output_slots(state: &State, pointer_slot: u32, length_slot: u32) -> Result<()> {
    if pointer_slot == length_slot {
        return Err(RuntimeError::new(
            ErrorCode::InvalidArgument,
            "output slots must not alias",
        ));
    }
    state.allocations.read(pointer_slot, 4)?;
    state.allocations.read(length_slot, 4)?;
    Ok(())
}

fn return_buffer(
    state: &mut State,
    response: Vec<u8>,
    pointer_slot: u32,
    length_slot: u32,
) -> Result<()> {
    let (pointer, length) = state.allocations.register(response)?;
    if let Err(error) = state
        .allocations
        .write_u32(pointer_slot, pointer)
        .and_then(|_| state.allocations.write_u32(length_slot, length))
    {
        let _ = state.allocations.free(pointer, length);
        return Err(error);
    }
    Ok(())
}

fn value_call<T: Copy>(failure_value: T, call: impl FnOnce(&mut State) -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.last_error.clear();
            match call(&mut state) {
                Ok(value) => value,
                Err(error) => {
                    state.last_error = error.message;
                    failure_value
                }
            }
        })
    })) {
        Ok(value) => value,
        Err(_) => {
            set_internal_panic();
            failure_value
        }
    }
}

fn value_call_preserve_error<T: Copy>(
    failure_value: T,
    call: impl FnOnce(&mut State) -> Result<T>,
) -> T {
    match catch_unwind(AssertUnwindSafe(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            match call(&mut state) {
                Ok(value) => value,
                Err(_) => failure_value,
            }
        })
    })) {
        Ok(value) => value,
        Err(_) => failure_value,
    }
}

fn status_call(call: impl FnOnce(&mut State) -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            state.last_error.clear();
            match call(&mut state) {
                Ok(()) => ErrorCode::Ok as i32,
                Err(error) => {
                    state.last_error = error.message;
                    error.code as i32
                }
            }
        })
    })) {
        Ok(status) => status,
        Err(_) => {
            set_internal_panic();
            ErrorCode::Internal as i32
        }
    }
}

fn status_call_preserve_error(call: impl FnOnce(&mut State) -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            match call(&mut state) {
                Ok(()) => ErrorCode::Ok as i32,
                Err(error) => error.code as i32,
            }
        })
    })) {
        Ok(status) => status,
        Err(_) => ErrorCode::Internal as i32,
    }
}

fn set_internal_panic() {
    STATE.with(|state| {
        if let Ok(mut state) = state.try_borrow_mut() {
            state.last_error = "internal runtime panic while processing request".to_owned();
        }
    });
}
