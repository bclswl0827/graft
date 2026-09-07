# WASM runtime build

The runtime is a Rust `cdylib` compiled for `wasm32-wasip1-threads`. The
resulting module uses WebAssembly SIMD, shared memory, atomics, and WASI
threads so tract can execute eligible inference kernels in parallel.

## Requirements

- A stable Rust toolchain with `rustup`
- GNU Make and `install`

Install the required Rust target:

```bash
rustup target add wasm32-wasip1-threads
```

## Build

From the repository root, run:

```bash
make wasm
```

This command builds:

```text
runtime/target/wasm32-wasip1-threads/release/graft_runtime.wasm
```

and copy it to `internal/wasm/runtime.wasm`, where Go embeds it into the
library.

To build the Rust artifact without replacing the embedded copy:

```bash
cargo build --manifest-path runtime/Cargo.toml \
  --config 'target.wasm32-wasip1-threads.rustflags=["-C","target-feature=+simd128"]' \
  --target wasm32-wasip1-threads \
  --release
```

The Makefile passes the same target-specific Cargo configuration automatically.
Keep the `--config` argument when invoking Cargo directly; tract selects its
WebAssembly vector kernels only when `simd128` is enabled at compile time.
`build.rs` also links `crt1-reactor.o`, which initializes the main WASI thread
before the runtime uses allocation, thread-local storage, or synchronization.
