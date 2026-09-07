RUST_TARGET ?= wasm32-wasip1-threads
RUNTIME_WASM := runtime/target/$(RUST_TARGET)/release/graft_runtime.wasm
EMBEDDED_WASM := internal/wasm/runtime.wasm
WASM_CARGO_CONFIG := target.$(RUST_TARGET).rustflags=["-C","target-feature=+simd128"]

.PHONY: wasm test test-rust test-go benchmark build

wasm:
	cargo --config '$(WASM_CARGO_CONFIG)' build --manifest-path runtime/Cargo.toml --release --target $(RUST_TARGET)
	install -m 0644 $(RUNTIME_WASM) $(EMBEDDED_WASM)

test: test-rust test-go

test-rust:
	cargo fmt --manifest-path runtime/Cargo.toml --check
	cargo clippy --manifest-path runtime/Cargo.toml --all-targets -- -D warnings
	cargo test --manifest-path runtime/Cargo.toml

test-go:
	CGO_ENABLED=0 go test ./...

benchmark:
	CGO_ENABLED=0 go test -bench=. -benchmem ./onnx

build: wasm
	CGO_ENABLED=0 go build ./...
