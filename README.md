# Graft

Portable ONNX inference for Go, powered by
[tract](https://github.com/sonos/tract) in embedded WebAssembly—no CGO or ONNX Runtime.

```text
Go application
	|
onnx package
	|
wazero
	|
tract/WASM
	|
ONNX model
```

Models are loaded at runtime. The embedded CPU runtime is architecture-independent
and supports WebAssembly SIMD and multithreaded execution.

## Install

```bash
go get github.com/bclswl0827/graft/onnx
```

## Usage

```go
engine, err := onnx.NewEngine(onnx.WithNumThreads(4))
if err != nil {
	return err
}
defer engine.Close()

modelData, err := os.ReadFile("model.onnx")
if err != nil {
	return err
}

model, err := engine.LoadModel(modelData)
if err != nil {
	return err
}
defer model.Close()

input, err := onnx.NewTensor([]int64{1, 3, 3001}, inputData)
if err != nil {
	return err
}

outputs, err := model.Run([]onnx.Tensor{input})
```

Use `Model.Inputs`, `Model.Outputs`, and `Model.RunNamed` for name-based model
integration. External-data models are supported through
`Engine.LoadModelWithExternalData`.

## Features

- Pure-Go API and deployment: no CGO, native shared library, or ONNX Runtime.
- Runtime-loaded models with reusable optimized execution plans.
- `float32`, `float64`, `int32`, `int64`, `uint8`, and `bool` tensors.
- Fixed, symbolic, and dynamic shape metadata.
- Configurable worker count and resource limits.
- Goroutine-safe engines, models, and inference calls.

See [`examples/temperature`](examples/temperature),
[`examples/iris`](examples/iris) and [`examples/llm`](examples/llm).

## Limitations

- CPU inference only.
- Operator and opset coverage follows tract and may differ from ONNX Runtime.
- WASM inference may be slower than a native runtime.
- WASM isolation does not prevent excessive CPU or memory use by hostile models;
  load models only from trusted or controlled sources.

## Development

Users of the checked-in runtime only need Go. Rebuilding the WASM runtime
requires stable Rust and the `wasm32-wasip1-threads` target.

```bash
make wasm
make test-rust
make test-go
make build
make benchmark
```
