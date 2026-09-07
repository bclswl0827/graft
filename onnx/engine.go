package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path"
	goruntime "runtime"
	"sort"
	"strings"
	"sync"

	wasmruntime "github.com/bclswl0827/graft/internal/wasm"
)

const defaultLimit = 256 << 20
const maxWASMThreads = 64

// EngineOptions controls inference parallelism and host-side resource limits.
// Limits include serialized tensor bytes but cannot bound all graph CPU
// complexity.
type EngineOptions struct {
	NumThreads     int
	MaxModelBytes  uint64
	MaxTensorBytes uint64
	MaxOutputBytes uint64
}

// DefaultEngineOptions returns the GOMAXPROCS-based worker count and
// conservative per-operation limits.
func DefaultEngineOptions() EngineOptions {
	threads := min(goruntime.GOMAXPROCS(0), maxWASMThreads)
	return EngineOptions{
		NumThreads:    threads,
		MaxModelBytes: defaultLimit, MaxTensorBytes: defaultLimit, MaxOutputBytes: defaultLimit,
	}
}

// Option configures an Engine.
type Option func(*EngineOptions) error

func WithEngineOptions(options EngineOptions) Option {
	return func(target *EngineOptions) error {
		*target = options
		return nil
	}
}

// WithNumThreads sets the number of Rayon workers used inside one inference.
// A value of one forces tract's serial executor.
func WithNumThreads(count int) Option {
	return func(options *EngineOptions) error { options.NumThreads = count; return nil }
}

func WithMaxModelBytes(limit uint64) Option {
	return func(options *EngineOptions) error { options.MaxModelBytes = limit; return nil }
}

func WithMaxTensorBytes(limit uint64) Option {
	return func(options *EngineOptions) error { options.MaxTensorBytes = limit; return nil }
}

func WithMaxOutputBytes(limit uint64) Option {
	return func(options *EngineOptions) error { options.MaxOutputBytes = limit; return nil }
}

// Engine owns an isolated wazero runtime, shared WASM memory, root module, and
// worker modules. All operations are safe for concurrent use and are
// serialized through that instance.
type Engine struct {
	mu      sync.Mutex
	runtime *wasmruntime.Runtime
	options EngineOptions
	models  map[*Model]struct{}
	closed  bool
}

// NewEngine creates and instantiates the embedded ONNX runtime.
func NewEngine(options ...Option) (*Engine, error) {
	config := DefaultEngineOptions()
	for _, option := range options {
		if option == nil {
			return nil, errors.New("onnx: nil engine option")
		}
		if err := option(&config); err != nil {
			return nil, fmt.Errorf("onnx: configure engine: %w", err)
		}
	}
	for name, limit := range map[string]uint64{
		"MaxModelBytes":  config.MaxModelBytes,
		"MaxTensorBytes": config.MaxTensorBytes,
		"MaxOutputBytes": config.MaxOutputBytes,
	} {
		if limit == 0 || limit > math.MaxUint32 {
			return nil, fmt.Errorf("onnx: %s must be between 1 and %d", name, uint64(math.MaxUint32))
		}
	}
	if config.NumThreads < 1 || config.NumThreads > maxWASMThreads {
		return nil, fmt.Errorf("onnx: NumThreads must be between 1 and %d", maxWASMThreads)
	}
	runtime, err := wasmruntime.New(context.Background(), uint32(config.NumThreads))
	if err != nil {
		return nil, fmt.Errorf("onnx: initialize runtime: %w", err)
	}
	return &Engine{runtime: runtime, options: config, models: make(map[*Model]struct{})}, nil
}

// LoadModel parses, optimizes, and retains raw ONNX protobuf bytes.
func (e *Engine) LoadModel(data []byte) (*Model, error) {
	return e.loadModel(data, nil)
}

// LoadModelWithExternalData loads an ONNX protobuf and its external tensor
// files. Keys must match TensorProto external_data locations exactly.
func (e *Engine) LoadModelWithExternalData(data []byte, externalData map[string][]byte) (*Model, error) {
	if len(externalData) == 0 {
		return nil, fmt.Errorf("%w: external data map is empty", ErrInvalidModel)
	}
	return e.loadModel(data, externalData)
}

func (e *Engine) loadModel(data []byte, externalData map[string][]byte) (*Model, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrEngineClosed
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("%w: model data is empty", ErrInvalidModel)
	}
	payload := data
	packaged := externalData != nil
	if packaged {
		var err error
		payload, err = encodeModelPackage(data, externalData, e.options.MaxModelBytes)
		if err != nil {
			return nil, err
		}
	} else if uint64(len(data)) > e.options.MaxModelBytes {
		return nil, fmt.Errorf("%w: model contains %d bytes; limit is %d", ErrResourceLimit, len(data), e.options.MaxModelBytes)
	}
	handle, err := e.runtime.LoadModel(payload, packaged)
	if err != nil {
		return nil, publicRuntimeError(err)
	}
	metadata, err := e.runtime.Metadata(handle)
	if err != nil {
		_ = e.runtime.UnloadModel(handle)
		return nil, publicRuntimeError(err)
	}
	inputs, outputs, err := decodeMetadata(metadata)
	if err != nil {
		_ = e.runtime.UnloadModel(handle)
		return nil, err
	}
	model := &Model{engine: e, handle: handle, inputs: inputs, outputs: outputs}
	e.models[model] = struct{}{}
	return model, nil
}

// Close releases all models and the embedded runtime. It is idempotent.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	for model := range e.models {
		model.closed = true
	}
	e.models = nil
	err := e.runtime.Close()
	e.runtime = nil
	if err != nil {
		return fmt.Errorf("onnx: close runtime: %w", err)
	}
	return nil
}

func encodeModelPackage(model []byte, externalData map[string][]byte, limit uint64) ([]byte, error) {
	names := make([]string, 0, len(externalData))
	total := uint64(24) + uint64(len(model))
	for name, data := range externalData {
		if name == "" || path.IsAbs(name) || path.Clean(name) != name || name == "." || name == ".." || strings.HasPrefix(name, "../") || strings.Contains(name, "\\") {
			return nil, fmt.Errorf("%w: invalid external data name %q", ErrInvalidModel, name)
		}
		if len(name) > 4096 {
			return nil, fmt.Errorf("%w: external data name is too long", ErrResourceLimit)
		}
		addition := uint64(12) + uint64(len(name)) + uint64(len(data))
		if total > math.MaxUint64-addition {
			return nil, fmt.Errorf("%w: model package size overflows", ErrResourceLimit)
		}
		total += addition
		names = append(names, name)
	}
	if len(names) > 4096 || total > limit || total > math.MaxUint32 {
		return nil, fmt.Errorf("%w: model package contains %d bytes; limit is %d", ErrResourceLimit, total, limit)
	}
	sort.Strings(names)
	encoder := binaryEncoder{bytes: make([]byte, 0, int(total))}
	encoder.u32(packageMagic)
	encoder.u16(protocolVer)
	encoder.u16(0)
	encoder.u64(uint64(len(model)))
	encoder.u32(uint32(len(names)))
	encoder.u32(0)
	encoder.bytes = append(encoder.bytes, model...)
	for _, name := range names {
		data := externalData[name]
		encoder.u32(uint32(len(name)))
		encoder.bytes = append(encoder.bytes, name...)
		encoder.u64(uint64(len(data)))
		encoder.bytes = append(encoder.bytes, data...)
	}
	return encoder.bytes, nil
}

func publicRuntimeError(err error) error {
	var runtimeError *wasmruntime.RuntimeError
	if errors.As(err, &runtimeError) {
		return &RuntimeError{Code: ErrorCode(runtimeError.Code), Message: runtimeError.Message}
	}
	return err
}
