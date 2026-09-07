// Package wasm owns the low-level wazero module and ONNX runtime ABI.
package wasm

import (
	"context"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/experimental"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

const abiVersion = 2

// A hard per-instance cap prevents an ONNX graph from growing WASM linear
// memory to the WebAssembly 4 GiB maximum. One page is 64 KiB.
const memoryLimitPages = 16_384

//go:embed runtime.wasm
var runtimeWASM []byte

// A threaded WASM module imports env.memory, so separate Engine instances
// cannot share one wazero namespace without also sharing their Rust heaps.
// Share only immutable machine code; each Runtime owns its wazero runtime,
// imported shared memory, root module, and worker modules.
var sharedCompilationCache = wazero.NewCompilationCache()

// RuntimeError is an error reported by the Rust runtime.
type RuntimeError struct {
	Code    int32
	Message string
}

func (e *RuntimeError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("ONNX WASM runtime error %d", e.Code)
	}
	return fmt.Sprintf("ONNX WASM runtime error %d: %s", e.Code, e.Message)
}

// Runtime wraps one WASM instance. Callers must serialize all methods.
type Runtime struct {
	ctx      context.Context
	host     wazero.Runtime
	module   api.Module
	memory   api.Memory
	threads  *threadManager
	shutdown api.Function

	configure    api.Function
	alloc        api.Function
	free         api.Function
	load         api.Function
	loadPackage  api.Function
	unload       api.Function
	run          api.Function
	metadata     api.Function
	lastErrorLen api.Function
	lastError    api.Function
}

// New instantiates an isolated threaded WASI runtime. Machine code is cached
// process-wide, while every instance gets independent shared linear memory.
func New(ctx context.Context, numThreads uint32) (*Runtime, error) {
	if len(runtimeWASM) == 0 {
		return nil, errors.New("embedded ONNX runtime is empty; run make wasm")
	}
	if numThreads == 0 {
		return nil, errors.New("ONNX runtime thread count must be greater than zero")
	}
	config := wazero.NewRuntimeConfig().
		WithCoreFeatures(api.CoreFeaturesV2 | experimental.CoreFeaturesThreads).
		WithMemoryLimitPages(memoryLimitPages).
		WithCompilationCache(sharedCompilationCache)
	host := wazero.NewRuntimeWithConfig(ctx, config)
	var threads *threadManager
	keepHost := false
	defer func() {
		if !keepHost {
			_ = host.Close(ctx)
			if threads != nil {
				_ = threads.close()
			}
		}
	}()

	compiled, err := host.CompileModule(ctx, runtimeWASM)
	if err != nil {
		return nil, fmt.Errorf("compile embedded ONNX runtime: %w", err)
	}
	memoryModule, err := compileSharedMemoryModule(compiled)
	if err != nil {
		return nil, err
	}
	if _, err = host.InstantiateWithConfig(ctx, memoryModule, wazero.NewModuleConfig().WithName("env")); err != nil {
		return nil, fmt.Errorf("instantiate ONNX shared memory: %w", err)
	}
	if _, err = wasi_snapshot_preview1.Instantiate(ctx, host); err != nil {
		return nil, fmt.Errorf("instantiate WASI: %w", err)
	}
	threads = &threadManager{ctx: ctx, host: host, compiled: compiled, nextID: 1}
	if _, err = host.NewHostModuleBuilder("wasi").
		NewFunctionBuilder().
		WithFunc(threads.spawn).
		Export("thread-spawn").
		Instantiate(ctx); err != nil {
		return nil, fmt.Errorf("instantiate WASI threads host: %w", err)
	}
	module, err := host.InstantiateModule(
		ctx,
		compiled,
		wazero.NewModuleConfig().WithName("graft_runtime").WithStartFunctions("_initialize"),
	)
	if err != nil {
		return nil, fmt.Errorf("instantiate embedded ONNX runtime: %w", err)
	}
	r := &Runtime{
		ctx: ctx, host: host, module: module, memory: module.Memory(), threads: threads,
		configure:    module.ExportedFunction("onnx_set_num_threads"),
		shutdown:     module.ExportedFunction("onnx_shutdown"),
		alloc:        module.ExportedFunction("onnx_alloc"),
		free:         module.ExportedFunction("onnx_free"),
		load:         module.ExportedFunction("onnx_model_load"),
		loadPackage:  module.ExportedFunction("onnx_model_load_package"),
		unload:       module.ExportedFunction("onnx_model_unload"),
		run:          module.ExportedFunction("onnx_model_run"),
		metadata:     module.ExportedFunction("onnx_model_metadata"),
		lastErrorLen: module.ExportedFunction("onnx_last_error_len"),
		lastError:    module.ExportedFunction("onnx_last_error"),
	}
	abiFunction := module.ExportedFunction("onnx_abi_version")
	if r.memory == nil {
		return nil, errors.New("embedded ONNX runtime does not export memory")
	}
	for name, function := range map[string]api.Function{
		"onnx_set_num_threads": r.configure, "onnx_shutdown": r.shutdown,
		"onnx_alloc": r.alloc, "onnx_free": r.free, "onnx_model_load": r.load,
		"onnx_model_load_package": r.loadPackage, "onnx_model_unload": r.unload,
		"onnx_model_run": r.run, "onnx_model_metadata": r.metadata,
		"onnx_last_error_len": r.lastErrorLen, "onnx_last_error": r.lastError,
		"onnx_abi_version": abiFunction,
	} {
		if function == nil {
			return nil, fmt.Errorf("embedded ONNX runtime does not export %s", name)
		}
	}
	result, err := abiFunction.Call(ctx)
	if err != nil {
		return nil, fmt.Errorf("query ONNX runtime ABI: %w", err)
	}
	if len(result) != 1 {
		return nil, errors.New("embedded ONNX runtime returned an invalid ABI version result")
	}
	if version := uint32(result[0]); version != abiVersion {
		return nil, fmt.Errorf("unsupported ONNX runtime ABI version %d", version)
	}
	result, err = r.configure.Call(ctx, uint64(numThreads))
	if err != nil {
		return nil, fmt.Errorf("configure ONNX runtime threads: %w", err)
	}
	if err = r.statusError(result); err != nil {
		return nil, err
	}
	keepHost = true
	return r, nil
}

// Close drops the Rust thread pool, joins every worker, and releases the
// isolated wazero runtime. The process-wide compilation cache remains alive.
func (r *Runtime) Close() error {
	if r.host == nil {
		return nil
	}
	var closeErrors []error
	shutdownOK := true
	if r.shutdown != nil {
		result, err := r.shutdown.Call(r.ctx)
		if err != nil {
			shutdownOK = false
			closeErrors = append(closeErrors, fmt.Errorf("shutdown ONNX runtime: %w", err))
		} else if err = r.statusError(result); err != nil {
			shutdownOK = false
			closeErrors = append(closeErrors, err)
		}
	}
	if r.threads != nil {
		r.threads.stop()
	}
	if !shutdownOK {
		if err := r.host.Close(r.ctx); err != nil {
			closeErrors = append(closeErrors, err)
		}
	} else if r.module != nil {
		if err := r.module.Close(r.ctx); err != nil {
			closeErrors = append(closeErrors, err)
		}
	}
	if r.threads != nil {
		if err := r.threads.wait(); err != nil {
			closeErrors = append(closeErrors, err)
		}
	}
	if shutdownOK {
		if err := r.host.Close(r.ctx); err != nil {
			closeErrors = append(closeErrors, err)
		}
	}
	r.host = nil
	r.module = nil
	r.memory = nil
	return errors.Join(closeErrors...)
}

type threadManager struct {
	ctx      context.Context
	host     wazero.Runtime
	compiled wazero.CompiledModule

	mu      sync.Mutex
	nextID  uint32
	closing bool
	err     error
	workers sync.WaitGroup
}

// spawn implements the WASI threads proposal. wasi-libc already allocated and
// initialized the worker's stack/TLS descriptor at startArgument in shared
// memory. A child module provides independent mutable globals and executes that
// descriptor through wasi_thread_start.
func (m *threadManager) spawn(_ context.Context, _ api.Module, startArgument uint32) int32 {
	m.mu.Lock()
	if m.closing || m.nextID > uint32(1<<29-1) {
		m.mu.Unlock()
		return -1
	}
	threadID := m.nextID
	m.nextID++
	m.workers.Add(1)
	m.mu.Unlock()

	child, err := m.host.InstantiateModule(
		m.ctx,
		m.compiled,
		wazero.NewModuleConfig().
			WithName(fmt.Sprintf("graft_worker_%d", threadID)).
			WithStartFunctions(),
	)
	if err != nil {
		m.workers.Done()
		return -1
	}
	start := child.ExportedFunction("wasi_thread_start")
	if start == nil {
		_ = child.Close(m.ctx)
		m.workers.Done()
		return -1
	}
	go func() {
		defer m.workers.Done()
		defer child.Close(m.ctx)
		if _, err := start.Call(m.ctx, uint64(threadID), uint64(startArgument)); err != nil {
			m.mu.Lock()
			if m.err == nil {
				m.err = fmt.Errorf("WASI inference worker %d: %w", threadID, err)
			}
			m.mu.Unlock()
		}
	}()
	return int32(threadID)
}

func (m *threadManager) close() error {
	m.stop()
	return m.wait()
}

func (m *threadManager) stop() {
	m.mu.Lock()
	m.closing = true
	m.mu.Unlock()
}

func (m *threadManager) wait() error {
	m.workers.Wait()
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.err
}

func compileSharedMemoryModule(compiled wazero.CompiledModule) ([]byte, error) {
	memories := compiled.ImportedMemories()
	if len(memories) != 1 {
		return nil, fmt.Errorf("embedded ONNX runtime imports %d memories; expected one", len(memories))
	}
	memory := memories[0]
	module, name, imported := memory.Import()
	maximum, bounded := memory.Max()
	if !imported || module != "env" || name != "memory" || !bounded || memory.Min() > maximum || maximum > memoryLimitPages {
		return nil, fmt.Errorf("embedded ONNX runtime has unsupported shared-memory import %q.%q min=%d max=%d", module, name, memory.Min(), maximum)
	}
	return sharedMemoryModule(memory.Min(), maximum), nil
}

func sharedMemoryModule(minimum, maximum uint32) []byte {
	module := []byte{'\x00', 'a', 's', 'm', '\x01', '\x00', '\x00', '\x00'}
	memorySection := []byte{1, 3} // one memory, min+max+shared flags
	memorySection = appendULEB32(memorySection, minimum)
	memorySection = appendULEB32(memorySection, maximum)
	module = appendSection(module, 5, memorySection)
	exportSection := []byte{1, 6, 'm', 'e', 'm', 'o', 'r', 'y', 2, 0}
	return appendSection(module, 7, exportSection)
}

func appendSection(module []byte, id byte, payload []byte) []byte {
	module = append(module, id)
	module = appendULEB32(module, uint32(len(payload)))
	return append(module, payload...)
}

func appendULEB32(target []byte, value uint32) []byte {
	for {
		current := byte(value & 0x7f)
		value >>= 7
		if value != 0 {
			current |= 0x80
		}
		target = append(target, current)
		if value == 0 {
			return target
		}
	}
}

// LoadModel loads raw ONNX protobuf bytes or an encoded external-data package.
func (r *Runtime) LoadModel(data []byte, packaged bool) (uint32, error) {
	pointer, err := r.writeBuffer(data)
	if err != nil {
		return 0, err
	}
	defer r.freeBuffer(pointer, uint32(len(data)))
	function := r.load
	if packaged {
		function = r.loadPackage
	}
	result, err := function.Call(r.ctx, uint64(pointer), uint64(uint32(len(data))))
	if err != nil {
		return 0, fmt.Errorf("call ONNX model load: %w", err)
	}
	if len(result) != 1 {
		return 0, errors.New("ONNX runtime returned an invalid model handle result")
	}
	handle := uint32(result[0])
	if handle == 0 {
		return 0, r.runtimeError(3)
	}
	return handle, nil
}

// UnloadModel removes a runtime model handle.
func (r *Runtime) UnloadModel(handle uint32) error {
	result, err := r.unload.Call(r.ctx, uint64(handle))
	if err != nil {
		return fmt.Errorf("call ONNX model unload: %w", err)
	}
	return r.statusError(result)
}

// Run executes a serialized inference request and passes a temporary view of
// the response to consume. The view is valid only for the duration of consume.
func (r *Runtime) Run(handle uint32, request []byte, consume func([]byte) error) error {
	requestPointer, err := r.writeBuffer(request)
	if err != nil {
		return err
	}
	defer r.freeBuffer(requestPointer, uint32(len(request)))
	err = r.callWithBuffer(consume, r.run, uint64(handle), uint64(requestPointer), uint64(uint32(len(request))))
	return err
}

// Metadata retrieves serialized model metadata.
func (r *Runtime) Metadata(handle uint32) ([]byte, error) {
	var response []byte
	err := r.callWithBuffer(func(view []byte) error {
		response = append([]byte(nil), view...)
		return nil
	}, r.metadata, uint64(handle))
	return response, err
}

func (r *Runtime) callWithBuffer(consume func([]byte) error, function api.Function, parameters ...uint64) error {
	pointerSlot, err := r.allocate(4)
	if err != nil {
		return err
	}
	defer r.freeBuffer(pointerSlot, 4)
	lengthSlot, err := r.allocate(4)
	if err != nil {
		return err
	}
	defer r.freeBuffer(lengthSlot, 4)
	parameters = append(parameters, uint64(pointerSlot), uint64(lengthSlot))
	result, err := function.Call(r.ctx, parameters...)
	if err != nil {
		return fmt.Errorf("call ONNX runtime: %w", err)
	}
	if err := r.statusError(result); err != nil {
		return err
	}
	pointerBytes, ok := r.memory.Read(pointerSlot, 4)
	if !ok {
		return errors.New("read ONNX response pointer")
	}
	lengthBytes, ok := r.memory.Read(lengthSlot, 4)
	if !ok {
		return errors.New("read ONNX response length")
	}
	pointer := binary.LittleEndian.Uint32(pointerBytes)
	length := binary.LittleEndian.Uint32(lengthBytes)
	if pointer == 0 || length == 0 {
		return errors.New("ONNX runtime returned an empty response")
	}
	defer r.freeBuffer(pointer, length)
	response, ok := r.memory.Read(pointer, length)
	if !ok {
		return errors.New("read ONNX response")
	}
	return consume(response)
}

func (r *Runtime) statusError(result []uint64) error {
	if len(result) != 1 {
		return errors.New("ONNX runtime returned an invalid status result")
	}
	code := int32(uint32(result[0]))
	if code == 0 {
		return nil
	}
	return r.runtimeError(code)
}

func (r *Runtime) runtimeError(code int32) error {
	result, err := r.lastErrorLen.Call(r.ctx)
	if err != nil || len(result) != 1 {
		return &RuntimeError{Code: code}
	}
	length := uint32(result[0])
	if length == 0 {
		return &RuntimeError{Code: code}
	}
	pointer, err := r.allocate(length)
	if err != nil {
		return &RuntimeError{Code: code, Message: "failed to allocate last-error buffer"}
	}
	defer r.freeBuffer(pointer, length)
	result, err = r.lastError.Call(r.ctx, uint64(pointer), uint64(length))
	if err != nil || len(result) != 1 || int32(uint32(result[0])) != 0 {
		return &RuntimeError{Code: code, Message: "failed to read runtime error message"}
	}
	message, ok := r.memory.Read(pointer, length)
	if !ok {
		return &RuntimeError{Code: code, Message: "failed to read runtime error memory"}
	}
	return &RuntimeError{Code: code, Message: string(message)}
}

func (r *Runtime) writeBuffer(data []byte) (uint32, error) {
	if len(data) == 0 || uint64(len(data)) > uint64(^uint32(0)) {
		return 0, errors.New("WASM input buffer size is invalid")
	}
	pointer, err := r.allocate(uint32(len(data)))
	if err != nil {
		return 0, err
	}
	if !r.memory.Write(pointer, data) {
		r.freeBuffer(pointer, uint32(len(data)))
		return 0, errors.New("write WASM input buffer")
	}
	return pointer, nil
}

func (r *Runtime) allocate(size uint32) (uint32, error) {
	result, err := r.alloc.Call(r.ctx, uint64(size))
	if err != nil {
		return 0, fmt.Errorf("allocate WASM memory: %w", err)
	}
	if len(result) != 1 || uint32(result[0]) == 0 {
		return 0, errors.New("allocate WASM memory: runtime rejected allocation")
	}
	return uint32(result[0]), nil
}

func (r *Runtime) freeBuffer(pointer, size uint32) {
	_, _ = r.free.Call(r.ctx, uint64(pointer), uint64(size))
}
