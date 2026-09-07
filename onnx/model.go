package onnx

import "fmt"

// Model is an optimized model retained inside its Engine's WASM instance.
type Model struct {
	engine  *Engine
	handle  uint32
	inputs  []ValueInfo
	outputs []ValueInfo
	closed  bool // guarded by engine.mu
}

// Inputs returns a deep copy of ordered input metadata.
func (m *Model) Inputs() []ValueInfo {
	if m == nil || m.engine == nil {
		return nil
	}
	m.engine.mu.Lock()
	defer m.engine.mu.Unlock()
	return cloneValueInfo(m.inputs)
}

// Outputs returns a deep copy of ordered output metadata.
func (m *Model) Outputs() []ValueInfo {
	if m == nil || m.engine == nil {
		return nil
	}
	m.engine.mu.Lock()
	defer m.engine.mu.Unlock()
	return cloneValueInfo(m.outputs)
}

// Run performs ordered inference. Calls on all models belonging to the same
// Engine are serialized because they share one WASM instance.
func (m *Model) Run(inputs []Tensor) ([]Tensor, error) {
	if m == nil || m.engine == nil {
		return nil, ErrModelClosed
	}
	e := m.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if m.closed {
		return nil, ErrModelClosed
	}
	if e.closed {
		return nil, ErrEngineClosed
	}
	if err := validateValues(inputs, m.inputs, "input"); err != nil {
		return nil, err
	}
	request, err := encodeRequest(inputs, e.options.MaxTensorBytes, e.options.MaxOutputBytes)
	if err != nil {
		return nil, err
	}
	var outputs []Tensor
	err = e.runtime.Run(m.handle, request, func(response []byte) error {
		var decodeError error
		outputs, decodeError = decodeResponse(response, e.options.MaxTensorBytes, e.options.MaxOutputBytes)
		return decodeError
	})
	if err != nil {
		return nil, publicRuntimeError(err)
	}
	if err := validateValues(outputs, m.outputs, "output"); err != nil {
		return nil, fmt.Errorf("runtime returned invalid model output: %w", err)
	}
	return outputs, nil
}

// RunNamed maps input and output names using model metadata.
func (m *Model) RunNamed(inputs map[string]Tensor) (map[string]Tensor, error) {
	if m == nil || m.engine == nil {
		return nil, ErrModelClosed
	}
	metadata := m.Inputs()
	if len(inputs) != len(metadata) {
		return nil, fmt.Errorf("%w: model expects %d named inputs, received %d", ErrInvalidTensor, len(metadata), len(inputs))
	}
	ordered := make([]Tensor, len(metadata))
	for index, value := range metadata {
		tensor, ok := inputs[value.Name]
		if !ok {
			return nil, fmt.Errorf("%w: missing input %q", ErrInvalidTensor, value.Name)
		}
		ordered[index] = tensor
	}
	outputs, err := m.Run(ordered)
	if err != nil {
		return nil, err
	}
	outputMetadata := m.Outputs()
	if len(outputs) != len(outputMetadata) {
		return nil, fmt.Errorf("%w: output metadata count mismatch", ErrInvalidModel)
	}
	result := make(map[string]Tensor, len(outputs))
	for index, tensor := range outputs {
		result[outputMetadata[index].Name] = tensor
	}
	return result, nil
}

// Close unloads the model. It is idempotent.
func (m *Model) Close() error {
	if m == nil || m.engine == nil {
		return nil
	}
	e := m.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	delete(e.models, m)
	if e.closed {
		return nil
	}
	if err := e.runtime.UnloadModel(m.handle); err != nil {
		return publicRuntimeError(err)
	}
	return nil
}

func validateValues(tensors []Tensor, metadata []ValueInfo, label string) error {
	if len(tensors) != len(metadata) {
		return fmt.Errorf("%w: model expects %d %ss, received %d", ErrInvalidTensor, len(metadata), label, len(tensors))
	}
	for index, tensor := range tensors {
		value := metadata[index]
		if tensor.dataType != value.Type {
			return fmt.Errorf("%w: %s %d (%q) expects %s, received %s", ErrInvalidTensor, label, index, value.Name, value.Type, tensor.dataType)
		}
		if !value.RankKnown {
			continue
		}
		if len(tensor.shape) != len(value.Shape) {
			return fmt.Errorf("%w: %s %d (%q) expects rank %d, received %d", ErrInvalidTensor, label, index, value.Name, len(value.Shape), len(tensor.shape))
		}
		for axis, dimension := range value.Shape {
			if dimension.Kind == DimensionKnown && tensor.shape[axis] != dimension.Value {
				return fmt.Errorf("%w: %s %d (%q) axis %d expects %d, received %d", ErrInvalidTensor, label, index, value.Name, axis, dimension.Value, tensor.shape[axis])
			}
		}
	}
	return nil
}
