package onnx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestEngineIntegration(t *testing.T) {
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Errorf("Engine.Close: %v", err)
		}
	}()

	t.Run("identity and metadata", func(t *testing.T) {
		model := loadFixture(t, engine, "identity.onnx")
		defer model.Close()
		inputs := model.Inputs()
		outputs := model.Outputs()
		if len(inputs) != 1 || inputs[0].Name != "x" || inputs[0].Type != Float32 {
			t.Fatalf("Inputs = %#v", inputs)
		}
		if len(outputs) != 1 || outputs[0].Name != "y" {
			t.Fatalf("Outputs = %#v", outputs)
		}
		input := mustTensor(t, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		actual := runFloat32(t, model, input)
		assertFloat32(t, actual, []float32{1, 2, 3, 4, 5, 6}, 0)
	})

	t.Run("operators", func(t *testing.T) {
		tests := []struct {
			name     string
			shape    []int64
			input    []float32
			expected []float32
		}{
			{name: "add.onnx", shape: []int64{2, 3}, input: []float32{1, 1, 1, 2, 2, 2}, expected: []float32{2, 3, 4, 3, 4, 5}},
			{name: "matmul.onnx", shape: []int64{2, 3}, input: []float32{1, 2, 3, 4, 5, 6}, expected: []float32{22, 28, 49, 64}},
			{name: "conv.onnx", shape: []int64{1, 1, 3, 3}, input: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, expected: []float32{12, 16, 24, 28}},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				model := loadFixture(t, engine, test.name)
				defer model.Close()
				actual := runFloat32(t, model, mustTensor(t, test.shape, test.input))
				assertFloat32(t, actual, test.expected, 1e-6)
			})
		}
	})

	t.Run("multiple inputs and named API", func(t *testing.T) {
		model := loadFixture(t, engine, "multi_input.onnx")
		defer model.Close()
		left := mustTensor(t, []int64{2}, []float32{1, 2})
		right := mustTensor(t, []int64{2}, []float32{3, 4})
		outputs, err := model.RunNamed(map[string]Tensor{"left": left, "right": right})
		if err != nil {
			t.Fatalf("RunNamed: %v", err)
		}
		values, ok := outputs["sum"].Float32Data()
		if !ok {
			t.Fatal("sum is not float32")
		}
		assertFloat32(t, values, []float32{4, 6}, 0)
	})

	t.Run("multiple outputs", func(t *testing.T) {
		model := loadFixture(t, engine, "multi_output.onnx")
		defer model.Close()
		outputs, err := model.Run([]Tensor{mustTensor(t, []int64{2}, []float32{2, -3})})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if len(outputs) != 2 {
			t.Fatalf("output count = %d", len(outputs))
		}
		same, _ := outputs[0].Float32Data()
		negative, _ := outputs[1].Float32Data()
		assertFloat32(t, same, []float32{2, -3}, 0)
		assertFloat32(t, negative, []float32{-2, 3}, 0)
	})

	t.Run("all supported types", func(t *testing.T) {
		model := loadFixture(t, engine, "types.onnx")
		defer model.Close()
		f32, _ := NewTensor([]int64{2}, []float32{1.25, -2.5})
		f64, _ := NewFloat64Tensor([]int64{2}, []float64{math.Pi, -math.E})
		i32, _ := NewInt32Tensor([]int64{2}, []int32{-1, math.MaxInt32})
		i64, _ := NewInt64Tensor([]int64{2}, []int64{math.MinInt64, math.MaxInt64})
		u8, _ := NewUint8Tensor([]int64{2}, []uint8{0, 255})
		boolean, _ := NewBoolTensor([]int64{2}, []bool{true, false})
		inputs := []Tensor{f32, f64, i32, i64, u8, boolean}
		outputs, err := model.Run(inputs)
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if len(outputs) != len(inputs) {
			t.Fatalf("output count = %d", len(outputs))
		}
		for index := range inputs {
			if outputs[index].DataType() != inputs[index].DataType() || !reflect.DeepEqual(outputs[index].Data(), inputs[index].Data()) {
				t.Errorf("output %d = %#v, want %#v", index, outputs[index].Data(), inputs[index].Data())
			}
		}
	})

	t.Run("dynamic shape metadata and inference", func(t *testing.T) {
		model := loadFixture(t, engine, "dynamic_shape.onnx")
		defer model.Close()
		inputs := model.Inputs()
		if len(inputs) != 1 || len(inputs[0].Shape) != 2 || inputs[0].Shape[0].Kind != DimensionSymbolic || inputs[0].Shape[0].Symbol != "batch" {
			t.Fatalf("Inputs = %#v", inputs)
		}
		actual := runFloat32(t, model, mustTensor(t, []int64{1, 3}, []float32{1, 2, 3}))
		assertFloat32(t, actual, []float32{1, 2, 3}, 0)
		actual = runFloat32(t, model, mustTensor(t, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}))
		assertFloat32(t, actual, []float32{1, 2, 3, 4, 5, 6}, 0)
	})

	t.Run("validation and lifecycle", func(t *testing.T) {
		model := loadFixture(t, engine, "identity.onnx")
		wrongType, _ := NewInt64Tensor([]int64{2, 3}, make([]int64, 6))
		if _, err := model.Run([]Tensor{wrongType}); !errors.Is(err, ErrInvalidTensor) {
			t.Fatalf("wrong type error = %v", err)
		}
		wrongShape := mustTensor(t, []int64{1, 6}, make([]float32, 6))
		if _, err := model.Run([]Tensor{wrongShape}); !errors.Is(err, ErrInvalidTensor) {
			t.Fatalf("wrong shape error = %v", err)
		}
		if err := model.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if err := model.Close(); err != nil {
			t.Fatalf("second Close: %v", err)
		}
		if _, err := model.Run(nil); !errors.Is(err, ErrModelClosed) {
			t.Fatalf("Run after Close error = %v", err)
		}
		invalid := []byte("not an ONNX model")
		if _, err := engine.LoadModel(invalid); !errors.Is(err, ErrInvalidModel) {
			t.Fatalf("invalid model error = %v", err)
		}
		first := loadFixture(t, engine, "identity.onnx")
		second := loadFixture(t, engine, "add.onnx")
		if err := first.Close(); err != nil {
			t.Fatalf("close first: %v", err)
		}
		if got := runFloat32(t, second, mustTensor(t, []int64{2, 3}, make([]float32, 6))); len(got) != 6 {
			t.Fatalf("second model output length = %d", len(got))
		}
		if err := second.Close(); err != nil {
			t.Fatalf("close second: %v", err)
		}
	})
}

func TestEngineClose(t *testing.T) {
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	model := loadFixture(t, engine, "identity.onnx")
	if err := engine.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := engine.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := engine.LoadModel([]byte{1}); !errors.Is(err, ErrEngineClosed) {
		t.Fatalf("LoadModel after close error = %v", err)
	}
	if _, err := model.Run(nil); !errors.Is(err, ErrModelClosed) {
		t.Fatalf("Run after engine close error = %v", err)
	}
}

func TestThreadCountsProduceSameOutput(t *testing.T) {
	expected := []float32{22, 28, 49, 64}
	input := mustTensor(t, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	for _, threads := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("threads=%d", threads), func(t *testing.T) {
			engine, err := NewEngine(WithNumThreads(threads))
			if err != nil {
				t.Fatalf("NewEngine: %v", err)
			}
			model := loadFixture(t, engine, "matmul.onnx")
			assertFloat32(t, runFloat32(t, model, input), expected, 0)
			if err := model.Close(); err != nil {
				t.Fatalf("Model.Close: %v", err)
			}
			if err := engine.Close(); err != nil {
				t.Fatalf("Engine.Close: %v", err)
			}
		})
	}
}

func TestEngineInstancesAreIsolated(t *testing.T) {
	identityEngine, err := NewEngine(WithNumThreads(2))
	if err != nil {
		t.Fatalf("NewEngine(identity): %v", err)
	}
	defer identityEngine.Close()
	addEngine, err := NewEngine(WithNumThreads(2))
	if err != nil {
		t.Fatalf("NewEngine(add): %v", err)
	}
	defer addEngine.Close()

	identity := loadFixture(t, identityEngine, "identity.onnx")
	defer identity.Close()
	add := loadFixture(t, addEngine, "add.onnx")
	defer add.Close()
	input := mustTensor(t, []int64{2, 3}, []float32{1, 1, 1, 2, 2, 2})

	type runResult struct {
		values []float32
		err    error
	}
	run := func(model *Model, result chan<- runResult) {
		outputs, err := model.Run([]Tensor{input})
		if err != nil {
			result <- runResult{err: err}
			return
		}
		values, ok := outputs[0].Float32Data()
		if !ok {
			result <- runResult{err: errors.New("output is not float32")}
			return
		}
		result <- runResult{values: values}
	}

	results := make(chan runResult, 2)
	var started sync.WaitGroup
	started.Add(2)
	go func() { defer started.Done(); run(identity, results) }()
	go func() { defer started.Done(); run(add, results) }()
	started.Wait()
	close(results)

	seenIdentity, seenAdd := false, false
	for result := range results {
		if result.err != nil {
			t.Fatalf("concurrent Run: %v", result.err)
		}
		switch {
		case reflect.DeepEqual(result.values, []float32{1, 1, 1, 2, 2, 2}):
			seenIdentity = true
		case reflect.DeepEqual(result.values, []float32{2, 3, 4, 3, 4, 5}):
			seenAdd = true
		default:
			t.Fatalf("unexpected output: %v", result.values)
		}
	}
	if !seenIdentity || !seenAdd {
		t.Fatalf("outputs identity=%v add=%v", seenIdentity, seenAdd)
	}

	if err := identityEngine.Close(); err != nil {
		t.Fatalf("close identity engine: %v", err)
	}
	assertFloat32(t, runFloat32(t, add, input), []float32{2, 3, 4, 3, 4, 5}, 0)
}

func TestPhaseNetReference(t *testing.T) {
	modelPath := filepath.Join("..", "reference", "custom-model", "phasenet_model.onnx")
	externalPath := modelPath + ".data"
	referencePath := filepath.Join("..", "tests", "models", "phasenet_zero_f32.bin")
	modelData, err := os.ReadFile(modelPath)
	if os.IsNotExist(err) {
		t.Skip("PhaseNet reference model is not present")
	}
	if err != nil {
		t.Fatalf("read model: %v", err)
	}
	externalData, err := os.ReadFile(externalPath)
	if err != nil {
		t.Fatalf("read external data: %v", err)
	}
	referenceData, err := os.ReadFile(referencePath)
	if err != nil {
		t.Fatalf("read reference output: %v", err)
	}
	engine, err := NewEngine()
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()
	model, err := engine.LoadModelWithExternalData(modelData, map[string][]byte{
		"phasenet_model.onnx.data": externalData,
	})
	if err != nil {
		t.Fatalf("LoadModelWithExternalData: %v", err)
	}
	defer model.Close()
	inputs, outputs := model.Inputs(), model.Outputs()
	if len(inputs) != 1 || inputs[0].Name != "input" || inputs[0].Type != Float32 {
		t.Fatalf("inputs = %#v", inputs)
	}
	if len(outputs) != 1 || outputs[0].Name != "output" || outputs[0].Type != Float32 {
		t.Fatalf("outputs = %#v", outputs)
	}
	actual := runFloat32(t, model, mustTensor(t, []int64{1, 3, 3001}, make([]float32, 3*3001)))
	if len(referenceData)%4 != 0 || len(actual) != len(referenceData)/4 {
		t.Fatalf("actual length = %d, reference bytes = %d", len(actual), len(referenceData))
	}
	var maximum, total float64
	for index, value := range actual {
		expected := math.Float32frombits(binary.LittleEndian.Uint32(referenceData[index*4:]))
		difference := math.Abs(float64(value - expected))
		tolerance := 1e-4 + 1e-4*math.Abs(float64(expected))
		if difference > maximum {
			maximum = difference
		}
		total += difference
		if difference > tolerance {
			t.Fatalf("value[%d] = %g, reference = %g, difference = %g, tolerance = %g", index, value, expected, difference, tolerance)
		}
	}
	t.Logf("maximum absolute error=%g mean absolute error=%g", maximum, total/float64(len(actual)))
}

func loadFixture(t testing.TB, engine *Engine, name string) *Model {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "tests", "models", name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	model, err := engine.LoadModel(data)
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", name, err)
	}
	return model
}

func mustTensor(t testing.TB, shape []int64, values []float32) Tensor {
	t.Helper()
	tensor, err := NewTensor(shape, values)
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	return tensor
}

func runFloat32(t testing.TB, model *Model, input Tensor) []float32 {
	t.Helper()
	outputs, err := model.Run([]Tensor{input})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("output count = %d", len(outputs))
	}
	values, ok := outputs[0].Float32Data()
	if !ok {
		t.Fatalf("output type = %s", outputs[0].DataType())
	}
	return values
}

func assertFloat32(t testing.TB, actual, expected []float32, tolerance float32) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("length = %d, want %d", len(actual), len(expected))
	}
	for index := range actual {
		if difference := float32(math.Abs(float64(actual[index] - expected[index]))); difference > tolerance {
			t.Errorf("value[%d] = %v, want %v (difference %v)", index, actual[index], expected[index], difference)
		}
	}
}
