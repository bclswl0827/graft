package main

import (
	_ "embed"
	"fmt"
	"os"

	"github.com/bclswl0827/graft/onnx"
)

//go:embed temperature.onnx
var modelData []byte

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	engine, err := onnx.NewEngine(onnx.WithNumThreads(1))
	if err != nil {
		return fmt.Errorf("create engine: %w", err)
	}
	defer engine.Close()

	model, err := engine.LoadModel(modelData)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer model.Close()

	celsius := []float32{0, 10, 20, 100}
	input, err := onnx.NewTensor([]int64{4}, celsius)
	if err != nil {
		return fmt.Errorf("create input: %w", err)
	}

	outputs, err := model.RunNamed(map[string]onnx.Tensor{"celsius": input})
	if err != nil {
		return fmt.Errorf("run model: %w", err)
	}
	output, ok := outputs["fahrenheit"]
	if !ok {
		return fmt.Errorf("model did not return fahrenheit output")
	}
	fahrenheit, ok := output.Float32Data()
	if !ok || len(fahrenheit) != len(celsius) {
		return fmt.Errorf("unexpected output: type=%s shape=%v", output.DataType(), output.Shape())
	}

	for index := range celsius {
		fmt.Printf("%.0f°C = %.0f°F\n", celsius[index], fahrenheit[index])
	}
	return nil
}
