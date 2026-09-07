package main

import (
	_ "embed"
	"fmt"
	"os"

	"github.com/bclswl0827/graft/onnx"
)

//go:embed iris.onnx
var modelData []byte

var species = [...]string{"setosa", "versicolor", "virginica"}

var samples = [...][4]float32{
	{5.1, 3.5, 1.4, 0.2},
	{5.9, 3.0, 4.2, 1.5},
	{6.5, 3.0, 5.8, 2.2},
}

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

	features := make([]float32, 0, len(samples)*len(samples[0]))
	for _, sample := range samples {
		features = append(features, sample[:]...)
	}
	input, err := onnx.NewTensor([]int64{int64(len(samples)), 4}, features)
	if err != nil {
		return fmt.Errorf("create input: %w", err)
	}

	outputs, err := model.RunNamed(map[string]onnx.Tensor{"features": input})
	if err != nil {
		return fmt.Errorf("run model: %w", err)
	}
	output, ok := outputs["probabilities"]
	if !ok {
		return fmt.Errorf("model did not return probabilities output")
	}
	probabilities, ok := output.Float32Data()
	if !ok || len(probabilities) != len(samples)*len(species) {
		return fmt.Errorf("unexpected output: type=%s shape=%v", output.DataType(), output.Shape())
	}

	for index, sample := range samples {
		row := probabilities[index*len(species) : (index+1)*len(species)]
		class := maximumIndex(row)
		fmt.Printf("%v -> %-10s (%.1f%%)\n", sample, species[class], row[class]*100)
	}
	return nil
}

func maximumIndex(values []float32) int {
	maximum := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[maximum] {
			maximum = index
		}
	}
	return maximum
}
