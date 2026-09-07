package onnx_test

import (
	"os"

	"github.com/bclswl0827/graft/onnx"
)

func Example() {
	engine, err := onnx.NewEngine()
	if err != nil {
		panic(err)
	}
	defer engine.Close()
	modelData, err := os.ReadFile("../tests/models/identity.onnx")
	if err != nil {
		panic(err)
	}
	model, err := engine.LoadModel(modelData)
	if err != nil {
		panic(err)
	}
	defer model.Close()
	input, err := onnx.NewTensor([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		panic(err)
	}
	_, err = model.Run([]onnx.Tensor{input})
	if err != nil {
		panic(err)
	}
}
