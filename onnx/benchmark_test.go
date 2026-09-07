package onnx

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkNewEngineWarm(b *testing.B) {
	for _, threads := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("threads=%d", threads), func(b *testing.B) {
			// Populate the process-wide wazero compilation cache before measuring
			// repeated Engine construction.
			warm, err := NewEngine(WithNumThreads(threads))
			if err != nil {
				b.Fatal(err)
			}
			if err := warm.Close(); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				engine, err := NewEngine(WithNumThreads(threads))
				if err != nil {
					b.Fatal(err)
				}
				if err := engine.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLoadModel(b *testing.B) {
	engine, err := NewEngine()
	if err != nil {
		b.Fatal(err)
	}
	defer engine.Close()
	data, err := os.ReadFile(filepath.Join("..", "tests", "models", "identity.onnx"))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		model, err := engine.LoadModel(data)
		if err != nil {
			b.Fatal(err)
		}
		if err := model.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInferenceSmall(b *testing.B) {
	engine, err := NewEngine()
	if err != nil {
		b.Fatal(err)
	}
	defer engine.Close()
	model := loadFixture(b, engine, "identity.onnx")
	defer model.Close()
	input := mustTensor(b, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := model.Run([]Tensor{input}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTensorProtocol(b *testing.B) {
	input := mustTensor(b, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b.ReportAllocs()
	for range b.N {
		request, err := encodeRequest([]Tensor{input}, 1024, 1024)
		if err != nil || len(request) == 0 {
			b.Fatal(err)
		}
	}
}

func BenchmarkPhaseNetInference(b *testing.B) {
	modelPath := filepath.Join("..", "reference", "custom-model", "phasenet_model.onnx")
	dataPath := modelPath + ".data"
	modelData, err := os.ReadFile(modelPath)
	if os.IsNotExist(err) {
		b.Skip("PhaseNet reference model is not present")
	}
	if err != nil {
		b.Fatal(err)
	}
	externalData, err := os.ReadFile(dataPath)
	if err != nil {
		b.Fatal(err)
	}
	for _, threads := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("threads=%d", threads), func(b *testing.B) {
			engine, err := NewEngine(WithNumThreads(threads))
			if err != nil {
				b.Fatal(err)
			}
			defer engine.Close()
			model, err := engine.LoadModelWithExternalData(modelData, map[string][]byte{"phasenet_model.onnx.data": externalData})
			if err != nil {
				b.Skipf("PhaseNet is not supported by tract: %v", err)
			}
			defer model.Close()
			input := mustTensor(b, []int64{1, 3, 3001}, make([]float32, 3*3001))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := model.Run([]Tensor{input}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
