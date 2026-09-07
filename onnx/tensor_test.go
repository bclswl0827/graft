package onnx

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestTensorValidation(t *testing.T) {
	tests := []struct {
		name  string
		shape []int64
		data  []float32
	}{
		{name: "negative dimension", shape: []int64{-1}, data: nil},
		{name: "wrong element count", shape: []int64{2, 2}, data: []float32{1}},
		{name: "nil nonempty data", shape: []int64{1}, data: nil},
		{name: "overflow", shape: []int64{math.MaxInt64, 3}, data: nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewTensor(test.shape, test.data)
			if !errors.Is(err, ErrInvalidTensor) {
				t.Fatalf("NewTensor error = %v, want ErrInvalidTensor", err)
			}
		})
	}
}

func TestZeroLengthTensor(t *testing.T) {
	tensor, err := NewTensor([]int64{2, 0, 3}, nil)
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	if got := tensor.Shape(); !reflect.DeepEqual(got, []int64{2, 0, 3}) {
		t.Fatalf("Shape = %v", got)
	}
}

func TestTensorCopiesInputAndOutput(t *testing.T) {
	shape := []int64{2}
	data := []float32{1, 2}
	tensor, err := NewTensor(shape, data)
	if err != nil {
		t.Fatalf("NewTensor: %v", err)
	}
	shape[0] = 99
	data[0] = 99
	got, ok := tensor.Float32Data()
	if !ok || !reflect.DeepEqual(got, []float32{1, 2}) {
		t.Fatalf("Float32Data = %v, %v", got, ok)
	}
	got[0] = 42
	again, _ := tensor.Float32Data()
	if again[0] != 1 {
		t.Fatal("Float32Data exposed mutable tensor storage")
	}
}

func TestDecodeResponseRejectsMalformed(t *testing.T) {
	for _, data := range [][]byte{nil, []byte("ONXR"), make([]byte, 20)} {
		if _, err := decodeResponse(data, 1024, 1024); err == nil {
			t.Fatalf("decodeResponse(%x) unexpectedly succeeded", data)
		}
	}
}

func TestProtocolResourceLimits(t *testing.T) {
	tensor, err := NewTensor([]int64{2, 3}, make([]float32, 6))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := encodeRequest([]Tensor{tensor}, 23, 1024); !errors.Is(err, ErrResourceLimit) {
		t.Fatalf("encodeRequest error = %v", err)
	}
	if _, err := encodeModelPackage([]byte{1}, map[string][]byte{"weights.bin": make([]byte, 10)}, 20); !errors.Is(err, ErrResourceLimit) {
		t.Fatalf("encodeModelPackage error = %v", err)
	}
	if _, err := encodeModelPackage([]byte{1}, map[string][]byte{"../weights.bin": nil}, 1024); !errors.Is(err, ErrInvalidModel) {
		t.Fatalf("invalid external path error = %v", err)
	}
}

func TestEngineOptionValidation(t *testing.T) {
	for _, option := range []Option{
		WithNumThreads(0),
		WithNumThreads(maxWASMThreads + 1),
		WithMaxModelBytes(0),
		WithMaxTensorBytes(math.MaxUint32 + 1),
		WithMaxOutputBytes(0),
	} {
		if _, err := NewEngine(option); err == nil {
			t.Fatal("NewEngine unexpectedly accepted an invalid limit")
		}
	}
}
