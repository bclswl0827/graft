package onnx

import (
	"fmt"
	"math"
)

// Tensor is an immutable, typed, row-major tensor. Constructors copy the input
// data, and accessor methods return copies.
type Tensor struct {
	shape    []int64
	dataType DataType
	data     any
}

// NewTensor constructs a float32 tensor.
func NewTensor(shape []int64, data []float32) (Tensor, error) {
	return newTensor(shape, Float32, append([]float32(nil), data...), len(data), data == nil)
}

func NewFloat64Tensor(shape []int64, data []float64) (Tensor, error) {
	return newTensor(shape, Float64, append([]float64(nil), data...), len(data), data == nil)
}

func NewInt32Tensor(shape []int64, data []int32) (Tensor, error) {
	return newTensor(shape, Int32, append([]int32(nil), data...), len(data), data == nil)
}

func NewInt64Tensor(shape []int64, data []int64) (Tensor, error) {
	return newTensor(shape, Int64, append([]int64(nil), data...), len(data), data == nil)
}

func NewUint8Tensor(shape []int64, data []uint8) (Tensor, error) {
	return newTensor(shape, Uint8, append([]uint8(nil), data...), len(data), data == nil)
}

func NewBoolTensor(shape []int64, data []bool) (Tensor, error) {
	return newTensor(shape, Bool, append([]bool(nil), data...), len(data), data == nil)
}

func newTensor(shape []int64, dataType DataType, data any, length int, nilData bool) (Tensor, error) {
	elements, err := elementCount(shape)
	if err != nil {
		return Tensor{}, err
	}
	if nilData && elements != 0 {
		return Tensor{}, fmt.Errorf("%w: data is nil for a non-empty tensor", ErrInvalidTensor)
	}
	if elements != uint64(length) {
		return Tensor{}, fmt.Errorf("%w: shape contains %d elements but data contains %d", ErrInvalidTensor, elements, length)
	}
	return Tensor{shape: append([]int64(nil), shape...), dataType: dataType, data: data}, nil
}

func elementCount(shape []int64) (uint64, error) {
	if len(shape) > 64 {
		return 0, fmt.Errorf("%w: tensor rank exceeds 64", ErrResourceLimit)
	}
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension < 0 {
			return 0, fmt.Errorf("%w: negative dimension %d", ErrInvalidTensor, dimension)
		}
		if dimension == 0 {
			elements = 0
			continue
		}
		if elements > math.MaxUint64/uint64(dimension) {
			return 0, fmt.Errorf("%w: shape element count overflows uint64", ErrInvalidTensor)
		}
		elements *= uint64(dimension)
	}
	return elements, nil
}

// Shape returns a copy of the tensor shape.
func (t Tensor) Shape() []int64 { return append([]int64(nil), t.shape...) }

// DataType returns the tensor element type.
func (t Tensor) DataType() DataType { return t.dataType }

// Data returns a copy of the typed backing slice.
func (t Tensor) Data() any {
	switch values := t.data.(type) {
	case []float32:
		return append([]float32(nil), values...)
	case []float64:
		return append([]float64(nil), values...)
	case []int32:
		return append([]int32(nil), values...)
	case []int64:
		return append([]int64(nil), values...)
	case []uint8:
		return append([]uint8(nil), values...)
	case []bool:
		return append([]bool(nil), values...)
	default:
		return nil
	}
}

func (t Tensor) Float32Data() ([]float32, bool) {
	values, ok := t.data.([]float32)
	return append([]float32(nil), values...), ok
}

func (t Tensor) Float64Data() ([]float64, bool) {
	values, ok := t.data.([]float64)
	return append([]float64(nil), values...), ok
}

func (t Tensor) Int32Data() ([]int32, bool) {
	values, ok := t.data.([]int32)
	return append([]int32(nil), values...), ok
}

func (t Tensor) Int64Data() ([]int64, bool) {
	values, ok := t.data.([]int64)
	return append([]int64(nil), values...), ok
}

func (t Tensor) Uint8Data() ([]uint8, bool) {
	values, ok := t.data.([]uint8)
	return append([]uint8(nil), values...), ok
}

func (t Tensor) BoolData() ([]bool, bool) {
	values, ok := t.data.([]bool)
	return append([]bool(nil), values...), ok
}
