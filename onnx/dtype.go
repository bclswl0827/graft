package onnx

import "fmt"

// DataType is a tensor element type supported by ABI version 1.
type DataType uint8

const (
	Float32 DataType = 1
	Float64 DataType = 2
	Int32   DataType = 3
	Int64   DataType = 4
	Uint8   DataType = 5
	Bool    DataType = 6
)

func (d DataType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint8:
		return "uint8"
	case Bool:
		return "bool"
	default:
		return fmt.Sprintf("DataType(%d)", d)
	}
}

func (d DataType) size() (uint64, bool) {
	switch d {
	case Float32, Int32:
		return 4, true
	case Float64, Int64:
		return 8, true
	case Uint8, Bool:
		return 1, true
	default:
		return 0, false
	}
}
