package onnx

// DimensionKind describes whether an ONNX dimension is fixed, symbolic, or dynamic.
type DimensionKind uint8

const (
	DimensionKnown    DimensionKind = 1
	DimensionSymbolic DimensionKind = 2
	DimensionDynamic  DimensionKind = 3
)

// Dimension is one axis in ONNX value metadata.
type Dimension struct {
	Kind   DimensionKind
	Value  int64
	Symbol string
}

// ValueInfo describes one model input or output. RankKnown distinguishes a
// scalar (known rank zero) from an ONNX value whose rank is unspecified.
type ValueInfo struct {
	Name      string
	Type      DataType
	RankKnown bool
	Shape     []Dimension
}

func cloneValueInfo(values []ValueInfo) []ValueInfo {
	result := make([]ValueInfo, len(values))
	for index, value := range values {
		result[index] = value
		result[index].Shape = append([]Dimension(nil), value.Shape...)
	}
	return result
}
