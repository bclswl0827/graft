package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	tensorMagic   = uint32('O') | uint32('N')<<8 | uint32('X')<<16 | uint32('R')<<24
	metadataMagic = uint32('O') | uint32('N')<<8 | uint32('X')<<16 | uint32('M')<<24
	packageMagic  = uint32('O') | uint32('N')<<8 | uint32('X')<<16 | uint32('P')<<24
	protocolVer   = uint16(1)
)

func encodeRequest(tensors []Tensor, maxTensorBytes, maxOutputBytes uint64) ([]byte, error) {
	if len(tensors) > 1024 {
		return nil, fmt.Errorf("%w: input tensor count exceeds 1024", ErrResourceLimit)
	}
	encoder := binaryEncoder{bytes: make([]byte, 0, 20)}
	encoder.u32(tensorMagic)
	encoder.u16(protocolVer)
	encoder.u16(0)
	encoder.u32(uint32(len(tensors)))
	encoder.u64(maxOutputBytes)
	for index, tensor := range tensors {
		if err := encoder.tensor(tensor, maxTensorBytes); err != nil {
			return nil, fmt.Errorf("input %d: %w", index, err)
		}
	}
	if uint64(len(encoder.bytes)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: serialized request exceeds WASM32", ErrResourceLimit)
	}
	return encoder.bytes, nil
}

func decodeResponse(data []byte, maxTensorBytes, maxOutputBytes uint64) ([]Tensor, error) {
	if uint64(len(data)) > maxOutputBytes {
		return nil, fmt.Errorf("%w: response exceeds output byte limit", ErrResourceLimit)
	}
	decoder := binaryDecoder{bytes: data}
	magic, err := decoder.u32()
	if err != nil || magic != tensorMagic {
		return nil, fmt.Errorf("%w: invalid response magic", ErrInvalidTensor)
	}
	version, err := decoder.u16()
	if err != nil || version != protocolVer {
		return nil, fmt.Errorf("%w: unsupported response protocol", ErrInvalidTensor)
	}
	if err := decoder.skip(2); err != nil {
		return nil, err
	}
	count, err := decoder.u32()
	if err != nil {
		return nil, err
	}
	if count > 1024 {
		return nil, fmt.Errorf("%w: output tensor count exceeds 1024", ErrResourceLimit)
	}
	if _, err := decoder.u64(); err != nil {
		return nil, err
	}
	outputs := make([]Tensor, 0, count)
	for index := uint32(0); index < count; index++ {
		tensor, err := decoder.tensor(maxTensorBytes)
		if err != nil {
			return nil, fmt.Errorf("output %d: %w", index, err)
		}
		outputs = append(outputs, tensor)
	}
	if !decoder.finished() {
		return nil, fmt.Errorf("%w: trailing response data", ErrInvalidTensor)
	}
	return outputs, nil
}

func decodeMetadata(data []byte) ([]ValueInfo, []ValueInfo, error) {
	decoder := binaryDecoder{bytes: data}
	magic, err := decoder.u32()
	if err != nil || magic != metadataMagic {
		return nil, nil, fmt.Errorf("%w: invalid metadata magic", ErrInvalidModel)
	}
	version, err := decoder.u16()
	if err != nil || version != protocolVer {
		return nil, nil, fmt.Errorf("%w: unsupported metadata protocol", ErrInvalidModel)
	}
	if err := decoder.skip(2); err != nil {
		return nil, nil, err
	}
	inputCount, err := decoder.u32()
	if err != nil {
		return nil, nil, err
	}
	outputCount, err := decoder.u32()
	if err != nil {
		return nil, nil, err
	}
	if inputCount > 1024 || outputCount > 1024 {
		return nil, nil, fmt.Errorf("%w: metadata value count exceeds 1024", ErrResourceLimit)
	}
	values := make([]ValueInfo, 0, uint64(inputCount)+uint64(outputCount))
	for index := uint32(0); index < inputCount+outputCount; index++ {
		value, err := decoder.valueInfo()
		if err != nil {
			return nil, nil, err
		}
		values = append(values, value)
	}
	if !decoder.finished() {
		return nil, nil, fmt.Errorf("%w: trailing metadata data", ErrInvalidModel)
	}
	return values[:inputCount], values[inputCount:], nil
}

type binaryEncoder struct{ bytes []byte }

func (e *binaryEncoder) u8(value uint8)   { e.bytes = append(e.bytes, value) }
func (e *binaryEncoder) u16(value uint16) { e.bytes = binary.LittleEndian.AppendUint16(e.bytes, value) }
func (e *binaryEncoder) u32(value uint32) { e.bytes = binary.LittleEndian.AppendUint32(e.bytes, value) }
func (e *binaryEncoder) u64(value uint64) { e.bytes = binary.LittleEndian.AppendUint64(e.bytes, value) }
func (e *binaryEncoder) i64(value int64)  { e.u64(uint64(value)) }

func (e *binaryEncoder) tensor(tensor Tensor, maxTensorBytes uint64) error {
	elementSize, ok := tensor.dataType.size()
	if !ok {
		return fmt.Errorf("%w: unsupported data type %s", ErrInvalidTensor, tensor.dataType)
	}
	elements, err := elementCount(tensor.shape)
	if err != nil {
		return err
	}
	if elements > math.MaxUint64/elementSize {
		return fmt.Errorf("%w: tensor byte length overflows", ErrInvalidTensor)
	}
	byteLength := elements * elementSize
	if byteLength > maxTensorBytes {
		return fmt.Errorf("%w: tensor contains %d bytes; limit is %d", ErrResourceLimit, byteLength, maxTensorBytes)
	}
	e.u8(uint8(tensor.dataType))
	e.bytes = append(e.bytes, 0, 0, 0)
	e.u32(uint32(len(tensor.shape)))
	e.u64(byteLength)
	for _, dimension := range tensor.shape {
		e.i64(dimension)
	}
	switch values := tensor.data.(type) {
	case []float32:
		for _, value := range values {
			e.u32(math.Float32bits(value))
		}
	case []float64:
		for _, value := range values {
			e.u64(math.Float64bits(value))
		}
	case []int32:
		for _, value := range values {
			e.u32(uint32(value))
		}
	case []int64:
		for _, value := range values {
			e.i64(value)
		}
	case []uint8:
		e.bytes = append(e.bytes, values...)
	case []bool:
		for _, value := range values {
			if value {
				e.u8(1)
			} else {
				e.u8(0)
			}
		}
	default:
		return fmt.Errorf("%w: tensor has invalid backing data", ErrInvalidTensor)
	}
	return nil
}

type binaryDecoder struct {
	bytes  []byte
	offset uint64
}

func (d *binaryDecoder) take(length uint64) ([]byte, error) {
	if length > uint64(len(d.bytes)) || d.offset > uint64(len(d.bytes))-length {
		return nil, fmt.Errorf("%w: truncated binary protocol", ErrInvalidTensor)
	}
	start := d.offset
	d.offset += length
	return d.bytes[start:d.offset], nil
}

func (d *binaryDecoder) skip(length uint64) error { _, err := d.take(length); return err }
func (d *binaryDecoder) finished() bool           { return d.offset == uint64(len(d.bytes)) }
func (d *binaryDecoder) u8() (uint8, error) {
	value, err := d.take(1)
	if err != nil {
		return 0, err
	}
	return value[0], nil
}
func (d *binaryDecoder) u16() (uint16, error) {
	value, err := d.take(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(value), nil
}
func (d *binaryDecoder) u32() (uint32, error) {
	value, err := d.take(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(value), nil
}
func (d *binaryDecoder) u64() (uint64, error) {
	value, err := d.take(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(value), nil
}
func (d *binaryDecoder) i64() (int64, error) {
	value, err := d.u64()
	return int64(value), err
}

func (d *binaryDecoder) tensor(maxTensorBytes uint64) (Tensor, error) {
	typeValue, err := d.u8()
	if err != nil {
		return Tensor{}, err
	}
	dataType := DataType(typeValue)
	elementSize, ok := dataType.size()
	if !ok {
		return Tensor{}, fmt.Errorf("%w: unsupported output data type %d", ErrInvalidTensor, typeValue)
	}
	if err := d.skip(3); err != nil {
		return Tensor{}, err
	}
	rank, err := d.u32()
	if err != nil {
		return Tensor{}, err
	}
	if rank > 64 {
		return Tensor{}, fmt.Errorf("%w: tensor rank exceeds 64", ErrResourceLimit)
	}
	byteLength, err := d.u64()
	if err != nil {
		return Tensor{}, err
	}
	if byteLength > maxTensorBytes {
		return Tensor{}, fmt.Errorf("%w: output tensor exceeds tensor byte limit", ErrResourceLimit)
	}
	shape := make([]int64, rank)
	for index := range shape {
		shape[index], err = d.i64()
		if err != nil || shape[index] < 0 {
			return Tensor{}, fmt.Errorf("%w: invalid output dimension", ErrInvalidTensor)
		}
	}
	elements, err := elementCount(shape)
	if err != nil || elements > math.MaxUint64/elementSize || elements*elementSize != byteLength {
		return Tensor{}, fmt.Errorf("%w: output shape and byte length differ", ErrInvalidTensor)
	}
	raw, err := d.take(byteLength)
	if err != nil {
		return Tensor{}, err
	}
	length := int(elements)
	var values any
	switch dataType {
	case Float32:
		data := make([]float32, length)
		for index := range data {
			data[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
		}
		values = data
	case Float64:
		data := make([]float64, length)
		for index := range data {
			data[index] = math.Float64frombits(binary.LittleEndian.Uint64(raw[index*8:]))
		}
		values = data
	case Int32:
		data := make([]int32, length)
		for index := range data {
			data[index] = int32(binary.LittleEndian.Uint32(raw[index*4:]))
		}
		values = data
	case Int64:
		data := make([]int64, length)
		for index := range data {
			data[index] = int64(binary.LittleEndian.Uint64(raw[index*8:]))
		}
		values = data
	case Uint8:
		values = append([]uint8(nil), raw...)
	case Bool:
		data := make([]bool, length)
		for index, value := range raw {
			if value > 1 {
				return Tensor{}, fmt.Errorf("%w: invalid bool byte %d", ErrInvalidTensor, value)
			}
			data[index] = value == 1
		}
		values = data
	}
	return Tensor{shape: shape, dataType: dataType, data: values}, nil
}

func (d *binaryDecoder) valueInfo() (ValueInfo, error) {
	nameLength, err := d.u32()
	if err != nil || nameLength > 1<<20 {
		return ValueInfo{}, fmt.Errorf("%w: invalid metadata name length", ErrInvalidModel)
	}
	name, err := d.take(uint64(nameLength))
	if err != nil {
		return ValueInfo{}, err
	}
	typeValue, err := d.u8()
	if err != nil {
		return ValueInfo{}, err
	}
	dataType := DataType(typeValue)
	if _, ok := dataType.size(); !ok {
		return ValueInfo{}, fmt.Errorf("%w: invalid metadata data type", ErrInvalidModel)
	}
	rankKnown, err := d.u8()
	if err != nil || rankKnown > 1 {
		return ValueInfo{}, fmt.Errorf("%w: invalid metadata rank flag", ErrInvalidModel)
	}
	if err := d.skip(2); err != nil {
		return ValueInfo{}, err
	}
	rank, err := d.u32()
	if err != nil || rank > 64 {
		return ValueInfo{}, fmt.Errorf("%w: invalid metadata rank", ErrInvalidModel)
	}
	dimensions := make([]Dimension, rank)
	for index := range dimensions {
		kind, err := d.u8()
		if err != nil {
			return ValueInfo{}, err
		}
		if err := d.skip(3); err != nil {
			return ValueInfo{}, err
		}
		value, err := d.i64()
		if err != nil {
			return ValueInfo{}, err
		}
		symbolLength, err := d.u32()
		if err != nil || symbolLength > 1<<20 {
			return ValueInfo{}, fmt.Errorf("%w: invalid dimension symbol length", ErrInvalidModel)
		}
		symbol, err := d.take(uint64(symbolLength))
		if err != nil {
			return ValueInfo{}, err
		}
		dimensions[index] = Dimension{Kind: DimensionKind(kind), Value: value, Symbol: string(symbol)}
		if dimensions[index].Kind < DimensionKnown || dimensions[index].Kind > DimensionDynamic {
			return ValueInfo{}, fmt.Errorf("%w: invalid dimension kind", ErrInvalidModel)
		}
	}
	return ValueInfo{Name: string(name), Type: dataType, RankKnown: rankKnown == 1, Shape: dimensions}, nil
}
