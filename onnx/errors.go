package onnx

import (
	"errors"
	"fmt"
)

var (
	ErrEngineClosed  = errors.New("onnx: engine is closed")
	ErrModelClosed   = errors.New("onnx: model is closed")
	ErrInvalidModel  = errors.New("onnx: invalid model")
	ErrInvalidTensor = errors.New("onnx: invalid tensor")
	ErrResourceLimit = errors.New("onnx: resource limit exceeded")
)

// ErrorCode is the stable error category returned by the WASM runtime.
type ErrorCode int32

const (
	ErrorOK                  ErrorCode = 0
	ErrorInvalidArgument     ErrorCode = 1
	ErrorInvalidHandle       ErrorCode = 2
	ErrorModelParse          ErrorCode = 3
	ErrorModelOptimization   ErrorCode = 4
	ErrorInference           ErrorCode = 5
	ErrorUnsupportedDataType ErrorCode = 6
	ErrorInvalidTensor       ErrorCode = 7
	ErrorBufferTooSmall      ErrorCode = 8
	ErrorSerialization       ErrorCode = 9
	ErrorInternal            ErrorCode = 10
	ErrorResourceLimit       ErrorCode = 11
)

// RuntimeError contains a categorized Rust runtime failure and its diagnostic.
type RuntimeError struct {
	Code    ErrorCode
	Message string
}

func (e *RuntimeError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("onnx: runtime error %d", e.Code)
	}
	return fmt.Sprintf("onnx: %s", e.Message)
}

func (e *RuntimeError) Unwrap() error {
	switch e.Code {
	case ErrorModelParse, ErrorModelOptimization:
		return ErrInvalidModel
	case ErrorInvalidTensor, ErrorUnsupportedDataType:
		return ErrInvalidTensor
	case ErrorResourceLimit:
		return ErrResourceLimit
	default:
		return nil
	}
}
