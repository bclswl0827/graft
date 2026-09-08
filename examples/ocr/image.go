package main

import (
	"fmt"
	"image"
	"math"
	"os"

	"github.com/bclswl0827/graft/onnx"
)

const (
	inputChannels = 3
	inputHeight   = 48
	inputWidth    = 320
)

func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	source, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return source, nil
}

func imageTensor(source image.Image) (onnx.Tensor, error) {
	bounds := source.Bounds()
	if bounds.Dx() == 0 || bounds.Dy() == 0 {
		return onnx.Tensor{}, fmt.Errorf("image has no pixels")
	}

	resizedWidth := int(math.Ceil(float64(inputHeight) * float64(bounds.Dx()) / float64(bounds.Dy())))
	resizedWidth = min(max(resizedWidth, 1), inputWidth)
	values := make([]float32, inputChannels*inputHeight*inputWidth)
	resizeAndNormalize(source, resizedWidth, values)

	return onnx.NewTensor(
		[]int64{1, inputChannels, inputHeight, inputWidth},
		values,
	)
}

// resizeAndNormalize matches PaddleOCR's BGR, CHW, [-1, 1] preprocessing.
// The zero-filled remainder is the model's expected normalized padding.
func resizeAndNormalize(source image.Image, resizedWidth int, destination []float32) {
	bounds := source.Bounds()
	planeSize := inputHeight * inputWidth
	for y := range inputHeight {
		y0, y1, yWeight := interpolationPoints(y, bounds.Dy(), inputHeight)
		for x := range resizedWidth {
			x0, x1, xWeight := interpolationPoints(x, bounds.Dx(), resizedWidth)
			topLeft := opaqueRGB(source, bounds.Min.X+x0, bounds.Min.Y+y0)
			topRight := opaqueRGB(source, bounds.Min.X+x1, bounds.Min.Y+y0)
			bottomLeft := opaqueRGB(source, bounds.Min.X+x0, bounds.Min.Y+y1)
			bottomRight := opaqueRGB(source, bounds.Min.X+x1, bounds.Min.Y+y1)

			var rgb [3]float64
			for channel := range rgb {
				top := topLeft[channel] + (topRight[channel]-topLeft[channel])*xWeight
				bottom := bottomLeft[channel] + (bottomRight[channel]-bottomLeft[channel])*xWeight
				rgb[channel] = top + (bottom-top)*yWeight
			}

			pixel := y*inputWidth + x
			destination[pixel] = normalizeChannel(rgb[2])
			destination[planeSize+pixel] = normalizeChannel(rgb[1])
			destination[2*planeSize+pixel] = normalizeChannel(rgb[0])
		}
	}
}

func interpolationPoints(index, sourceSize, targetSize int) (int, int, float64) {
	coordinate := (float64(index)+0.5)*float64(sourceSize)/float64(targetSize) - 0.5
	lower := int(math.Floor(coordinate))
	upper := lower + 1
	weight := coordinate - math.Floor(coordinate)
	lower = min(max(lower, 0), sourceSize-1)
	upper = min(max(upper, 0), sourceSize-1)
	return lower, upper, weight
}

func opaqueRGB(source image.Image, x, y int) [3]float64 {
	red, green, blue, alpha := source.At(x, y).RGBA()
	background := uint32(0xffff) - alpha
	return [3]float64{
		float64(red + background),
		float64(green + background),
		float64(blue + background),
	}
}

func normalizeChannel(value float64) float32 {
	return float32(value/32767.5 - 1)
}
