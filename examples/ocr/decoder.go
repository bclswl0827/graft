package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/bclswl0827/graft/onnx"
)

func loadCharacters(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var characters []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		characters = append(characters, strings.TrimSuffix(scanner.Text(), "\r"))
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(characters) == 0 {
		return nil, fmt.Errorf("dictionary is empty")
	}

	// The PP-OCRv4 model was exported with use_space_char enabled.
	return append(characters, " "), nil
}

func decodeCTC(output onnx.Tensor, characters []string) (string, error) {
	probabilities, ok := output.Float32Data()
	shape := output.Shape()
	if !ok || len(shape) != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 1 {
		return "", fmt.Errorf("unexpected output: type=%s shape=%v", output.DataType(), shape)
	}
	steps, classes := shape[1], shape[2]
	if int64(len(probabilities)) != steps*classes {
		return "", fmt.Errorf("output shape %v does not match %d probabilities", shape, len(probabilities))
	}
	if int64(len(characters))+1 != classes {
		return "", fmt.Errorf("model has %d classes but dictionary defines %d characters", classes, len(characters))
	}

	var text strings.Builder
	previous := int64(-1)
	for step := int64(0); step < steps; step++ {
		offset := step * classes
		class := int64(0)
		for candidate := int64(1); candidate < classes; candidate++ {
			if probabilities[int(offset+candidate)] > probabilities[int(offset+class)] {
				class = candidate
			}
		}
		if class != 0 && class != previous {
			text.WriteString(characters[class-1])
		}
		previous = class
	}
	return text.String(), nil
}
