package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strings"

	"github.com/bclswl0827/graft/onnx"
)

const (
	defaultModelPath      = "examples/ocr/model.onnx"
	defaultDictionaryPath = "examples/ocr/ppocr_keys_v1.txt"
)

func main() {
	modelPath := flag.String("model", defaultModelPath, "path to the ONNX recognition model")
	dictionaryPath := flag.String("dict", defaultDictionaryPath, "path to the character dictionary")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: go run ./examples/ocr [options] IMAGE\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*modelPath, *dictionaryPath, flag.Arg(0)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(modelPath, dictionaryPath, imagePath string) error {
	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}
	characters, err := loadCharacters(dictionaryPath)
	if err != nil {
		return fmt.Errorf("load character dictionary: %w", err)
	}
	source, err := loadImage(imagePath)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	engine, err := onnx.NewEngine(onnx.WithNumThreads(1))
	if err != nil {
		return fmt.Errorf("create engine: %w", err)
	}
	defer engine.Close()

	model, err := engine.LoadModel(modelData)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer model.Close()

	lines := detectTextLines(source)
	recognizedLines := make([]string, 0, len(lines))
	for lineIndex, line := range lines {
		words := make([]string, 0, len(line))
		for wordIndex, bounds := range line {
			text, err := recognize(model, imageRegion{Image: source, bounds: bounds}, characters)
			if err != nil {
				return fmt.Errorf("recognize line %d word %d: %w", lineIndex+1, wordIndex+1, err)
			}
			if text = strings.TrimSpace(text); text != "" {
				words = append(words, text)
			}
		}
		if len(words) != 0 {
			recognizedLines = append(recognizedLines, strings.Join(words, " "))
		}
	}
	if len(recognizedLines) == 0 {
		return fmt.Errorf("no text recognized")
	}
	fmt.Println(strings.Join(recognizedLines, "\n"))
	return nil
}

func recognize(model *onnx.Model, source image.Image, characters []string) (string, error) {
	input, err := imageTensor(source)
	if err != nil {
		return "", fmt.Errorf("prepare image: %w", err)
	}
	outputs, err := model.Run([]onnx.Tensor{input})
	if err != nil {
		return "", fmt.Errorf("run model: %w", err)
	}
	if len(outputs) != 1 {
		return "", fmt.Errorf("model returned %d outputs, expected 1", len(outputs))
	}
	text, err := decodeCTC(outputs[0], characters)
	if err != nil {
		return "", fmt.Errorf("decode output: %w", err)
	}
	return text, nil
}

type imageRegion struct {
	image.Image
	bounds image.Rectangle
}

func (region imageRegion) Bounds() image.Rectangle {
	return region.bounds
}
