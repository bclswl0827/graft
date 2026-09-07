package main

import (
	"bufio"
	_ "embed"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/bclswl0827/graft/onnx"
)

//go:embed model.onnx
var modelData []byte

//go:embed tokenizer.json
var tokenizerData []byte

const (
	maxSequenceLength = 128
	maxNewTokens      = 96
	topK              = 40
	topP              = 0.9
	temperature       = 0.6
)

const defaultPrompt = "Once upon a time, there was a little girl named Lily"

func main() {
	prompt := defaultPrompt
	if len(os.Args) > 1 {
		prompt = strings.Join(os.Args[1:], " ")
	}
	if err := run(prompt); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(prompt string) error {
	tokenizer, err := newBPETokenizer(tokenizerData)
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
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

	scanner := bufio.NewScanner(os.Stdin)
	for {
		tokenIDs, err := tokenizer.encode(prompt)
		if err != nil {
			return fmt.Errorf("encode prompt: %w", err)
		}
		promptTokens := len(tokenIDs)
		started := time.Now()
		tokenIDs, ended, err := generate(model, tokenizer, tokenIDs, maxNewTokens)
		elapsed := time.Since(started)
		if err != nil {
			return err
		}
		text, err := tokenizer.decode(tokenIDs)
		if err != nil {
			return fmt.Errorf("decode output: %w", err)
		}
		if !ended {
			text = trimIncompleteSentence(text)
		}
		fmt.Println(text)
		generatedTokens := len(tokenIDs) - promptTokens
		fmt.Printf(
			"\nGenerated %d tokens in %.2fs (%.2f tokens/s)\n",
			generatedTokens,
			elapsed.Seconds(),
			float64(generatedTokens)/elapsed.Seconds(),
		)

		for {
			fmt.Print("\nPress Enter to generate again, or enter q to quit: ")
			if !scanner.Scan() {
				if err := scanner.Err(); err != nil {
					return fmt.Errorf("read command: %w", err)
				}
				return nil
			}
			command := strings.TrimSpace(scanner.Text())
			if strings.EqualFold(command, "q") {
				return nil
			}
			if command == "" {
				break
			}
		}
	}
}

func generate(
	model *onnx.Model,
	tokenizer *bpeTokenizer,
	prompt []int64,
	limit int,
) ([]int64, bool, error) {
	if len(prompt) == 0 {
		return nil, false, fmt.Errorf("prompt must contain at least one token")
	}
	if len(prompt) >= maxSequenceLength {
		return nil, false, fmt.Errorf("prompt exceeds the %d-token context window", maxSequenceLength)
	}
	limit = min(limit, maxSequenceLength-len(prompt))

	tokenIDs := append([]int64(nil), prompt...)
	for range limit {
		next, err := runStep(model, tokenizer.vocabularySize(), tokenIDs)
		if err != nil {
			return nil, false, err
		}
		tokenIDs = append(tokenIDs, next)
		if tokenizer.endsStory(tokenIDs) {
			return tokenIDs, true, nil
		}
	}
	return tokenIDs, false, nil
}

func runStep(model *onnx.Model, vocabularySize int, tokenIDs []int64) (int64, error) {
	sequenceLength := len(tokenIDs)
	input, err := onnx.NewInt64Tensor(
		[]int64{1, int64(sequenceLength)},
		tokenIDs,
	)
	if err != nil {
		return 0, fmt.Errorf("create input IDs: %w", err)
	}

	positionValues := make([]int64, sequenceLength)
	for index := range positionValues {
		positionValues[index] = int64(index)
	}
	positionIDs, err := onnx.NewInt64Tensor(
		[]int64{1, int64(sequenceLength)},
		positionValues,
	)
	if err != nil {
		return 0, fmt.Errorf("create position IDs: %w", err)
	}

	attentionValues := make([]float32, sequenceLength*sequenceLength)
	for row := range sequenceLength {
		for column := row + 1; column < sequenceLength; column++ {
			attentionValues[row*sequenceLength+column] = -10_000
		}
	}
	attentionMask, err := onnx.NewTensor(
		[]int64{1, 1, int64(sequenceLength), int64(sequenceLength)},
		attentionValues,
	)
	if err != nil {
		return 0, fmt.Errorf("create attention mask: %w", err)
	}

	outputs, err := model.RunNamed(map[string]onnx.Tensor{
		"input_ids":      input,
		"position_ids":   positionIDs,
		"attention_mask": attentionMask,
	})
	if err != nil {
		return 0, fmt.Errorf("run model: %w", err)
	}
	logitsTensor, ok := outputs["logits"]
	if !ok {
		return 0, fmt.Errorf("model did not return logits output")
	}
	logits, ok := logitsTensor.Float32Data()
	shape := logitsTensor.Shape()
	if !ok || len(shape) != 3 || shape[0] != 1 ||
		shape[1] != int64(sequenceLength) || shape[2] != int64(vocabularySize) {
		return 0, fmt.Errorf(
			"unexpected logits: type=%s shape=%v",
			logitsTensor.DataType(),
			shape,
		)
	}
	offset := (sequenceLength - 1) * vocabularySize
	return sampleToken(logits[offset : offset+vocabularySize]), nil
}

type tokenCandidate struct {
	token  int64
	logit  float64
	weight float64
}

func sampleToken(logits []float32) int64 {
	candidates := make([]tokenCandidate, len(logits))
	for token, logit := range logits {
		candidates[token] = tokenCandidate{token: int64(token), logit: float64(logit)}
	}
	sort.Slice(candidates, func(left, right int) bool {
		return candidates[left].logit > candidates[right].logit
	})
	candidates = candidates[:min(topK, len(candidates))]

	maximum := candidates[0].logit
	var total float64
	for index := range candidates {
		candidates[index].weight = math.Exp((candidates[index].logit - maximum) / temperature)
		total += candidates[index].weight
	}

	cutoff := len(candidates)
	var cumulative float64
	for index, candidate := range candidates {
		cumulative += candidate.weight
		if cumulative/total >= topP {
			cutoff = index + 1
			break
		}
	}
	candidates = candidates[:cutoff]

	var retained float64
	for _, candidate := range candidates {
		retained += candidate.weight
	}
	target := rand.Float64() * retained
	for _, candidate := range candidates {
		target -= candidate.weight
		if target <= 0 {
			return candidate.token
		}
	}
	return candidates[len(candidates)-1].token
}

func trimIncompleteSentence(text string) string {
	end := strings.LastIndexAny(text, ".!?")
	if end < 0 {
		return text
	}
	end++
	for end < len(text) && (text[end] == '\'' || text[end] == '"') {
		end++
	}
	return strings.TrimSpace(text[:end])
}
