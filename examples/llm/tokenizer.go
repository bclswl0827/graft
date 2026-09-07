package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	beginningOfStory = "<|start_story|>"
	endOfStory       = "<|end_story|>"
)

type tokenPair struct {
	left  string
	right string
}

type bpeTokenizer struct {
	tokenToID  map[string]int64
	idToToken  []string
	mergeRanks map[tokenPair]int
	unknownID  int64
	bosID      int64
	eosID      int64
}

type tokenizerDocument struct {
	Model struct {
		Type         string           `json:"type"`
		UnknownToken string           `json:"unk_token"`
		Vocabulary   map[string]int64 `json:"vocab"`
		Merges       []string         `json:"merges"`
	} `json:"model"`
}

func newBPETokenizer(data []byte) (*bpeTokenizer, error) {
	var document tokenizerDocument
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("parse tokenizer JSON: %w", err)
	}
	if document.Model.Type != "BPE" || len(document.Model.Vocabulary) == 0 {
		return nil, fmt.Errorf("expected a non-empty BPE vocabulary")
	}

	maximumID := int64(-1)
	for _, tokenID := range document.Model.Vocabulary {
		if tokenID < 0 {
			return nil, fmt.Errorf("vocabulary contains negative token ID %d", tokenID)
		}
		maximumID = max(maximumID, tokenID)
	}
	idToToken := make([]string, maximumID+1)
	assigned := make([]bool, maximumID+1)
	for token, tokenID := range document.Model.Vocabulary {
		if assigned[tokenID] {
			return nil, fmt.Errorf("vocabulary contains duplicate token ID %d", tokenID)
		}
		idToToken[tokenID] = token
		assigned[tokenID] = true
	}
	for tokenID, ok := range assigned {
		if !ok {
			return nil, fmt.Errorf("vocabulary is missing token ID %d", tokenID)
		}
	}

	mergeRanks := make(map[tokenPair]int, len(document.Model.Merges))
	for rank, merge := range document.Model.Merges {
		parts := strings.SplitN(merge, " ", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid BPE merge %q", merge)
		}
		mergeRanks[tokenPair{left: parts[0], right: parts[1]}] = rank
	}

	unknownID, ok := document.Model.Vocabulary[document.Model.UnknownToken]
	if !ok {
		return nil, fmt.Errorf("vocabulary does not contain unknown token %q", document.Model.UnknownToken)
	}
	bosID, ok := document.Model.Vocabulary[beginningOfStory]
	if !ok {
		return nil, fmt.Errorf("vocabulary does not contain %q", beginningOfStory)
	}
	eosID, ok := document.Model.Vocabulary[endOfStory]
	if !ok {
		return nil, fmt.Errorf("vocabulary does not contain %q", endOfStory)
	}

	return &bpeTokenizer{
		tokenToID:  document.Model.Vocabulary,
		idToToken:  idToToken,
		mergeRanks: mergeRanks,
		unknownID:  unknownID,
		bosID:      bosID,
		eosID:      eosID,
	}, nil
}

func (t *bpeTokenizer) vocabularySize() int {
	return len(t.idToToken)
}

func (t *bpeTokenizer) encode(text string) ([]int64, error) {
	normalized := "▁" + strings.ReplaceAll(text, " ", "▁")
	pieces := make([]string, 0, len(normalized))
	for _, character := range normalized {
		pieces = append(pieces, string(character))
	}
	pieces = t.merge(pieces)

	tokenIDs := make([]int64, 1, len(pieces)+1)
	tokenIDs[0] = t.bosID
	for _, piece := range pieces {
		tokenID, ok := t.tokenToID[piece]
		if !ok {
			tokenID = t.unknownID
		}
		tokenIDs = append(tokenIDs, tokenID)
	}
	return tokenIDs, nil
}

func (t *bpeTokenizer) merge(pieces []string) []string {
	for len(pieces) > 1 {
		bestRank := len(t.mergeRanks)
		var best tokenPair
		found := false
		for index := 0; index+1 < len(pieces); index++ {
			pair := tokenPair{left: pieces[index], right: pieces[index+1]}
			rank, ok := t.mergeRanks[pair]
			if ok && rank < bestRank {
				bestRank = rank
				best = pair
				found = true
			}
		}
		if !found {
			break
		}

		merged := make([]string, 0, len(pieces))
		for index := 0; index < len(pieces); {
			if index+1 < len(pieces) && pieces[index] == best.left && pieces[index+1] == best.right {
				merged = append(merged, best.left+best.right)
				index += 2
				continue
			}
			merged = append(merged, pieces[index])
			index++
		}
		pieces = merged
	}
	return pieces
}

func (t *bpeTokenizer) decode(tokenIDs []int64) (string, error) {
	var decoded strings.Builder
	for _, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= int64(len(t.idToToken)) {
			return "", fmt.Errorf("token ID %d is outside the vocabulary", tokenID)
		}
		if tokenID == t.bosID || tokenID == t.eosID {
			continue
		}
		decoded.WriteString(t.idToToken[tokenID])
	}
	text := strings.ReplaceAll(decoded.String(), "▁", " ")
	text = strings.ReplaceAll(text, beginningOfStory, "")
	text = strings.ReplaceAll(text, endOfStory, "")
	return strings.TrimPrefix(text, " "), nil
}

func (t *bpeTokenizer) endsStory(tokenIDs []int64) bool {
	if len(tokenIDs) == 0 {
		return false
	}
	if tokenIDs[len(tokenIDs)-1] == t.eosID {
		return true
	}

	start := max(0, len(tokenIDs)-8)
	var suffix strings.Builder
	for _, tokenID := range tokenIDs[start:] {
		if tokenID >= 0 && tokenID < int64(len(t.idToToken)) {
			suffix.WriteString(t.idToToken[tokenID])
		}
	}
	return strings.HasSuffix(suffix.String(), endOfStory)
}
