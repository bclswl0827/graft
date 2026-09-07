package wasm

import (
	"context"
	"testing"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/experimental"
)

func TestSharedMemoryModule(t *testing.T) {
	ctx := context.Background()
	host := wazero.NewRuntimeWithConfig(ctx, wazero.NewRuntimeConfig().WithCoreFeatures(api.CoreFeaturesV2|experimental.CoreFeaturesThreads))
	defer host.Close(ctx)
	compiled, err := host.CompileModule(ctx, sharedMemoryModule(31, 16_384))
	if err != nil {
		t.Fatalf("CompileModule: %v", err)
	}
	memory, ok := compiled.ExportedMemories()["memory"]
	if !ok {
		t.Fatal("memory export is missing")
	}
	maximum, bounded := memory.Max()
	if memory.Min() != 31 || !bounded || maximum != 16_384 {
		t.Fatalf("memory limits = min %d, max %d, bounded %v", memory.Min(), maximum, bounded)
	}
}

func TestAppendULEB32(t *testing.T) {
	actual := appendULEB32(nil, 16_384)
	expected := []byte{0x80, 0x80, 0x01}
	if string(actual) != string(expected) {
		t.Fatalf("appendULEB32 = %x, want %x", actual, expected)
	}
}
