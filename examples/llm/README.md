# TinyStories ONNX LLM example

This example runs the pretrained
[`TinyStories-656K`](https://huggingface.co/raincandy-u/TinyStories-656K)
language model. It is a 656,000-parameter Llama model with two transformer
layers, grouped-query attention, and a 2,048-token BPE vocabulary.

Run it from the repository root:

```bash
go run ./examples/llm
```

The default prompt is `Once upon a time, there was a little girl named Lily`.
Pass arguments to use another English prompt:

```bash
go run ./examples/llm "Once upon a time, a small robot"
```

The example embeds the FP32 ONNX model and tokenizer so inference does not
require network access. The Go code implements the model-specific BPE subset,
constructs a causal attention mask, and performs autoregressive decoding with
the model author's recommended temperature `0.6`, top-k `40`, and top-p `0.9`
sampling settings. The random sampling produces a different completion on each
run. The context is limited to 128 tokens and generation stops at the
end-of-story marker or after 96 new tokens. If the context limit is reached,
the displayed text is trimmed to its last complete sentence.

The upstream decoder-with-past export contains dynamic mask and cache shape
operations that tract cannot currently optimize. The checked-in model keeps
the same pretrained weights and Llama operations, but uses a simpler no-cache
graph whose causal mask is supplied by Go. It recomputes the sequence on each
step, which favors compatibility and clarity over generation speed.

`model.onnx` and `tokenizer.json` are derived from the pinned
[`onnx-community/TinyStories-656K-ONNX`](https://huggingface.co/onnx-community/TinyStories-656K-ONNX/tree/38d7e066648e4fd96f65c4c5d6f0cb0abc094a6d)
conversion. The upstream model is licensed under Apache-2.0; see
[`MODEL_LICENSE`](MODEL_LICENSE). To rebuild the checked-in model from the
verified upstream artifact, install Python `numpy` and `onnx`, then run:

```bash
python3 examples/llm/download_model.py
```
