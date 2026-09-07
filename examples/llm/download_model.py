#!/usr/bin/env python3
"""Build a tract-friendly ONNX graph from the pinned TinyStories model."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REVISION = "38d7e066648e4fd96f65c4c5d6f0cb0abc094a6d"
BASE_URL = f"https://huggingface.co/onnx-community/TinyStories-656K-ONNX/resolve/{REVISION}"
MODEL_URL = f"{BASE_URL}/onnx/model.onnx"
MODEL_SHA256 = "f2be66f8b09d10836e55a20daa538184bd82ddefead94db40be646e84ceb43aa"
TOKENIZER_URL = f"{BASE_URL}/tokenizer.json"
TOKENIZER_SHA256 = "9a7917d490ac034a4e69eb081096226971887e9e66b0969355421613f62b12e7"

CONTEXT_LENGTH = 128
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 384
ATTENTION_HEADS = 8
KEY_VALUE_HEADS = 4
HEAD_SIZE = 16
VOCABULARY_SIZE = 2048
RMS_NORM_EPSILON = 1e-6

LAYER_WEIGHTS = (
    {
        "input_norm": "model.layers.0.input_layernorm.weight",
        "query": "onnx::MatMul_847",
        "key": "onnx::MatMul_848",
        "value": "onnx::MatMul_849",
        "output": "onnx::MatMul_874",
        "post_norm": "model.layers.0.post_attention_layernorm.weight",
        "gate": "onnx::MatMul_875",
        "up": "onnx::MatMul_876",
        "down": "onnx::MatMul_877",
    },
    {
        "input_norm": "model.layers.1.input_layernorm.weight",
        "query": "onnx::MatMul_878",
        "key": "onnx::MatMul_879",
        "value": "onnx::MatMul_880",
        "output": "onnx::MatMul_905",
        "post_norm": "model.layers.1.post_attention_layernorm.weight",
        "gate": "onnx::MatMul_906",
        "up": "onnx::MatMul_907",
        "down": "onnx::MatMul_908",
    },
)


def file_digest(path):
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, expected_digest):
    directory = Path(__file__).parent
    request = Request(url, headers={"User-Agent": "graft-llm-example"})
    temporary_path = None
    try:
        with urlopen(request, timeout=120) as response, NamedTemporaryFile(
            dir=directory, prefix=".download.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = sha256()
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise RuntimeError(f"SHA-256 mismatch for {url}")
        result = temporary_path
        temporary_path = None
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def add_node(nodes, operation, inputs, output, **attributes):
    nodes.append(
        helper.make_node(
            operation,
            inputs,
            [output],
            name=output,
            **attributes,
        )
    )
    return output


def rms_norm(nodes, value, weight, prefix):
    square = add_node(nodes, "Mul", [value, value], f"{prefix}.square")
    mean = add_node(
        nodes,
        "ReduceMean",
        [square],
        f"{prefix}.mean",
        axes=[-1],
        keepdims=1,
    )
    variance = add_node(nodes, "Add", [mean, "rms.epsilon"], f"{prefix}.variance")
    root = add_node(nodes, "Sqrt", [variance], f"{prefix}.root")
    normalized = add_node(nodes, "Div", [value, root], f"{prefix}.normalized")
    return add_node(nodes, "Mul", [normalized, weight], f"{prefix}.output")


def apply_rope(nodes, value, cosine, sine, prefix):
    reordered = add_node(
        nodes,
        "Gather",
        [value, "rope.rotate_indices"],
        f"{prefix}.reordered",
        axis=3,
    )
    rotated = add_node(
        nodes,
        "Mul",
        [reordered, "rope.rotate_signs"],
        f"{prefix}.rotated",
    )
    direct = add_node(nodes, "Mul", [value, cosine], f"{prefix}.direct")
    crossed = add_node(nodes, "Mul", [rotated, sine], f"{prefix}.crossed")
    return add_node(nodes, "Add", [direct, crossed], f"{prefix}.output")


def source_weight(source_initializers, source_name, target_name, shape):
    initializer = source_initializers.get(source_name)
    if initializer is None:
        raise RuntimeError(f"source model is missing initializer {source_name}")
    values = numpy_helper.to_array(initializer).astype(np.float32)
    if values.shape != shape:
        raise RuntimeError(
            f"initializer {source_name} has shape {values.shape}, expected {shape}"
        )
    return numpy_helper.from_array(values, target_name)


def build_model(source_path, target_path):
    source = onnx.load(source_path)
    source_initializers = {
        initializer.name: initializer for initializer in source.graph.initializer
    }
    nodes = []
    initializers = [
        source_weight(
            source_initializers,
            "model.embed_tokens.weight",
            "token_embedding",
            (VOCABULARY_SIZE, HIDDEN_SIZE),
        ),
        source_weight(
            source_initializers,
            "model.norm.weight",
            "final_norm.weight",
            (HIDDEN_SIZE,),
        ),
        numpy_helper.from_array(
            np.array(RMS_NORM_EPSILON, dtype=np.float32),
            "rms.epsilon",
        ),
        numpy_helper.from_array(
            np.array(1 / np.sqrt(HEAD_SIZE), dtype=np.float32),
            "attention.scale",
        ),
        numpy_helper.from_array(
            np.array([1, -1, ATTENTION_HEADS, HEAD_SIZE], dtype=np.int64),
            "query.shape",
        ),
        numpy_helper.from_array(
            np.array([1, -1, KEY_VALUE_HEADS, HEAD_SIZE], dtype=np.int64),
            "key_value.shape",
        ),
        numpy_helper.from_array(
            np.array([1, -1, HIDDEN_SIZE], dtype=np.int64),
            "attention_output.shape",
        ),
        numpy_helper.from_array(
            np.array([1], dtype=np.int64),
            "rope.unsqueeze_axes",
        ),
        numpy_helper.from_array(
            np.array(
                list(range(HEAD_SIZE // 2, HEAD_SIZE))
                + list(range(HEAD_SIZE // 2)),
                dtype=np.int64,
            ),
            "rope.rotate_indices",
        ),
        numpy_helper.from_array(
            np.array(
                [-1] * (HEAD_SIZE // 2) + [1] * (HEAD_SIZE // 2),
                dtype=np.float32,
            ),
            "rope.rotate_signs",
        ),
        numpy_helper.from_array(
            np.repeat(
                np.arange(KEY_VALUE_HEADS),
                ATTENTION_HEADS // KEY_VALUE_HEADS,
            ).astype(np.int64),
            "key_value.head_indices",
        ),
    ]

    positions = np.arange(CONTEXT_LENGTH, dtype=np.float32)[:, None]
    dimensions = np.arange(0, HEAD_SIZE, 2, dtype=np.float32)
    inverse_frequency = 1.0 / (10000.0 ** (dimensions / HEAD_SIZE))
    frequencies = positions * inverse_frequency[None, :]
    rope = np.concatenate((frequencies, frequencies), axis=1)
    initializers.extend(
        [
            numpy_helper.from_array(np.cos(rope).astype(np.float32), "rope.cosine"),
            numpy_helper.from_array(np.sin(rope).astype(np.float32), "rope.sine"),
        ]
    )

    for layer, names in enumerate(LAYER_WEIGHTS):
        prefix = f"layer.{layer}"
        initializers.extend(
            [
                source_weight(
                    source_initializers,
                    names["input_norm"],
                    f"{prefix}.input_norm.weight",
                    (HIDDEN_SIZE,),
                ),
                source_weight(
                    source_initializers,
                    names["query"],
                    f"{prefix}.query.weight",
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                ),
                source_weight(
                    source_initializers,
                    names["key"],
                    f"{prefix}.key.weight",
                    (HIDDEN_SIZE, KEY_VALUE_HEADS * HEAD_SIZE),
                ),
                source_weight(
                    source_initializers,
                    names["value"],
                    f"{prefix}.value.weight",
                    (HIDDEN_SIZE, KEY_VALUE_HEADS * HEAD_SIZE),
                ),
                source_weight(
                    source_initializers,
                    names["output"],
                    f"{prefix}.output.weight",
                    (HIDDEN_SIZE, HIDDEN_SIZE),
                ),
                source_weight(
                    source_initializers,
                    names["post_norm"],
                    f"{prefix}.post_norm.weight",
                    (HIDDEN_SIZE,),
                ),
                source_weight(
                    source_initializers,
                    names["gate"],
                    f"{prefix}.gate.weight",
                    (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                ),
                source_weight(
                    source_initializers,
                    names["up"],
                    f"{prefix}.up.weight",
                    (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                ),
                source_weight(
                    source_initializers,
                    names["down"],
                    f"{prefix}.down.weight",
                    (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                ),
            ]
        )

    hidden = add_node(
        nodes,
        "Gather",
        ["token_embedding", "input_ids"],
        "embedding.output",
        axis=0,
    )
    cosine = add_node(
        nodes,
        "Gather",
        ["rope.cosine", "position_ids"],
        "rope.cosine.selected",
        axis=0,
    )
    cosine = add_node(
        nodes,
        "Unsqueeze",
        [cosine, "rope.unsqueeze_axes"],
        "rope.cosine.output",
    )
    sine = add_node(
        nodes,
        "Gather",
        ["rope.sine", "position_ids"],
        "rope.sine.selected",
        axis=0,
    )
    sine = add_node(
        nodes,
        "Unsqueeze",
        [sine, "rope.unsqueeze_axes"],
        "rope.sine.output",
    )

    for layer in range(len(LAYER_WEIGHTS)):
        prefix = f"layer.{layer}"
        residual = hidden
        normalized = rms_norm(
            nodes,
            hidden,
            f"{prefix}.input_norm.weight",
            f"{prefix}.input_norm",
        )
        query = add_node(
            nodes,
            "MatMul",
            [normalized, f"{prefix}.query.weight"],
            f"{prefix}.query.projected",
        )
        key = add_node(
            nodes,
            "MatMul",
            [normalized, f"{prefix}.key.weight"],
            f"{prefix}.key.projected",
        )
        value = add_node(
            nodes,
            "MatMul",
            [normalized, f"{prefix}.value.weight"],
            f"{prefix}.value.projected",
        )
        query = add_node(
            nodes,
            "Reshape",
            [query, "query.shape"],
            f"{prefix}.query.reshaped",
        )
        key = add_node(
            nodes,
            "Reshape",
            [key, "key_value.shape"],
            f"{prefix}.key.reshaped",
        )
        value = add_node(
            nodes,
            "Reshape",
            [value, "key_value.shape"],
            f"{prefix}.value.reshaped",
        )
        query = add_node(
            nodes,
            "Transpose",
            [query],
            f"{prefix}.query.transposed",
            perm=[0, 2, 1, 3],
        )
        key = add_node(
            nodes,
            "Transpose",
            [key],
            f"{prefix}.key.transposed",
            perm=[0, 2, 1, 3],
        )
        value = add_node(
            nodes,
            "Transpose",
            [value],
            f"{prefix}.value.transposed",
            perm=[0, 2, 1, 3],
        )
        query = apply_rope(nodes, query, cosine, sine, f"{prefix}.query.rope")
        key = apply_rope(nodes, key, cosine, sine, f"{prefix}.key.rope")
        key = add_node(
            nodes,
            "Gather",
            [key, "key_value.head_indices"],
            f"{prefix}.key.repeated",
            axis=1,
        )
        value = add_node(
            nodes,
            "Gather",
            [value, "key_value.head_indices"],
            f"{prefix}.value.repeated",
            axis=1,
        )
        key = add_node(
            nodes,
            "Transpose",
            [key],
            f"{prefix}.key.for_scores",
            perm=[0, 1, 3, 2],
        )
        scores = add_node(
            nodes,
            "MatMul",
            [query, key],
            f"{prefix}.attention.scores",
        )
        scores = add_node(
            nodes,
            "Mul",
            [scores, "attention.scale"],
            f"{prefix}.attention.scaled",
        )
        scores = add_node(
            nodes,
            "Add",
            [scores, "attention_mask"],
            f"{prefix}.attention.masked",
        )
        probabilities = add_node(
            nodes,
            "Softmax",
            [scores],
            f"{prefix}.attention.probabilities",
            axis=-1,
        )
        attention = add_node(
            nodes,
            "MatMul",
            [probabilities, value],
            f"{prefix}.attention.context",
        )
        attention = add_node(
            nodes,
            "Transpose",
            [attention],
            f"{prefix}.attention.transposed",
            perm=[0, 2, 1, 3],
        )
        attention = add_node(
            nodes,
            "Reshape",
            [attention, "attention_output.shape"],
            f"{prefix}.attention.reshaped",
        )
        attention = add_node(
            nodes,
            "MatMul",
            [attention, f"{prefix}.output.weight"],
            f"{prefix}.attention.output",
        )
        hidden = add_node(
            nodes,
            "Add",
            [residual, attention],
            f"{prefix}.attention.residual",
        )

        residual = hidden
        normalized = rms_norm(
            nodes,
            hidden,
            f"{prefix}.post_norm.weight",
            f"{prefix}.post_norm",
        )
        gate = add_node(
            nodes,
            "MatMul",
            [normalized, f"{prefix}.gate.weight"],
            f"{prefix}.mlp.gate",
        )
        activated = add_node(
            nodes,
            "Sigmoid",
            [gate],
            f"{prefix}.mlp.sigmoid",
        )
        activated = add_node(
            nodes,
            "Mul",
            [gate, activated],
            f"{prefix}.mlp.activated",
        )
        up = add_node(
            nodes,
            "MatMul",
            [normalized, f"{prefix}.up.weight"],
            f"{prefix}.mlp.up",
        )
        hidden = add_node(
            nodes,
            "Mul",
            [activated, up],
            f"{prefix}.mlp.product",
        )
        hidden = add_node(
            nodes,
            "MatMul",
            [hidden, f"{prefix}.down.weight"],
            f"{prefix}.mlp.output",
        )
        hidden = add_node(
            nodes,
            "Add",
            [residual, hidden],
            f"{prefix}.output",
        )

    hidden = rms_norm(nodes, hidden, "final_norm.weight", "final_norm")
    embedding_transposed = add_node(
        nodes,
        "Transpose",
        ["token_embedding"],
        "token_embedding.transposed",
        perm=[1, 0],
    )
    add_node(nodes, "MatMul", [hidden, embedding_transposed], "logits")

    input_ids = helper.make_tensor_value_info(
        "input_ids",
        TensorProto.INT64,
        [1, "sequence_length"],
    )
    position_ids = helper.make_tensor_value_info(
        "position_ids",
        TensorProto.INT64,
        [1, "sequence_length"],
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask",
        TensorProto.FLOAT,
        [1, 1, "sequence_length", "sequence_length"],
    )
    logits = helper.make_tensor_value_info(
        "logits",
        TensorProto.FLOAT,
        [1, "sequence_length", VOCABULARY_SIZE],
    )
    graph = helper.make_graph(
        nodes,
        "tinystories-656k-no-cache",
        [input_ids, position_ids, attention_mask],
        [logits],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="graft-tinystories-example",
    )
    model.ir_version = 8
    helper.set_model_props(
        model,
        {
            "source": "onnx-community/TinyStories-656K-ONNX",
            "source_revision": REVISION,
            "adaptation": "tract-friendly no-cache graph",
        },
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, target_path)


def main():
    directory = Path(__file__).parent
    source_model = download(MODEL_URL, MODEL_SHA256)
    tokenizer = download(TOKENIZER_URL, TOKENIZER_SHA256)
    prepared_model = None
    try:
        with NamedTemporaryFile(
            dir=directory,
            prefix=".model.onnx.",
            delete=False,
        ) as temporary:
            prepared_model = Path(temporary.name)
        build_model(source_model, prepared_model)
        prepared_model.replace(directory / "model.onnx")
        prepared_model = None
        tokenizer.replace(directory / "tokenizer.json")
        tokenizer = None
        print("model.onnx and tokenizer.json: updated")
    finally:
        source_model.unlink(missing_ok=True)
        if tokenizer is not None:
            tokenizer.unlink(missing_ok=True)
        if prepared_model is not None:
            prepared_model.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
