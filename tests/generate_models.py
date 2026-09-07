#!/usr/bin/env python3
"""Regenerate the small ONNX fixtures used by Go integration tests."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent / "models"
OPSET = [helper.make_opsetid("", 13)]


def save(name, nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=OPSET, producer_name="graft-tests")
    onnx.checker.check_model(model)
    onnx.save_model(model, ROOT / name)


def value(name, data_type=TensorProto.FLOAT, shape=(2, 3)):
    return helper.make_tensor_value_info(name, data_type, list(shape))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    save(
        "identity.onnx",
        [helper.make_node("Identity", ["x"], ["y"])],
        [value("x")],
        [value("y")],
    )

    bias = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), "bias")
    save(
        "add.onnx",
        [helper.make_node("Add", ["x", "bias"], ["y"])],
        [value("x")],
        [value("y")],
        [bias],
    )

    weights = numpy_helper.from_array(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32), "weights"
    )
    save(
        "matmul.onnx",
        [helper.make_node("MatMul", ["x", "weights"], ["y"])],
        [value("x")],
        [value("y", shape=(2, 2))],
        [weights],
    )

    kernel = numpy_helper.from_array(np.ones((1, 1, 2, 2), dtype=np.float32), "kernel")
    save(
        "conv.onnx",
        [helper.make_node("Conv", ["x", "kernel"], ["y"])],
        [value("x", shape=(1, 1, 3, 3))],
        [value("y", shape=(1, 1, 2, 2))],
        [kernel],
    )

    save(
        "multi_input.onnx",
        [helper.make_node("Add", ["left", "right"], ["sum"])],
        [value("left", shape=(2,)), value("right", shape=(2,))],
        [value("sum", shape=(2,))],
    )

    save(
        "multi_output.onnx",
        [
            helper.make_node("Identity", ["x"], ["same"]),
            helper.make_node("Neg", ["x"], ["negative"]),
        ],
        [value("x", shape=(2,))],
        [value("same", shape=(2,)), value("negative", shape=(2,))],
    )

    save(
        "dynamic_shape.onnx",
        [helper.make_node("Identity", ["x"], ["y"])],
        [value("x", shape=("batch", 3))],
        [value("y", shape=("batch", 3))],
    )

    type_specs = [
        ("f32", TensorProto.FLOAT),
        ("f64", TensorProto.DOUBLE),
        ("i32", TensorProto.INT32),
        ("i64", TensorProto.INT64),
        ("u8", TensorProto.UINT8),
        ("bool", TensorProto.BOOL),
    ]
    type_inputs = [value(name, data_type, (2,)) for name, data_type in type_specs]
    type_outputs = [value(f"{name}_out", data_type, (2,)) for name, data_type in type_specs]
    type_nodes = [helper.make_node("Identity", [name], [f"{name}_out"]) for name, _ in type_specs]
    save("types.onnx", type_nodes, type_inputs, type_outputs)


if __name__ == "__main__":
    main()
