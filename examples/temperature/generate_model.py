#!/usr/bin/env python3
"""Generate the ONNX graph used by the temperature example."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    celsius = helper.make_tensor_value_info("celsius", TensorProto.FLOAT, [4])
    fahrenheit = helper.make_tensor_value_info("fahrenheit", TensorProto.FLOAT, [4])
    scale = numpy_helper.from_array(np.array([1.8], dtype=np.float32), "scale")
    offset = numpy_helper.from_array(np.array([32.0], dtype=np.float32), "offset")

    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["celsius", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "offset"], ["fahrenheit"]),
        ],
        "celsius-to-fahrenheit",
        [celsius],
        [fahrenheit],
        [scale, offset],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="graft-temperature-example",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save_model(model, Path(__file__).with_name("temperature.onnx"))


if __name__ == "__main__":
    main()
