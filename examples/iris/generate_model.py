#!/usr/bin/env python3
"""Generate the softmax linear classifier used by the Iris example."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# These fixed coefficients come from multinomial logistic regression fitted to
# standardized Iris data. The standardization has been folded into the weights
# and bias so the ONNX graph accepts the four raw measurements directly.
WEIGHTS = np.array(
    [
        [-1.3013932, 0.7122254, 0.58916783],
        [2.6704285, -0.8329066, -1.8375219],
        [-1.0973196, -0.20704629, 1.3043660],
        [-2.3847651, -1.0883435, 3.4731085],
    ],
    dtype=np.float32,
)
BIAS = np.array([6.2186236, 1.8043755, -8.0230000], dtype=np.float32)


def main():
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, [3, 4])
    probabilities = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, [3, 3]
    )

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["products"]),
            helper.make_node("Add", ["products", "bias"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "iris-linear-classifier",
        [features],
        [probabilities],
        [
            numpy_helper.from_array(WEIGHTS, "weights"),
            numpy_helper.from_array(BIAS, "bias"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="graft-iris-example",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save_model(model, Path(__file__).with_name("iris.onnx"))


if __name__ == "__main__":
    main()
