# Iris linear classifier example

This example classifies three Iris flowers from sepal length, sepal width,
petal length, and petal width. The ONNX graph is a multinomial linear model:

```text
features -> MatMul(weights) -> Add(bias) -> Softmax -> probabilities
```

Run it from the repository root:

```bash
go run ./examples/iris
```

Expected classifications:

```text
[5.1 3.5 1.4 0.2] -> setosa
[5.9 3 4.2 1.5]   -> versicolor
[6.5 3 5.8 2.2]   -> virginica
```

The four values are measured in centimeters. This small model demonstrates
multiclass inference; it is not intended as an accuracy benchmark.

`iris.onnx` is embedded in the example. To regenerate it, install Python
`numpy` and `onnx`, then run:

```bash
python examples/iris/generate_model.py
```
