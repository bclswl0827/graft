# Temperature example

This example runs a two-operation ONNX graph that converts four Celsius values
to Fahrenheit:

```text
fahrenheit = celsius * 1.8 + 32
```

Run it from the repository root:

```bash
go run ./examples/temperature
```

Expected output:

```text
0°C = 32°F
10°C = 50°F
20°C = 68°F
100°C = 212°F
```

`temperature.onnx` is embedded in the example. To regenerate it, install
Python `numpy` and `onnx`, then run:

```bash
python examples/temperature/generate_model.py
```
