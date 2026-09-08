# PP-OCRv4 text recognition example

This example reads a high-contrast screenshot or document image and prints the
recognized UTF-8 text. It uses RapidOCR's ONNX conversion of PaddleOCR's
Chinese PP-OCRv4 mobile recognition model.

Install the model preparation dependency, then download the pinned model and
character dictionary:

```bash
python3 -m pip install onnx
python3 examples/ocr/download_model.py
```

The upstream model declares dynamic image dimensions that tract cannot fully
resolve while optimizing the graph. The download script specializes its input
to the `[1, 3, 48, 320]` shape used by this example and verifies the resulting
model digest.

Then pass a PNG, JPEG, or GIF image:

```bash
go run ./examples/ocr path/to/text-line.png
```

The example segments a high-contrast image into lines and words, then resizes
each region to a height of 48 pixels, pads it to 320 pixels wide, converts it
to the model's BGR/CHW tensor layout, and normalizes it to `[-1, 1]`. Model
outputs are converted to text with greedy CTC decoding and reassembled with
spaces and line breaks. Use `-model` and `-dict` to override the downloaded
asset paths.

The built-in segmentation targets screenshots and documents with a mostly
uniform background. Natural scenes, rotated text, and complex layouts require
a dedicated text-detection model before recognition.

The model and dictionary are downloaded from
[`RapidAI/RapidOCR`](https://github.com/RapidAI/RapidOCR) and
[`PaddlePaddle/PaddleOCR`](https://github.com/PaddlePaddle/PaddleOCR),
respectively. Both upstream projects distribute these assets under the
Apache-2.0 license. The download script pins the source versions and verifies
their SHA-256 digests.
