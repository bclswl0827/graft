#!/usr/bin/env python3
"""Download the model and character dictionary used by the OCR example."""

from hashlib import sha256
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import onnx

MODEL_NAME = "model.onnx"
MODEL_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx"
MODEL_SOURCE_DIGEST = (
    "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b"
)
MODEL_SPECIALIZED_DIGEST = (
    "03de797f8d61d68c634e9d45558191770b241741b48717a87b063dd7f0c94cb9"
)
MODEL_INPUT_SHAPE = (1, 3, 48, 320)

DICTIONARY_NAME = "ppocr_keys_v1.txt"
DICTIONARY_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/8cce9b6fd7ccb50226d0c38f94054d81c29b8184/ppocr/utils/ppocr_keys_v1.txt"
DICTIONARY_DIGEST = (
    "28b2362ad4ab2dc38769aa72feb535e3a9ddb3fd2a7585a05920e6393b1dc7f7"
)


def digest(path):
    checksum = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(directory, name, url, expected_digest):
    request = Request(url, headers={"User-Agent": "graft-ocr-example"})
    temporary_path = None
    try:
        with urlopen(request) as response, NamedTemporaryFile(
            dir=directory, prefix=".download.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            copyfileobj(response, temporary)
        actual_digest = digest(temporary_path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"{name}: SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
            )
        return temporary_path
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise


def prepare_model(directory):
    destination = directory / MODEL_NAME
    current_digest = digest(destination) if destination.is_file() else None
    if current_digest == MODEL_SPECIALIZED_DIGEST:
        print(f"{MODEL_NAME}: already downloaded and specialized")
        return

    source_path = None
    specialized_path = None
    try:
        if current_digest == MODEL_SOURCE_DIGEST:
            source_path = destination
        else:
            source_path = download(
                directory, MODEL_NAME, MODEL_URL, MODEL_SOURCE_DIGEST
            )

        with NamedTemporaryFile(
            dir=directory, prefix=".specialized.", suffix=".onnx", delete=False
        ) as temporary:
            specialized_path = Path(temporary.name)

        model = onnx.load(source_path, load_external_data=False)
        if len(model.graph.input) != 1:
            raise RuntimeError(
                f"{MODEL_NAME}: expected one input, got {len(model.graph.input)}"
            )
        dimensions = model.graph.input[0].type.tensor_type.shape.dim
        if len(dimensions) != len(MODEL_INPUT_SHAPE):
            raise RuntimeError(
                f"{MODEL_NAME}: expected rank {len(MODEL_INPUT_SHAPE)}, "
                f"got {len(dimensions)}"
            )
        for dimension, value in zip(dimensions, MODEL_INPUT_SHAPE):
            dimension.ClearField("dim_param")
            dimension.dim_value = value

        onnx.checker.check_model(model)
        onnx.save_model(model, specialized_path)
        actual_digest = digest(specialized_path)
        if actual_digest != MODEL_SPECIALIZED_DIGEST:
            raise RuntimeError(
                f"{MODEL_NAME}: specialized SHA-256 mismatch: expected "
                f"{MODEL_SPECIALIZED_DIGEST}, got {actual_digest}"
            )
        specialized_path.replace(destination)
        print(f"{MODEL_NAME}: downloaded and specialized to {MODEL_INPUT_SHAPE}")
    finally:
        if source_path is not None and source_path != destination:
            source_path.unlink(missing_ok=True)
        if specialized_path is not None:
            specialized_path.unlink(missing_ok=True)


def prepare_dictionary(directory):
    destination = directory / DICTIONARY_NAME
    if destination.is_file() and digest(destination) == DICTIONARY_DIGEST:
        print(f"{DICTIONARY_NAME}: already downloaded")
        return
    temporary_path = download(
        directory, DICTIONARY_NAME, DICTIONARY_URL, DICTIONARY_DIGEST
    )
    temporary_path.replace(destination)
    print(f"{DICTIONARY_NAME}: downloaded")


def main():
    directory = Path(__file__).resolve().parent
    prepare_model(directory)
    prepare_dictionary(directory)


if __name__ == "__main__":
    main()
