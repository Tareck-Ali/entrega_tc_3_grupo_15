import joblib

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

import settings


def convert():
    model = joblib.load(settings.BASELINE_MODEL)

    initial_type = [
        ("text", StringTensorType([None, 1]))
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
    )

    with open(settings.ONNX_MODEL, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved to {settings.ONNX_MODEL}")


if __name__ == "__main__":
    convert()