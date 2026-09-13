import onnxruntime as ort

import settings


def test_onnx_model_loads():
    session = ort.InferenceSession(
        str(settings.ONNX_MODEL)
    )

    assert session.get_inputs()
    assert session.get_outputs()