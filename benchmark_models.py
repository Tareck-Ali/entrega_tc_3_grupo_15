import time

import joblib
import onnxruntime as ort
import pandas as pd

import settings


def benchmark():
    df = pd.read_csv(settings.TEST_DATA)

    texts = df["medical_abstract"].fillna("").tolist()

    # Load models before benchmarking.
    joblib_model = joblib.load(settings.BASELINE_MODEL)

    onnx_session = ort.InferenceSession(
        str(settings.ONNX_MODEL)
    )

    onnx_input = onnx_session.get_inputs()[0].name

    # Benchmark Joblib
    start = time.perf_counter()

    joblib_model.predict(texts)

    joblib_time = time.perf_counter() - start

    # Benchmark ONNX
    start = time.perf_counter()

    onnx_session.run(
        None,
        {onnx_input: texts},
    )

    onnx_time = time.perf_counter() - start

    print(f"Test samples: {len(texts)}")
    print(f"Joblib: {joblib_time:.4f}s")
    print(f"ONNX:   {onnx_time:.4f}s")

    print()
    print(
        f"Joblib: {joblib_time / len(texts) * 1000:.2f} ms/sample"
    )

    print(
        f"ONNX:   {onnx_time / len(texts) * 1000:.2f} ms/sample"
    )

    print()
    print(
        f"Speedup: {joblib_time / onnx_time:.2f}x"
    )


if __name__ == "__main__":
    benchmark()