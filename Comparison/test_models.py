import joblib
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import classification_report

import settings


def test():
    df = pd.read_csv(settings.TEST_DATA)

    texts = df["medical_abstract"].fillna("").tolist()
    labels = df["condition_label"].tolist()

    # Joblib model
    model = joblib.load(settings.BASELINE_MODEL)

    joblib_predictions = model.predict(texts)

    # ONNX model
    session = ort.InferenceSession(
        str(settings.ONNX_MODEL)
    )

    input_name = session.get_inputs()[0].name

    onnx_predictions = session.run(
        None,
        {input_name: texts},
    )[0]

    print("=== Joblib ===")
    print(classification_report(labels, joblib_predictions))

    print("=== ONNX ===")
    print(classification_report(labels, onnx_predictions))

    print(
        "Predictions match:",
        all(
            joblib_prediction == onnx_prediction
            for joblib_prediction, onnx_prediction
            in zip(joblib_predictions, onnx_predictions)
        ),
    )


if __name__ == "__main__":
    test()