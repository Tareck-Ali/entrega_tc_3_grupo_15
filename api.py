import onnxruntime as ort

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

import settings


app = FastAPI(title="Medical Abstract Classifier")


session = ort.InferenceSession(
    str(settings.ONNX_MODEL)
)

input_name = session.get_inputs()[0].name


class PredictionRequest(BaseModel):
    abstract: str


@app.post("/predict")
def predict(request: PredictionRequest):
    result = session.run(
        None,
        {input_name: [request.abstract]},
    )

    return {
        "prediction": result[0][0]
    }


Instrumentator().instrument(app).expose(app)
