from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA = BASE_DIR / "data" / "medical_tc_train.csv"

MODELS = BASE_DIR / "models"

BASELINE_MODEL = MODELS / "baseline.joblib"
ONNX_MODEL = MODELS / "baseline.onnx"