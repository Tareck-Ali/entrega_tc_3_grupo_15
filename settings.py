from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TRAIN_DATA = BASE_DIR / "data" / "medical_tc_train.csv"
TEST_DATA = BASE_DIR / "data" / "medical_tc_test.csv"

MODELS = BASE_DIR / "models"

BASELINE_MODEL = MODELS / "baseline.joblib"
ONNX_MODEL = MODELS / "baseline.onnx"