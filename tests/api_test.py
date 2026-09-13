from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_predict():
    response = client.post(
        "/predict",
        json={
            "abstract": "Patient presents with a medical condition."
        },
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
