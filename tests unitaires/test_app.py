from fastapi.testclient import TestClient
from main import app, predict

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "Welcome to Score Prediction API."}

def test_predict():
    response = client.post("/predict", json={"customer_id": 100001})
    assert response.status_code == 200
    assert "score" in response.json()
    assert isinstance(response.json()["score"], float) or response.json()["score"] == -1

def test_predict_invalid_customer_id():
    response = client.post("/predict", json={"customer_id": 999999})
    assert response.status_code == 200
    assert response.json() == {"score": -1}
