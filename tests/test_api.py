from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "SIFT"
    assert data["status"] == "running"


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You won a free prize!"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "spam_probability" in data

    assert data["prediction"] in ["spam", "ham"]

    assert 0 <= data["spam_probability"] <= 1