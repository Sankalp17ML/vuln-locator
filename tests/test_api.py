import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["specialist_available"])

    def test_models_endpoint(self):
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("specialist", data)
        self.assertIn("generalist", data)
        self.assertTrue(data["specialist"]["available"])

    def test_predict_endpoint_valid(self):
        payload = {
            "code": "def fetch(val):\n    return cursor.execute('SELECT * FROM data WHERE id=' + val)",
            "top_k": 5,
            "include_localization": True,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vulnerable", data)
        self.assertIn("confidence", data)
        self.assertIn("findings", data)
        self.assertIn("model_used", data)
        self.assertIn("detected_language", data)
        self.assertIn("routing_reason", data)
        self.assertLessEqual(len(data["findings"]), 5)

    def test_predict_routes_c_to_specialist(self):
        payload = {
            "code": "void copy(char *s) { char buf[10]; strcpy(buf, s); }",
            "top_k": 5,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Specialist", data["model_used"])

    def test_predict_endpoint_empty(self):
        response = self.client.post("/predict", json={"code": "   "})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
