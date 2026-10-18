import unittest
from src.inference import CodeAnalyzer, DEFAULT_MODEL_PATH


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = CodeAnalyzer(model_path=DEFAULT_MODEL_PATH)

    def test_analyzer_initialization(self):
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.tokenizer)
        self.assertTrue(0.0 <= self.analyzer.threshold <= 1.0)

    def test_analyzer_predict_structure(self):
        code = (
            'def get_user(name):\n'
            '    query = "SELECT * FROM users WHERE name = \'" + name + "\'"\n'
            '    cursor.execute(query)\n'
        )
        result = self.analyzer.analyze(code, top_k=5, include_localization=True)

        self.assertIn("vulnerable", result)
        self.assertIsInstance(result["vulnerable"], bool)
        self.assertIn(result["label"], ("Vulnerable", "Safe"))
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertTrue(0.0 <= result["vulnerable_probability"] <= 1.0)
        self.assertLessEqual(len(result["findings"]), 5)

        if result["findings"]:
            finding = result["findings"][0]
            self.assertIn("token", finding)
            self.assertIn("text", finding)
            self.assertIn("score", finding)
            self.assertIn("line", finding)
            self.assertTrue(0.0 <= finding["score"] <= 1.0)

    def test_analyzer_empty_input_raises(self):
        with self.assertRaises(ValueError):
            self.analyzer.analyze("   ")

    def test_analyzer_invalid_threshold(self):
        with self.assertRaises(ValueError):
            CodeAnalyzer(threshold=1.5)

    def test_analyzer_batch_probabilities(self):
        samples = [
            "def safe_fn(x): return x + 1",
            "char dest[10]; strcpy(dest, input);",
        ]
        probs = self.analyzer.predict_probabilities(samples, batch_size=2)
        self.assertEqual(len(probs), 2)
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)


if __name__ == "__main__":
    unittest.main()
