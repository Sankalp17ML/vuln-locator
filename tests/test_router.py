import unittest
from src.router import detect_language, LanguageRouter


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.router = LanguageRouter()

    def test_detect_c_cpp(self):
        code = '#include <stdio.h>\nvoid fn(char *p) { char *b = malloc(10); strcpy(b, p); }'
        lang, score = detect_language(code)
        self.assertIn(lang, ("c", "cpp"))
        self.assertGreater(score, 0)

    def test_detect_python(self):
        code = 'import os\ndef run_cmd(user_arg):\n    return os.system("ping " + user_arg)'
        lang, score = detect_language(code)
        self.assertEqual(lang, "python")
        self.assertGreater(score, 0)

    def test_detect_java(self):
        code = 'import java.sql.*;\npublic class DBService {\n    public void query(String q) { System.out.println(q); }\n}'
        lang, score = detect_language(code)
        self.assertEqual(lang, "java")
        self.assertGreater(score, 0)

    def test_detect_php(self):
        code = '<?php\n$host = $_GET["host"];\nsystem("ping -c 1 " . $host);\n?>'
        lang, score = detect_language(code)
        self.assertEqual(lang, "php")
        self.assertGreater(score, 0)

    def test_detect_javascript(self):
        code = 'const { exec } = require("child_process");\nfunction run(cmd) {\n    console.log(cmd);\n}'
        lang, score = detect_language(code)
        self.assertEqual(lang, "javascript")
        self.assertGreater(score, 0)

    def test_route_c_to_specialist(self):
        code = 'char *p = malloc(100); free(p);'
        decision = self.router.route(code)
        self.assertEqual(decision["model_key"], "specialist")
        self.assertIn("Specialist", decision["model_name"])

    def test_route_java_to_generalist(self):
        code = 'import java.io.*;\npublic class Test {}'
        decision = self.router.route(code)
        self.assertEqual(decision["model_key"], "generalist")
        self.assertIn("Generalist", decision["model_name"])

    def test_forced_model_override(self):
        code = 'def python_fn(): pass'
        decision = self.router.route(code, forced_model="specialist")
        self.assertEqual(decision["model_key"], "specialist")


if __name__ == "__main__":
    unittest.main()
