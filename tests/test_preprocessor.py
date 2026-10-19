import unittest
from src.preprocessor import (
    strip_comments_preserve_offsets,
    filter_syntactic_findings,
    is_pure_punctuation,
)


class TestPreprocessor(unittest.TestCase):
    def test_strip_c_single_and_multi_line_comments(self):
        code = (
            "// Top comment\n"
            "/* Multi\n"
            "   line */\n"
            "int add(int a, int b) {\n"
            "    return a + b; // inline comment\n"
            "}\n"
        )
        clean, stats = strip_comments_preserve_offsets(code)
        self.assertNotIn("Top comment", clean)
        self.assertNotIn("Multi", clean)
        self.assertNotIn("inline comment", clean)
        self.assertIn("int add(int a, int b)", clean)
        self.assertEqual(code.count("\n"), clean.count("\n"))
        self.assertEqual(stats["comments_found"], 3)

    def test_preserve_string_literals_with_comment_markers(self):
        code = (
            'char *url = "http://example.com/api#section";\n'
            'char *query = "SELECT /* not a comment */ * FROM t";\n'
            '// Real comment\n'
        )
        clean, stats = strip_comments_preserve_offsets(code)
        self.assertIn('"http://example.com/api#section"', clean)
        self.assertIn('"SELECT /* not a comment */ * FROM t"', clean)
        self.assertNotIn("Real comment", clean)
        self.assertEqual(stats["comments_found"], 1)

    def test_python_docstrings_and_comments(self):
        code = (
            '"""Module docstring"""\n'
            'def foo():\n'
            '    # A comment\n'
            '    return 42\n'
        )
        clean, stats = strip_comments_preserve_offsets(code)
        self.assertNotIn("Module docstring", clean)
        self.assertNotIn("A comment", clean)
        self.assertIn("def foo():", clean)

    def test_empty_syntax_raises_error(self):
        comment_only = (
            "// Just a comment\n"
            "/* Another comment */\n"
            "# Python comment\n"
        )
        with self.assertRaises(ValueError):
            strip_comments_preserve_offsets(comment_only)

    def test_is_pure_punctuation(self):
        self.assertTrue(is_pure_punctuation(";"))
        self.assertTrue(is_pure_punctuation("();"))
        self.assertTrue(is_pure_punctuation("{ }"))
        self.assertTrue(is_pure_punctuation("->"))
        self.assertFalse(is_pure_punctuation("strcpy"))
        self.assertFalse(is_pure_punctuation("buf"))
        self.assertFalse(is_pure_punctuation("SELECT"))

    def test_filter_syntactic_findings(self):
        raw_findings = [
            {"text": ";", "score": 1.0, "line": 1},
            {"text": "(", "score": 0.9, "line": 2},
            {"text": "strcpy", "score": 0.8, "line": 3},
            {"text": "include", "score": 0.7, "line": 1},
            {"text": "buffer", "score": 0.6, "line": 3},
        ]
        filtered = filter_syntactic_findings(
            raw_findings,
            top_k=2,
            filter_punctuation=True,
            filter_boilerplate_headers=True,
        )
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["text"], "strcpy")
        self.assertEqual(filtered[1]["text"], "buffer")
        # Relative score should be re-normalized with top token = 1.0
        self.assertEqual(filtered[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
