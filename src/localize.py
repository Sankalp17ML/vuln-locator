import argparse
from pathlib import Path

from src.inference import CodeAnalyzer


DEMO_CODE = """\
def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify code and print the highest-attention source tokens."
    )
    parser.add_argument("--file", type=Path, help="Read code from a source file.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of tokens to show.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold.")
    parser.add_argument(
        "--keep-comments",
        action="store_true",
        help="Do not strip comments (default: comments are stripped to prevent lexical bias).",
    )
    parser.add_argument(
        "--keep-punct",
        action="store_true",
        help="Do not filter out structural punctuation tokens in findings.",
    )
    parser.add_argument(
        "--strip-headers",
        action="store_true",
        help="Strip C/C++ boilerplate #include headers to reduce positional bias.",
    )
    args = parser.parse_args()

    code = args.file.read_text(encoding="utf-8") if args.file else DEMO_CODE
    result = CodeAnalyzer(threshold=args.threshold).analyze(
        code,
        top_k=args.top_k,
        strip_comments=not args.keep_comments,
        filter_punctuation=not args.keep_punct,
        strip_headers=args.strip_headers,
    )
    print(
        f"Prediction: {result['label']} "
        f"(confidence: {result['confidence']:.1%}, "
        f"vulnerable probability: {result['vulnerable_probability']:.1%})"
    )
    if result.get("comments_stripped", 0) > 0:
        print(f"Preprocessed: stripped {result['comments_stripped']} comments ({result['comment_chars_removed']} chars) to rely purely on code syntax.")
    print("Top suspicious tokens (attention rollout):")
    for finding in result["findings"]:
        print(
            f"  line {finding['line']:<4} {finding['text']!r:<20} "
            f"relative attention: {finding['score']:.2f}"
        )


if __name__ == "__main__":
    main()