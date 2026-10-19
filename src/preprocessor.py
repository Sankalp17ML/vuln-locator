"""
Code Syntax Preprocessor & Lexical Bias Mitigation Engine
=========================================================
Strips natural language comments, docstrings, and non-executable artifacts
to ensure deep learning models (CodeBERT) and Attention Rollout evaluate pure
code syntax rather than relying on natural language comment cues or lexical bias.

Key Features:
1. Strict string literal preservation (escaped quotes, URLs, SQL queries containing # or //).
2. Space-filling replacement: preserves line numbers and offsets 1:1 with the original code.
3. Header / preprocessor handling (#include, #define) to mitigate attention sink bias.
4. Syntactic token filtering for Attention Rollout: removes pure punctuation delimiters
   (; , { } [ ] ( ) : . #) so findings highlight actual function names, variables, and keywords.
"""

import re
from typing import Any

# Punctuation set considered structural delimiters rather than syntactic identifiers
STRUCTURAL_PUNCTUATION = set(";,(){}[]:.<>#*&+-=!|/\\%^~'\"` \t\n\r")
BOILERPLATE_DIRECTIVES = {"include", "stdio", "stdlib", "string", "iostream", "h"}


def strip_comments_preserve_offsets(
    code: str,
    strip_comments: bool = True,
    strip_headers: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Strips single-line, multi-line, docstring, and hash comments from source code.
    Replaces non-newline comment characters with whitespace so that the total character
    count and newline positions remain 100% synchronized with the original source.
    """
    if not code:
        return "", {"comment_chars_removed": 0, "comments_found": 0, "headers_removed": 0}

    # Regex matches:
    # 1. Triple-quoted block strings ("""...""" or '''...''')
    # 2. Double-quoted string literals with escapes ("...")
    # 3. Single-quoted string literals with escapes ('...')
    # 4. Multi-line C/C++/Java block comments (/*...*/)
    # 5. Single-line C/C++/Java comments (//...)
    # 6. Python/Shell hash comments or C preprocessor directives (#...)
    token_pattern = re.compile(
        r"""
        (
            \"\"\"[\s\S]*?\"\"\"                   # Triple double-quoted string
          | \x27\x27\x27[\s\S]*?\x27\x27\x27       # Triple single-quoted string
          | \"(?:\\.|[^\"\\\n])*\"                 # Double-quoted string literal
          | \x27(?:\\.|[^\x27\\\n])*\x27           # Single-quoted string literal
          | /\*[\s\S]*?\*/                         # C-style block comment
          | //[^\n]*                               # C-style line comment
          | \#[^\n]*                               # Hash comment or preprocessor
        )
        """,
        re.VERBOSE,
    )

    stats = {
        "comment_chars_removed": 0,
        "comments_found": 0,
        "headers_removed": 0,
    }

    def replacer(match: re.Match) -> str:
        token = match.group(0)

        # Multi-line comment: /* ... */
        if token.startswith("/*"):
            stats["comments_found"] += 1
            stats["comment_chars_removed"] += len(token)
            # Replace all characters except newlines with spaces to keep line numbers intact
            return re.sub(r"[^\n]", " ", token)

        # Single-line comment: // ...
        if token.startswith("//"):
            stats["comments_found"] += 1
            stats["comment_chars_removed"] += len(token)
            return re.sub(r"[^\n]", " ", token)

        # Hash comments or preprocessors: # ...
        if token.startswith("#"):
            is_preprocessor = bool(
                re.match(r"^#(?:include|define|ifdef|ifndef|endif|pragma)\b", token)
            )
            if is_preprocessor:
                if strip_headers and token.startswith("#include"):
                    stats["headers_removed"] += 1
                    return re.sub(r"[^\n]", " ", token)
                return token

            # Regular Python / Bash / Shell comment
            stats["comments_found"] += 1
            stats["comment_chars_removed"] += len(token)
            return re.sub(r"[^\n]", " ", token)

        # Python standalone docstrings
        if (token.startswith('"""') or token.startswith("'''")) and strip_comments:
            # Check if this block looks like a docstring (multi-line comment)
            stats["comments_found"] += 1
            stats["comment_chars_removed"] += len(token)
            return re.sub(r"[^\n]", " ", token)

        # String literal: preserve intact
        return token

    processed_code = token_pattern.sub(replacer, code) if strip_comments else code

    # Check for empty syntax edge case
    non_whitespace = processed_code.strip()
    if code.strip() and not non_whitespace:
        raise ValueError(
            "Code input contains no executable syntax (only comments or whitespace were found)."
        )

    return processed_code, stats


def is_pure_punctuation(text: str) -> bool:
    """Returns True if the text consists exclusively of structural punctuation/whitespace."""
    stripped = text.strip()
    if not stripped:
        return True
    return all(char in STRUCTURAL_PUNCTUATION for char in stripped)


def filter_syntactic_findings(
    findings: list[dict[str, Any]],
    top_k: int = 10,
    filter_punctuation: bool = True,
    filter_boilerplate_headers: bool = True,
) -> list[dict[str, Any]]:
    """
    Filters raw attention findings to focus on meaningful syntax:
    identifiers, function names, operators, variables, and keywords.
    Removes attention sink delimiters (;, {, }, (, ), [, ]).
    """
    if not filter_punctuation and not filter_boilerplate_headers:
        return findings[:top_k]

    filtered = []
    for item in findings:
        txt = item["text"].strip()
        if not txt:
            continue

        if filter_punctuation and is_pure_punctuation(txt):
            continue

        if filter_boilerplate_headers:
            clean_token = re.sub(r"[^a-zA-Z0-9_]", "", txt).lower()
            if clean_token in BOILERPLATE_DIRECTIVES:
                continue

        filtered.append(item)

    if not filtered:
        # Fallback to original findings if all were filtered
        return findings[:top_k]

    # Re-normalize scores relative to the top retained syntactic token
    max_score = max(item["score"] for item in filtered)
    if max_score > 0:
        for item in filtered:
            item["score"] = round(item["score"] / max_score, 4)

    return filtered[:top_k]
