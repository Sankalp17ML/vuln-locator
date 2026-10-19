"""
Language Router & Mixture-of-Experts (MoE) Engine
=================================================
Implements the Router Architecture for VulnLocator V2.0:
Prevents "Catastrophic Forgetting" by dynamically routing code to the
appropriate expert model:
1. Specialist Model: CodeBERT fine-tuned on C/C++ (PrimeVul) for memory safety,
   pointers, buffer overflows, and low-level system vulnerabilities.
2. Generalist Model: CodeBERT fine-tuned on CrossVul (Java, Python, PHP, JS, etc.)
   for web, API, injection, and higher-level vulnerabilities.
"""

import re
from pathlib import Path
from typing import Any, Literal


SPECIALIST_LANGUAGES = {"c", "cpp"}
GENERALIST_LANGUAGES = {"java", "python", "php", "javascript", "js"}


def detect_language(code: str) -> tuple[str, float]:
    """
    Detects the programming language of a code snippet using syntactic markers.
    Returns (language_code, confidence_score).
    """
    code_clean = code.strip()
    if not code_clean:
        return "unknown", 0.0

    scores = {
        "c": 0,
        "cpp": 0,
        "php": 0,
        "java": 0,
        "javascript": 0,
        "python": 0,
    }

    # 1. PHP detection
    if "<?php" in code_clean or "<?" in code_clean or re.search(r"\$(?:_GET|_POST|_REQUEST|_SERVER)\b", code_clean):
        scores["php"] += 15
    if re.search(r"\$[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*", code_clean):
        scores["php"] += 4

    # 2. C / C++ detection
    if re.search(r"#\s*include\b", code_clean):
        scores["c"] += 8
        scores["cpp"] += 8
    if re.search(r"\b(?:malloc|calloc|realloc|free|sizeof|kzalloc|kfree)\s*\(", code_clean):
        scores["c"] += 8
        scores["cpp"] += 6
    if re.search(r"\b(?:printf|sprintf|snprintf|strcpy|strncpy|strcat|strncat|memcpy|memset)\s*\(", code_clean):
        scores["c"] += 6
        scores["cpp"] += 5
    if re.search(r"\b(?:struct|typedef\s+struct|size_t|ssize_t|uint32_t|int32_t|char\s*\*|void\s*\*)\b", code_clean):
        scores["c"] += 6
        scores["cpp"] += 5
    if "std::" in code_clean or "cout <<" in code_clean or "nullptr" in code_clean or "template<" in code_clean:
        scores["cpp"] += 12

    # 3. Java detection
    if re.search(r"\bimport\s+(?:java|javax)\b", code_clean):
        scores["java"] += 12
    if re.search(r"\b(?:public|private|protected)\s+(?:class|interface|enum|void|int|String|boolean)\b", code_clean):
        scores["java"] += 8
    if "System.out.print" in code_clean or "@Override" in code_clean or "throws Exception" in code_clean:
        scores["java"] += 8

    # 4. Python detection
    if re.search(r"\bdef\s+[a-zA-Z_]\w*\s*\(.*?\)\s*:", code_clean):
        scores["python"] += 12
    if re.search(r"\b(?:elif|except|finally)\b\s*:", code_clean):
        scores["python"] += 7
    if re.search(r"\bfrom\s+[a-zA-Z_]\w*\s+import\b|\bimport\s+[a-zA-Z_]\w*\b", code_clean):
        scores["python"] += 5
    if re.search(r"\b(?:self|None|True|False)\b", code_clean):
        scores["python"] += 4

    # 5. JavaScript detection
    if re.search(r"\b(?:const|let|var)\s+[a-zA-Z_$]\w*\s*=", code_clean):
        scores["javascript"] += 7
    if "console.log" in code_clean or "module.exports" in code_clean or re.search(r"\brequire\s*\(", code_clean):
        scores["javascript"] += 8
    if re.search(r"=>|\b===|\b!==\b", code_clean):
        scores["javascript"] += 5

    # Group C and C++
    top_lang, top_score = max(scores.items(), key=lambda item: item[1])
    if top_score == 0:
        return "general", 0.0

    return top_lang, float(top_score)


class LanguageRouter:
    """
    Mixture-of-Experts Router: Routes code to the C/C++ Specialist
    or the Multilingual Generalist based on language syntax.
    """

    def __init__(
        self,
        specialist_path: str | Path = "models/codebert-baseline",
        generalist_path: str | Path = "models/codebert-multilingual",
    ):
        self.specialist_path = Path(specialist_path).expanduser().resolve()
        self.generalist_path = Path(generalist_path).expanduser().resolve()

    def is_specialist_available(self) -> bool:
        required = ("config.json", "model.safetensors", "tokenizer.json")
        return all((self.specialist_path / f).is_file() for f in required)

    def is_generalist_available(self) -> bool:
        required = ("config.json", "model.safetensors", "tokenizer.json")
        return all((self.generalist_path / f).is_file() for f in required)

    def route(
        self,
        code: str,
        forced_language: str | None = None,
        forced_model: Literal["auto", "specialist", "generalist"] | None = "auto",
    ) -> dict[str, Any]:
        """
        Determines which expert model should analyze the input code.
        """
        # 1. Manual model override
        if forced_model == "specialist":
            return {
                "model_key": "specialist",
                "model_name": "C/C++ Specialist (PrimeVul)",
                "detected_language": forced_language or "c_cpp",
                "reason": "Explicitly routed to C/C++ Specialist by user override.",
                "model_path": self.specialist_path,
            }
        elif forced_model == "generalist":
            return {
                "model_key": "generalist",
                "model_name": "Multilingual Generalist (CrossVul)",
                "detected_language": forced_language or "multilingual",
                "reason": "Explicitly routed to Multilingual Generalist by user override.",
                "model_path": self.generalist_path if self.is_generalist_available() else self.specialist_path,
            }

        # 2. Language detection
        detected_lang, confidence = (
            (forced_language.lower(), 10.0)
            if forced_language and forced_language != "auto"
            else detect_language(code)
        )

        # 3. Routing decision
        if detected_lang in SPECIALIST_LANGUAGES:
            model_key = "specialist"
            model_name = "C/C++ Specialist (PrimeVul)"
            reason = f"Detected C/C++ syntax ({detected_lang.upper()}) -> Routed to C/C++ Specialist Model to prevent catastrophic forgetting."
            model_path = self.specialist_path
        else:
            model_key = "generalist"
            model_name = "Multilingual Generalist (CrossVul)"
            reason = f"Detected {detected_lang.capitalize()} syntax -> Routed to Multilingual Generalist Model."
            model_path = self.generalist_path if self.is_generalist_available() else self.specialist_path

        return {
            "model_key": model_key,
            "model_name": model_name,
            "detected_language": detected_lang,
            "confidence_score": confidence,
            "reason": reason,
            "model_path": model_path,
        }
