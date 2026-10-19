import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import CodeAnalyzer, DEFAULT_MODEL_PATH
from src.router import LanguageRouter


BASE_DIR = Path(__file__).resolve().parents[1]
SPECIALIST_PATH = Path(os.getenv("VULNLOCATOR_SPECIALIST_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()
GENERALIST_PATH = Path(os.getenv("VULNLOCATOR_GENERALIST_PATH", str(BASE_DIR / "models" / "codebert-multilingual"))).expanduser().resolve()

router = LanguageRouter(specialist_path=SPECIALIST_PATH, generalist_path=GENERALIST_PATH)

app = FastAPI(
    title="VulnLocator V2.0 API (Mixture of Experts)",
    description="Multilingual vulnerability triage routing between C/C++ Specialist and CrossVul Generalist models.",
    version="2.0.0",
)


class PredictionRequest(BaseModel):
    code: str = Field(min_length=1, max_length=100_000)
    top_k: int = Field(default=10, ge=1, le=50)
    include_localization: bool = True
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    strip_comments: bool = True
    filter_punctuation: bool = True
    strip_headers: bool = False
    language: str | None = Field(default="auto", description="Programming language ('auto', 'c', 'cpp', 'java', 'python', 'php', 'javascript')")
    model_choice: Literal["auto", "specialist", "generalist"] = Field(
        default="auto",
        description="Routing mode ('auto' uses AST/syntax router; 'specialist' forces C/C++; 'generalist' forces Multilingual)"
    )


class Finding(BaseModel):
    token: str
    text: str
    score: float
    start: int
    end: int
    line: int
    line_text: str


class PredictionResponse(BaseModel):
    vulnerable: bool
    label: str
    confidence: float
    vulnerable_probability: float
    threshold: float
    device: str
    findings: list[Finding]
    preprocessed: bool = False
    comments_stripped: int = 0
    comment_chars_removed: int = 0
    model_used: str
    detected_language: str
    routing_reason: str


@lru_cache(maxsize=1)
def get_specialist() -> CodeAnalyzer:
    if not router.is_specialist_available():
        raise HTTPException(
            status_code=503,
            detail=f"C/C++ Specialist model weights were not found at {SPECIALIST_PATH}.",
        )
    return CodeAnalyzer(model_path=SPECIALIST_PATH)


@lru_cache(maxsize=1)
def get_generalist() -> CodeAnalyzer:
    if router.is_generalist_available():
        return CodeAnalyzer(model_path=GENERALIST_PATH)
    # Graceful fallback to specialist if generalist is not yet initialized
    return get_specialist()


@app.get("/health")
def health() -> dict[str, Any]:
    specialist_ok = router.is_specialist_available()
    generalist_ok = router.is_generalist_available()
    return {
        "status": "ok" if (specialist_ok or generalist_ok) else "model_unavailable",
        "specialist_available": specialist_ok,
        "generalist_available": generalist_ok,
        "version": "2.0.0",
        "architecture": "Mixture-of-Experts (Router)",
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    return {
        "specialist": {
            "name": "C/C++ Specialist (PrimeVul)",
            "available": router.is_specialist_available(),
            "path": str(SPECIALIST_PATH),
            "target_domains": ["C", "C++", "memory safety", "buffer overflows", "pointers"],
        },
        "generalist": {
            "name": "Multilingual Generalist (CrossVul)",
            "available": router.is_generalist_available(),
            "path": str(GENERALIST_PATH),
            "target_domains": ["Java", "Python", "PHP", "JavaScript", "web & API security"],
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    if not request.code.strip():
        raise HTTPException(status_code=422, detail="Code input must not be empty.")

    # 1. Determine Model via Router
    route_info = router.route(
        request.code,
        forced_language=request.language,
        forced_model=request.model_choice,
    )
    model_key = route_info["model_key"]

    analyzer = get_specialist() if model_key == "specialist" else get_generalist()

    # 2. Run Inference with Syntax Preprocessor & Attention Rollout
    result = analyzer.analyze(
        request.code,
        top_k=request.top_k,
        include_localization=request.include_localization,
        threshold=request.threshold,
        strip_comments=request.strip_comments,
        filter_punctuation=request.filter_punctuation,
        strip_headers=request.strip_headers,
    )

    # 3. Attach Routing Metadata
    result["model_used"] = route_info["model_name"]
    result["detected_language"] = route_info["detected_language"]
    result["routing_reason"] = route_info["reason"]
    return result


@app.post("/predict/specialist", response_model=PredictionResponse)
def predict_specialist(request: PredictionRequest) -> dict:
    request.model_choice = "specialist"
    return predict(request)


@app.post("/predict/generalist", response_model=PredictionResponse)
def predict_generalist(request: PredictionRequest) -> dict:
    request.model_choice = "generalist"
    return predict(request)