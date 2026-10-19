import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import CodeAnalyzer, DEFAULT_MODEL_PATH


MODEL_PATH = Path(os.getenv("VULNLOCATOR_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
app = FastAPI(
    title="VulnLocator API",
    description="CodeBERT-based vulnerability classification with attention-rollout localization.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    code: str = Field(min_length=1, max_length=100_000)
    top_k: int = Field(default=10, ge=1, le=50)
    include_localization: bool = True
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


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


@lru_cache(maxsize=1)
def get_analyzer() -> CodeAnalyzer:
    return CodeAnalyzer(model_path=MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str | bool]:
    model_available = all(
        (MODEL_PATH / filename).is_file()
        for filename in ("config.json", "model.safetensors", "tokenizer.json")
    )
    return {
        "status": "ok" if model_available else "model_unavailable",
        "model_available": model_available,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    if not request.code.strip():
        raise HTTPException(status_code=422, detail="Code input must not be empty.")
    if not all(
        (MODEL_PATH / filename).is_file()
        for filename in ("config.json", "model.safetensors", "tokenizer.json")
    ):
        raise HTTPException(
            status_code=503,
            detail=f"Trained model weights were not found at {MODEL_PATH}.",
        )
    return get_analyzer().analyze(
        request.code,
        top_k=request.top_k,
        include_localization=request.include_localization,
        threshold=request.threshold,
    )