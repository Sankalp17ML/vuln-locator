import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "codebert-baseline"
MAX_LENGTH = 512


class CodeAnalyzer:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.threshold = (
            float(os.getenv("VULNLOCATOR_THRESHOLD", "0.5"))
            if threshold is None
            else threshold
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("The vulnerability threshold must be between 0 and 1.")

        requested_device = device or os.getenv("VULNLOCATOR_DEVICE", "auto")
        if requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but no CUDA device is available.")

        required_files = ("config.json", "model.safetensors", "tokenizer.json")
        missing_files = [name for name in required_files if not (self.model_path / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Model files are missing from {self.model_path}: {', '.join(missing_files)}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

    def analyze(
        self,
        code: str,
        top_k: int = 10,
        include_localization: bool = True,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        if not code.strip():
            raise ValueError("Code input must not be empty.")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        active_threshold = self.threshold if threshold is None else threshold
        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        encoded = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=include_localization,
        )
        offsets = encoded.pop("offset_mapping", None)
        model_inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                output_attentions=include_localization,
                return_dict=True,
            )
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            vulnerable_probability = float(probabilities[1].item())

            if include_localization:
                if not outputs.attentions:
                    raise RuntimeError("The model did not return attention weights for localization.")
                importance = self._attention_rollout(outputs.attentions)
                findings = self._top_findings(
                    code,
                    encoded["input_ids"][0],
                    offsets[0],
                    importance,
                    top_k,
                )
            else:
                findings = []

        vulnerable = vulnerable_probability >= active_threshold
        return {
            "vulnerable": vulnerable,
            "label": "Vulnerable" if vulnerable else "Safe",
            "confidence": (
                vulnerable_probability if vulnerable else 1.0 - vulnerable_probability
            ),
            "vulnerable_probability": vulnerable_probability,
            "threshold": active_threshold,
            "device": str(self.device),
            "findings": findings,
        }

    def predict_probabilities(self, code_samples: list[str], batch_size: int = 8) -> list[float]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        probabilities: list[float] = []
        for start in range(0, len(code_samples), batch_size):
            batch = code_samples[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            model_inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**model_inputs, return_dict=True).logits
                probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return probabilities

    @staticmethod
    def _attention_rollout(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
        sequence_length = attentions[0].shape[-1]
        identity = torch.eye(sequence_length, device=attentions[0].device)
        rollout = identity

        for layer_attention in attentions:
            averaged_attention = layer_attention[0].mean(dim=0)
            residual_attention = 0.5 * averaged_attention + 0.5 * identity
            residual_attention = residual_attention / residual_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(residual_attention.dtype).eps)
            rollout = residual_attention @ rollout

        return rollout[0]

    def _top_findings(
        self,
        code: str,
        input_ids: torch.Tensor,
        offsets: torch.Tensor,
        importance: torch.Tensor,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidates: list[tuple[float, int, int, str]] = []
        token_ids = input_ids.detach().cpu().tolist()
        raw_scores = importance.detach().cpu().tolist()

        for token_id, (start_offset, end_offset), score in zip(token_ids, offsets, raw_scores):
            start = int(start_offset)
            end = int(end_offset)
            if start >= end or not code[start:end].strip():
                continue
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            candidates.append((float(score), start, end, token))

        if not candidates:
            return []

        max_score = max(score for score, _, _, _ in candidates)
        if max_score <= 0:
            return []

        lines = code.splitlines()
        candidates.sort(key=lambda item: item[0], reverse=True)
        findings = []
        for score, start, end, token in candidates[:top_k]:
            line_number = code.count("\n", 0, start) + 1
            line_text = lines[line_number - 1] if line_number <= len(lines) else ""
            findings.append(
                {
                    "token": token,
                    "text": code[start:end],
                    "score": score / max_score,
                    "start": start,
                    "end": end,
                    "line": line_number,
                    "line_text": line_text,
                }
            )
        return findings
