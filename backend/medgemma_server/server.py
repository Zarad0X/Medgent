from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 9000
    model_id: str = "google/medgemma-4b-it"
    local_model_dir: str = "/home/answer12_as/workspace/Medgent/backend/models"
    local_files_only: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    max_new_tokens: int = 128
    temperature: float = 0.0


def resolve_torch_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    return torch.float32


@dataclass
class ModelRuntime:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


settings = Settings()
runtime: ModelRuntime | None = None


class InferRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    notes: str = Field(min_length=1, max_length=8000)


class InferResponse(BaseModel):
    case_id: str
    summary: str
    findings: list[str]
    confidence: float


def _looks_like_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def _find_nested_local_model_dir(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if _looks_like_local_model_dir(child):
            return child
    return None


def resolve_model_source() -> tuple[str, bool]:
    local_path = Path(settings.local_model_dir).expanduser()
    if _looks_like_local_model_dir(local_path):
        return str(local_path), True
    nested = _find_nested_local_model_dir(local_path)
    if nested:
        return str(nested), True

    model_path = Path(settings.model_id).expanduser()
    if _looks_like_local_model_dir(model_path):
        return str(model_path), True

    return settings.model_id, settings.local_files_only


def load_runtime() -> ModelRuntime:
    model_source, local_only = resolve_model_source()
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, local_files_only=local_only)
    torch_dtype = resolve_torch_dtype(settings.dtype)

    model_kwargs: dict = {"dtype": torch_dtype}
    use_quantization = settings.load_in_4bit or settings.load_in_8bit

    if use_quantization and settings.device == "cpu":
        # bitsandbytes quantization is meant for GPU execution.
        use_quantization = False

    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=settings.load_in_4bit,
            load_in_8bit=settings.load_in_8bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=settings.bnb_4bit_use_double_quant,
        )
        model_kwargs["device_map"] = "auto"
    elif settings.device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=local_only,
        **model_kwargs,
    )

    if settings.device in {"cpu", "cuda"}:
        device = torch.device(settings.device)
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    return ModelRuntime(tokenizer=tokenizer, model=model, device=device)


def infer_text(notes: str) -> str:
    if runtime is None:
        raise RuntimeError("runtime_not_initialized")

    prompt = (
        "You are an expert radiology assistant.\n"
        "Summarize clinically relevant findings from the following notes in concise Chinese.\n\n"
        f"Notes:\n{notes}\n"
    )
    inputs = runtime.tokenizer(prompt, return_tensors="pt").to(runtime.device)

    with torch.inference_mode():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.temperature > 0.0,
            temperature=max(settings.temperature, 1e-6),
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime
    runtime = load_runtime()
    yield


app = FastAPI(title="MedGemma Inference Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    ready = "ready" if runtime is not None else "loading"
    return {"status": ready, "model_id": settings.model_id}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    try:
        text = infer_text(req.notes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference_failed: {exc}")

    summary = text[:300] if text else "模型未返回有效文本。"
    findings = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    if not findings:
        findings = [summary]

    return InferResponse(
        case_id=req.case_id,
        summary=summary,
        findings=findings[:5],
        confidence=0.7,
    )
