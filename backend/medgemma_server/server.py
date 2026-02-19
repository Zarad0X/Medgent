from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 9000
    model_id: str = "google/medgemma-4b-it"
    local_model_dir: str = "/home/answer12_as/workspace/Medgent/backend/models"
    local_files_only: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    use_chat_template: bool = True
    min_new_tokens: int = 32
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    use_explicit_eos_pad: bool = False


def resolve_torch_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    return torch.float32


@dataclass
class ModelRuntime:
    processor: AutoProcessor
    model: AutoModelForImageTextToText
    device: torch.device
    input_device: torch.device
    model_source: str
    local_only: bool
    run_mode: str


settings = Settings()
runtime: ModelRuntime | None = None
logger = logging.getLogger("medgemma_server")


class InferRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=8000)
    images: list[str] = Field(default_factory=list, max_length=8)
    rag_context: str | None = Field(default=None, max_length=8000)

    @model_validator(mode="after")
    def validate_modalities(self) -> "InferRequest":
        has_notes = bool((self.notes or "").strip())
        has_images = len(self.images) > 0
        if not has_notes and not has_images:
            raise ValueError("either notes or images must be provided")
        return self


class InferResponse(BaseModel):
    case_id: str
    summary: str
    findings: list[str]
    confidence: float
    used_fallback: bool = False
    run_mode: str
    model_source: str
    generated_token_count: int
    raw_generated_text: str | None = None
    raw_generated_text_with_special: str | None = None


class StructuredModelOutput(BaseModel):
    summary: str = Field(min_length=1, max_length=2000)
    findings: list[str] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("findings")
    @classmethod
    def validate_findings(cls, values: list[str]) -> list[str]:
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            raise ValueError("findings_empty")
        return cleaned


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
    processor = AutoProcessor.from_pretrained(model_source, local_files_only=local_only)
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

    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        local_files_only=local_only,
        **model_kwargs,
    )

    if settings.device in {"cpu", "cuda"}:
        device = torch.device(settings.device)
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    input_device = device
    if getattr(model, "hf_device_map", None):
        for _, mapped in model.hf_device_map.items():
            if isinstance(mapped, int):
                input_device = torch.device(f"cuda:{mapped}")
                break
            if isinstance(mapped, str) and mapped.startswith("cuda"):
                input_device = torch.device(mapped)
                break
        if str(input_device) == "meta":
            input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif str(input_device) == "meta":
        input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_quantization and settings.load_in_4bit:
        run_mode = "4bit"
    elif use_quantization and settings.load_in_8bit:
        run_mode = "8bit"
    else:
        run_mode = f"native_{settings.dtype}"

    return ModelRuntime(
        processor=processor,
        model=model,
        device=device,
        input_device=input_device,
        model_source=model_source,
        local_only=local_only,
        run_mode=run_mode,
    )


def load_input_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"image_not_found:{image_path}")
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def _extract_first_json_object(text: str) -> str | None:
    s = text.strip()
    if not s:
        return None

    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            fenced = "\n".join(lines[1:-1]).strip()
            if fenced:
                s = fenced

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : idx + 1]
    return None


def parse_structured_output(text: str) -> StructuredModelOutput:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    extracted = _extract_first_json_object(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            return StructuredModelOutput.model_validate(data)
        except Exception:
            continue
    raise ValueError("structured_json_parse_failed")


def infer_text(notes: str | None, image_paths: list[str], rag_context: str | None) -> tuple[str, str, int]:
    if runtime is None:
        raise RuntimeError("runtime_not_initialized")

    normalized_notes = (notes or "").strip()
    user_prompt = (
        "你是放射科辅助分析助手。请基于输入内容输出严格 JSON，且只能输出 JSON 对象本身，不要 Markdown。"
        '\nJSON Schema: {"summary": string, "findings": string[], "confidence": number(0-1)}'
        '\n示例: {"summary":"右上肺结节较前稳定。","findings":["关键发现:右上肺结节","病灶变化:较前无明显变化","建议:继续随访"],"confidence":0.78}'
    )
    if normalized_notes:
        user_prompt = f"{user_prompt}\n随访记录：{normalized_notes}"
    else:
        user_prompt = f"{user_prompt}\n请基于提供的影像进行判断。"
    if rag_context and rag_context.strip():
        user_prompt = f"{user_prompt}\n参考以下临床指南与知识库（优先采纳高相关内容）：\n{rag_context.strip()}"

    content: list[dict[str, str]] = [{"type": "text", "text": user_prompt}]
    for _ in image_paths:
        content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]

    images = load_input_images(image_paths) if image_paths else None
    if settings.use_chat_template and hasattr(runtime.processor, "apply_chat_template"):
        prompt = runtime.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = runtime.processor(text=prompt, images=images, return_tensors="pt").to(runtime.input_device)
        input_tokens = int(model_inputs["input_ids"].shape[-1])
    else:
        model_inputs = runtime.processor(text=user_prompt, images=images, return_tensors="pt").to(runtime.input_device)
        input_tokens = int(model_inputs["input_ids"].shape[-1])

    generate_kwargs = {
        "min_new_tokens": settings.min_new_tokens,
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.temperature > 0.0,
        "temperature": max(settings.temperature, 1e-6),
        "top_p": settings.top_p,
    }
    if settings.use_explicit_eos_pad:
        eos_token_id = runtime.processor.tokenizer.eos_token_id
        generate_kwargs["pad_token_id"] = eos_token_id
        generate_kwargs["eos_token_id"] = eos_token_id

    with torch.inference_mode():
        output_ids = runtime.model.generate(**model_inputs, **generate_kwargs)
    if getattr(runtime.model.config, "is_encoder_decoder", False):
        generated_ids = output_ids[0]
    else:
        generated_ids = output_ids[0][input_tokens:]

    raw_with_special = runtime.processor.decode(generated_ids, skip_special_tokens=False).strip()
    text = runtime.processor.decode(generated_ids, skip_special_tokens=True).strip()

    pad_id = runtime.processor.tokenizer.pad_token_id
    eos_id = runtime.processor.tokenizer.eos_token_id
    effective_count = 0
    for tid in generated_ids.tolist():
        if pad_id is not None and tid == pad_id:
            continue
        if eos_id is not None and tid == eos_id:
            continue
        effective_count += 1

    return text, raw_with_special, int(effective_count)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime
    runtime = load_runtime()
    logger.info(
        "model loaded source=%s run_mode=%s device=%s input_device=%s",
        runtime.model_source,
        runtime.run_mode,
        runtime.device,
        runtime.input_device,
    )
    yield


app = FastAPI(title="MedGemma Inference Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    ready = "ready" if runtime is not None else "loading"
    base = {"status": ready, "model_id": settings.model_id}
    if runtime is None:
        return base
    return {
        **base,
        "run_mode": runtime.run_mode,
        "model_source": runtime.model_source,
        "local_only": runtime.local_only,
        "device": str(runtime.device),
        "input_device": str(runtime.input_device),
    }


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    try:
        text, raw_with_special, generated_token_count = infer_text(req.notes, req.images, req.rag_context)
        parsed = parse_structured_output(text)
    except ValueError as exc:
        detail = str(exc)
        if detail in {"structured_json_parse_failed"}:
            raise HTTPException(status_code=502, detail=detail)
        raise HTTPException(status_code=400, detail=detail)
    except Exception as exc:
        logger.exception("infer failed")
        raise HTTPException(status_code=500, detail=f"inference_failed: {exc}")

    used_fallback = False
    raw_generated_text = text if text else None
    raw_generated_text_with_special = raw_with_special if raw_with_special else None

    return InferResponse(
        case_id=req.case_id,
        summary=parsed.summary,
        findings=parsed.findings[:5],
        confidence=parsed.confidence,
        used_fallback=used_fallback,
        run_mode=runtime.run_mode if runtime else "unknown",
        model_source=runtime.model_source if runtime else settings.model_id,
        generated_token_count=generated_token_count,
        raw_generated_text=raw_generated_text,
        raw_generated_text_with_special=raw_generated_text_with_special,
    )
