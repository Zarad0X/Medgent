import json
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request

from app.core.config import get_settings


@dataclass
class InferenceResult:
    case_id: str
    summary: str
    findings: list[str]
    confidence: float
    used_fallback: bool = False
    run_mode: str = "unknown"
    model_source: str | None = None
    generated_token_count: int = 0
    raw_generated_text: str | None = None
    raw_generated_text_with_special: str | None = None


class InferenceProvider(Protocol):
    def run(
        self,
        *,
        case_id: str,
        notes: str | None,
        images: list[str] | None = None,
        rag_context: str | None = None,
    ) -> InferenceResult:
        ...


class InferenceProviderError(Exception):
    pass


class MockInferenceProvider:
    def run(
        self,
        *,
        case_id: str,
        notes: str | None,
        images: list[str] | None = None,
        rag_context: str | None = None,
    ) -> InferenceResult:
        text = (notes or "").strip()
        image_count = len(images or [])
        if not text:
            text = f"收到 {image_count} 张影像。"
        short = text if len(text) <= 80 else f"{text[:77]}..."
        findings = [
            "病灶较前变化稳定，建议继续随访。",
            "未见明显新发高危征象。",
        ]
        if image_count:
            findings.append(f"已接收影像数量: {image_count}")
        return InferenceResult(
            case_id=case_id,
            summary=f"Mock推理结论：{short}",
            findings=findings,
            confidence=0.72,
            used_fallback=False,
            run_mode="mock",
            model_source="mock_provider",
            generated_token_count=0,
            raw_generated_text=None,
            raw_generated_text_with_special=None,
        )


class MedGemmaInferenceProvider:
    def __init__(self, *, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        *,
        case_id: str,
        notes: str | None,
        images: list[str] | None = None,
        rag_context: str | None = None,
    ) -> InferenceResult:
        payload = json.dumps(
            {
                "case_id": case_id,
                "notes": notes,
                "images": images or [],
                "rag_context": rag_context,
            }
        ).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/infer",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            message = f"medgemma_http_{exc.code}"
            if detail:
                message = f"{message}: {detail}"
            raise InferenceProviderError(message) from exc
        except error.URLError as exc:
            raise InferenceProviderError(f"medgemma_unreachable: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise InferenceProviderError("medgemma_invalid_json") from exc

        findings = data.get("findings", [])
        if not isinstance(findings, list):
            raise InferenceProviderError("medgemma_invalid_findings")

        return InferenceResult(
            case_id=str(data.get("case_id") or case_id),
            summary=str(data.get("summary") or ""),
            findings=[str(item) for item in findings],
            confidence=float(data.get("confidence") or 0.0),
            used_fallback=bool(data.get("used_fallback", False)),
            run_mode=str(data.get("run_mode") or "medgemma_unknown"),
            model_source=str(data.get("model_source") or self.base_url),
            generated_token_count=int(data.get("generated_token_count") or 0),
            raw_generated_text=(
                str(data.get("raw_generated_text")) if data.get("raw_generated_text") is not None else None
            ),
            raw_generated_text_with_special=(
                str(data.get("raw_generated_text_with_special"))
                if data.get("raw_generated_text_with_special") is not None
                else None
            ),
        )


def get_configured_provider() -> InferenceProvider:
    settings = get_settings()
    if settings.inference_provider == "medgemma":
        return MedGemmaInferenceProvider(
            base_url=settings.medgemma_base_url,
            timeout_seconds=settings.medgemma_timeout_seconds,
        )
    return MockInferenceProvider()


def run_mock_inference(
    case_id: str,
    notes: str | None,
    images: list[str] | None = None,
    rag_context: str | None = None,
) -> dict:
    provider = MockInferenceProvider()
    result = provider.run(case_id=case_id, notes=notes, images=images, rag_context=rag_context)
    return {
        "case_id": result.case_id,
        "summary": result.summary,
        "findings": result.findings,
        "confidence": result.confidence,
        "used_fallback": result.used_fallback,
        "run_mode": result.run_mode,
        "model_source": result.model_source,
        "generated_token_count": result.generated_token_count,
        "raw_generated_text": result.raw_generated_text,
        "raw_generated_text_with_special": result.raw_generated_text_with_special,
    }


def run_configured_inference(
    case_id: str,
    notes: str | None,
    images: list[str] | None = None,
    rag_context: str | None = None,
) -> dict:
    provider = get_configured_provider()
    result = provider.run(case_id=case_id, notes=notes, images=images, rag_context=rag_context)
    return {
        "case_id": result.case_id,
        "summary": result.summary,
        "findings": result.findings,
        "confidence": result.confidence,
        "used_fallback": result.used_fallback,
        "run_mode": result.run_mode,
        "model_source": result.model_source,
        "generated_token_count": result.generated_token_count,
        "raw_generated_text": result.raw_generated_text,
        "raw_generated_text_with_special": result.raw_generated_text_with_special,
    }
