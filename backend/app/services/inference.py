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


class InferenceProvider(Protocol):
    def run(self, *, case_id: str, notes: str) -> InferenceResult:
        ...


class InferenceProviderError(Exception):
    pass


class MockInferenceProvider:
    def run(self, *, case_id: str, notes: str) -> InferenceResult:
        text = notes.strip()
        short = text if len(text) <= 80 else f"{text[:77]}..."
        findings = [
            "病灶较前变化稳定，建议继续随访。",
            "未见明显新发高危征象。",
        ]
        return InferenceResult(
            case_id=case_id,
            summary=f"Mock推理结论：{short}",
            findings=findings,
            confidence=0.72,
        )


class MedGemmaInferenceProvider:
    def __init__(self, *, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def run(self, *, case_id: str, notes: str) -> InferenceResult:
        payload = json.dumps({"case_id": case_id, "notes": notes}).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/infer",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
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
        )


def get_configured_provider() -> InferenceProvider:
    settings = get_settings()
    if settings.inference_provider == "medgemma":
        return MedGemmaInferenceProvider(
            base_url=settings.medgemma_base_url,
            timeout_seconds=settings.medgemma_timeout_seconds,
        )
    return MockInferenceProvider()


def run_mock_inference(case_id: str, notes: str) -> dict:
    provider = MockInferenceProvider()
    result = provider.run(case_id=case_id, notes=notes)
    return {
        "case_id": result.case_id,
        "summary": result.summary,
        "findings": result.findings,
        "confidence": result.confidence,
    }


def run_configured_inference(case_id: str, notes: str) -> dict:
    provider = get_configured_provider()
    result = provider.run(case_id=case_id, notes=notes)
    return {
        "case_id": result.case_id,
        "summary": result.summary,
        "findings": result.findings,
        "confidence": result.confidence,
    }
