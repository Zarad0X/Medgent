from datetime import datetime

from pydantic import BaseModel, Field

from app.models import CaseStatus, JobStage, JobState


class ORMBaseModel(BaseModel):
    model_config = {"from_attributes": True}


class CaseCreateRequest(BaseModel):
    patient_pseudo_id: str = Field(min_length=1, max_length=128)


class CaseResponse(ORMBaseModel):
    case_id: str
    patient_pseudo_id: str
    status: CaseStatus
    created_at: datetime
    updated_at: datetime


class JobCreateRequest(BaseModel):
    case_id: str
    stage: JobStage
    idempotency_key: str = Field(min_length=8, max_length=128)


class JobAdvanceRequest(BaseModel):
    target_state: JobState
    error_code: str | None = None


class JobResponse(ORMBaseModel):
    job_id: str
    case_id: str
    stage: JobStage
    state: JobState
    retry_count: int
    idempotency_key: str
    error_code: str | None
    created_at: datetime
    updated_at: datetime


class QueuePullResponse(BaseModel):
    job: JobResponse | None


class ArtifactResponse(ORMBaseModel):
    artifact_id: str
    case_id: str
    kind: str
    file_name: str
    file_path: str
    created_at: datetime


class RagIngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=128)
    source_version: str = Field(default="v1", min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=8000)


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    top_k: int = Field(default=3, ge=1, le=10)


class RagSearchItem(BaseModel):
    doc_id: str
    source: str
    source_version: str
    title: str
    snippet: str
    score: float


class RagSearchResponse(BaseModel):
    items: list[RagSearchItem]


class QCEvaluateRequest(BaseModel):
    findings: str = Field(min_length=1, max_length=4000)


class QCEvaluateResponse(BaseModel):
    status: str
    issues: list[str]


class MockInferenceRequest(BaseModel):
    case_id: str
    notes: str = Field(min_length=1, max_length=4000)


class MockInferenceResponse(BaseModel):
    case_id: str
    summary: str
    findings: list[str]
    confidence: float


class WorkflowSubmitRequest(BaseModel):
    patient_pseudo_id: str = Field(min_length=1, max_length=128)
    notes: str = Field(min_length=1, max_length=4000)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)


class WorkflowSubmitResponse(BaseModel):
    case: CaseResponse
    job: JobResponse


class WorkflowResultResponse(BaseModel):
    job: JobResponse
    output: dict | None


class InferencePingResponse(BaseModel):
    provider: str
    reachable: bool
    detail: dict | str | None = None
