from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.db import get_db
from app.models import Artifact, Case, Job
from app.schemas import (
    ArtifactResponse,
    CaseCreateRequest,
    CaseResponse,
    InferencePingResponse,
    JobAdvanceRequest,
    JobCreateRequest,
    JobResponse,
    MockInferenceRequest,
    MockInferenceResponse,
    QCEvaluateRequest,
    QCEvaluateResponse,
    QueuePullResponse,
    RagIngestRequest,
    RagSearchItem,
    RagSearchRequest,
    RagSearchResponse,
    WorkflowResultResponse,
    WorkflowSubmitRequest,
    WorkflowSubmitResponse,
)
from app.core.config import get_settings
from app.services.inference import run_mock_inference
from app.services.inference import InferenceProviderError, run_configured_inference
from app.services.medgemma import ping_medgemma_health
from app.services.orchestrator import (
    InvalidTransitionError,
    NotFoundError,
    advance_job_state,
    create_job_with_idempotency,
    pull_next_queued_job,
)
from app.services.qc import evaluate_findings
from app.services.rag import ingest_doc, search_docs
from app.services.storage import save_upload_file
from app.services.workflow import get_job_output, submit_workflow

public_router = APIRouter()
router = APIRouter(dependencies=[Depends(require_api_key)])


@public_router.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@public_router.get("/health/ready")
def health_ready(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ready"}


@router.post("/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_case(payload: CaseCreateRequest, db: Session = Depends(get_db)) -> Case:
    case = Case(patient_pseudo_id=payload.patient_pseudo_id)
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


@router.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id: str, db: Session = Depends(get_db)) -> Case:
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case_not_found")
    return case


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreateRequest, db: Session = Depends(get_db)) -> Job:
    case = db.get(Case, payload.case_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case_not_found")

    job, created = create_job_with_idempotency(
        db,
        case_id=payload.case_id,
        stage=payload.stage,
        idempotency_key=payload.idempotency_key,
    )
    if not created:
        return job
    return job


@router.post("/workflow/submit", response_model=WorkflowSubmitResponse, status_code=status.HTTP_201_CREATED)
def workflow_submit(payload: WorkflowSubmitRequest, db: Session = Depends(get_db)) -> WorkflowSubmitResponse:
    try:
        case, job = submit_workflow(
            db,
            patient_pseudo_id=payload.patient_pseudo_id,
            notes=payload.notes,
            images=payload.images,
            idempotency_key=payload.idempotency_key,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return WorkflowSubmitResponse(case=case, job=job)


@router.post("/queue/pull", response_model=QueuePullResponse)
def pull_queue_job(db: Session = Depends(get_db)) -> QueuePullResponse:
    job = pull_next_queued_job(db)
    return QueuePullResponse(job=job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> Job:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    return job


@router.get("/workflow/jobs/{job_id}/result", response_model=WorkflowResultResponse)
def get_workflow_result(job_id: str, db: Session = Depends(get_db)) -> WorkflowResultResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    output = get_job_output(db, job_id=job_id)
    return WorkflowResultResponse(job=job, output=output)


@router.post("/jobs/{job_id}/advance", response_model=JobResponse)
def advance_job(job_id: str, payload: JobAdvanceRequest, db: Session = Depends(get_db)) -> Job:
    try:
        return advance_job_state(
            db,
            job_id=job_id,
            target_state=payload.target_state,
            error_code=payload.error_code,
        )
    except NotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.post("/cases/{case_id}/artifacts", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
def upload_artifact(
    case_id: str,
    kind: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Artifact:
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="case_not_found")

    file_name, file_path = save_upload_file(case_id, file)
    artifact = Artifact(case_id=case_id, kind=kind, file_name=file_name, file_path=file_path)
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


@router.get("/artifacts/{artifact_id}")
def download_artifact(artifact_id: str, db: Session = Depends(get_db)) -> FileResponse:
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="artifact_not_found")

    path = Path(artifact.file_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="artifact_file_missing")
    return FileResponse(path=str(path), filename=artifact.file_name)


@router.post("/rag/ingest")
def rag_ingest(payload: RagIngestRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    doc = ingest_doc(
        db,
        source=payload.source,
        source_version=payload.source_version,
        title=payload.title,
        content=payload.content,
    )
    return {"doc_id": doc.doc_id}


@router.post("/rag/search", response_model=RagSearchResponse)
def rag_search(payload: RagSearchRequest, db: Session = Depends(get_db)) -> RagSearchResponse:
    rows = search_docs(db, query=payload.query, top_k=payload.top_k)
    items = [
        RagSearchItem(
            doc_id=doc.doc_id,
            source=doc.source,
            source_version=doc.source_version,
            title=doc.title,
            snippet=doc.content[:200],
            score=score,
        )
        for doc, score in rows
    ]
    return RagSearchResponse(items=items)


@router.post("/qc/evaluate", response_model=QCEvaluateResponse)
def qc_evaluate(payload: QCEvaluateRequest) -> QCEvaluateResponse:
    result, issues = evaluate_findings(payload.findings)
    return QCEvaluateResponse(status=result, issues=issues)


@router.post("/inference/mock", response_model=MockInferenceResponse)
def mock_inference(payload: MockInferenceRequest) -> MockInferenceResponse:
    return MockInferenceResponse(**run_mock_inference(payload.case_id, payload.notes, payload.images))


@router.post("/inference/run", response_model=MockInferenceResponse)
def run_inference(payload: MockInferenceRequest) -> MockInferenceResponse:
    try:
        return MockInferenceResponse(
            **run_configured_inference(payload.case_id, payload.notes, payload.images)
        )
    except InferenceProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.get("/inference/ping", response_model=InferencePingResponse)
def ping_inference_provider() -> InferencePingResponse:
    settings = get_settings()
    provider = settings.inference_provider
    if provider == "mock":
        return InferencePingResponse(provider="mock", reachable=True, detail={"status": "mock_ready"})

    result = ping_medgemma_health()
    if not result.get("reachable"):
        return InferencePingResponse(
            provider="medgemma",
            reachable=False,
            detail={"error": result.get("error", "unknown_error")},
        )
    return InferencePingResponse(provider="medgemma", reachable=True, detail=result.get("raw"))
