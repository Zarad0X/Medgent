import json
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Artifact, Case, Job, JobStage
from app.services.orchestrator import create_job_with_idempotency
from app.services.storage import save_json_file, save_text_file


def submit_workflow(db: Session, *, patient_pseudo_id: str, notes: str, idempotency_key: str | None) -> tuple[Case, Job]:
    case = Case(patient_pseudo_id=patient_pseudo_id)
    db.add(case)
    db.commit()
    db.refresh(case)

    notes_path = save_text_file(
        case.case_id,
        prefix="input",
        file_name="notes.txt",
        content=notes,
    )
    artifact = Artifact(
        case_id=case.case_id,
        kind="input_notes",
        file_name="notes.txt",
        file_path=notes_path,
    )
    db.add(artifact)
    db.commit()

    idem = idempotency_key or f"workflow-{uuid.uuid4()}"
    job, _ = create_job_with_idempotency(
        db,
        case_id=case.case_id,
        stage=JobStage.inference,
        idempotency_key=idem,
    )
    return case, job


def get_case_input_notes(db: Session, *, case_id: str) -> str:
    artifact = db.scalar(
        select(Artifact)
        .where(Artifact.case_id == case_id, Artifact.kind == "input_notes")
        .order_by(Artifact.created_at.desc())
        .limit(1)
    )
    if not artifact:
        return ""
    path = Path(artifact.file_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_agent_output(db: Session, *, case_id: str, payload: dict) -> Artifact:
    file_path = save_json_file(
        case_id,
        prefix="output",
        file_name="agent_output.json",
        payload=payload,
    )
    artifact = Artifact(
        case_id=case_id,
        kind="agent_output",
        file_name="agent_output.json",
        file_path=file_path,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def get_job_output(db: Session, *, job_id: str) -> dict | None:
    job = db.get(Job, job_id)
    if not job:
        return None
    artifact = db.scalar(
        select(Artifact)
        .where(Artifact.case_id == job.case_id, Artifact.kind == "agent_output")
        .order_by(Artifact.created_at.desc())
        .limit(1)
    )
    if not artifact:
        return None
    path = Path(artifact.file_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
