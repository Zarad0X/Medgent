import json
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Artifact, Case, Job, JobStage
from app.services.orchestrator import create_job_with_idempotency
from app.services.storage import save_json_file, save_local_file, save_text_file


def _validate_source_images(images: list[str]) -> None:
    for image_path in images:
        source = Path(image_path).expanduser()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"source_file_not_found:{image_path}")


def submit_workflow(
    db: Session,
    *,
    patient_pseudo_id: str,
    notes: str | None,
    images: list[str] | None,
    idempotency_key: str | None,
) -> tuple[Case, Job]:
    normalized_notes = (notes or "").strip()
    normalized_images = images or []
    _validate_source_images(normalized_images)

    created_files: list[str] = []
    case = Case(patient_pseudo_id=patient_pseudo_id)
    job: Job | None = None

    try:
        db.add(case)
        db.flush()

        if normalized_notes:
            notes_path = save_text_file(
                case.case_id,
                prefix="input",
                file_name="notes.txt",
                content=normalized_notes,
            )
            created_files.append(notes_path)
            artifact = Artifact(
                case_id=case.case_id,
                kind="input_notes",
                file_name="notes.txt",
                file_path=notes_path,
            )
            db.add(artifact)

        for image_path in normalized_images:
            original_name, stored_path = save_local_file(
                case.case_id,
                prefix="input",
                source_path=image_path,
            )
            created_files.append(stored_path)
            artifact = Artifact(
                case_id=case.case_id,
                kind="input_image",
                file_name=original_name,
                file_path=stored_path,
            )
            db.add(artifact)

        idem = idempotency_key or f"workflow-{uuid.uuid4()}"
        job, _ = create_job_with_idempotency(
            db,
            case_id=case.case_id,
            stage=JobStage.inference,
            idempotency_key=idem,
            autocommit=False,
        )

        db.commit()
    except Exception:
        db.rollback()
        for file_path in created_files:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        raise

    db.refresh(case)
    if job is None:
        raise RuntimeError("job_not_created")
    db.refresh(job)
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


def get_case_input_images(db: Session, *, case_id: str) -> list[str]:
    artifacts = db.scalars(
        select(Artifact)
        .where(Artifact.case_id == case_id, Artifact.kind == "input_image")
        .order_by(Artifact.created_at.asc())
    ).all()
    image_paths: list[str] = []
    for artifact in artifacts:
        path = Path(artifact.file_path)
        if path.exists():
            image_paths.append(str(path))
    return image_paths


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

    artifacts = db.scalars(
        select(Artifact)
        .where(Artifact.case_id == job.case_id, Artifact.kind == "agent_output")
        .order_by(Artifact.created_at.desc(), Artifact.artifact_id.desc())
    ).all()
    if not artifacts:
        return None

    # Backward compatibility for legacy outputs that didn't include job_id.
    legacy_fallback: dict | None = None
    for artifact in artifacts:
        path = Path(artifact.file_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        payload_job_id = payload.get("job_id")
        if payload_job_id is None and legacy_fallback is None:
            legacy_fallback = payload
            continue
        if str(payload_job_id) == job_id:
            return payload

    return legacy_fallback
