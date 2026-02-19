from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Job, JobStage, JobState


class OrchestratorError(Exception):
    pass


class InvalidTransitionError(OrchestratorError):
    pass


class NotFoundError(OrchestratorError):
    pass


ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.queued: {JobState.running, JobState.failed},
    JobState.running: {JobState.succeeded, JobState.failed},
    JobState.failed: {JobState.queued, JobState.dead_letter},
    JobState.succeeded: set(),
    JobState.dead_letter: set(),
}


def create_job_with_idempotency(
    db: Session,
    *,
    case_id: str,
    stage: JobStage,
    idempotency_key: str,
) -> tuple[Job, bool]:
    existing = db.scalar(
        select(Job).where(
            Job.case_id == case_id,
            Job.idempotency_key == idempotency_key,
        )
    )
    if existing:
        return existing, False

    job = Job(case_id=case_id, stage=stage, idempotency_key=idempotency_key)
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        existing_same_case = db.scalar(
            select(Job).where(
                Job.case_id == case_id,
                Job.idempotency_key == idempotency_key,
            )
        )
        if existing_same_case:
            return existing_same_case, False

        # If this fires, your local DB likely still has the legacy global unique
        # constraint on idempotency_key. Recreate or migrate the DB schema.
        raise OrchestratorError("idempotency_key_conflict_across_case") from exc
    db.refresh(job)
    return job, True


def advance_job_state(db: Session, *, job_id: str, target_state: JobState, error_code: str | None) -> Job:
    job = db.get(Job, job_id)
    if not job:
        raise NotFoundError("job_not_found")

    if target_state not in ALLOWED_TRANSITIONS[job.state]:
        raise InvalidTransitionError(f"invalid_transition: {job.state} -> {target_state}")

    job.state = target_state
    if target_state == JobState.failed:
        job.retry_count += 1
        job.error_code = error_code or "unknown_failure"
    if target_state in {JobState.succeeded, JobState.dead_letter}:
        job.error_code = error_code

    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def pull_next_queued_job(db: Session) -> Job | None:
    job = db.scalar(
        select(Job).where(Job.state == JobState.queued).order_by(Job.created_at.asc()).limit(1)
    )
    if not job:
        return None

    job.state = JobState.running
    db.add(job)
    db.commit()
    db.refresh(job)
    return job
