import argparse
import time

from app.core.config import get_settings
from app.db import Base, SessionLocal, engine
from app.services.inference import InferenceProviderError, run_configured_inference
from app.services.orchestrator import advance_job_state, pull_next_queued_job
from app.services.qc import evaluate_findings
from app.services.workflow import get_case_input_notes, save_agent_output
from app.models import JobState


def process_next_job() -> dict | None:
    settings = get_settings()
    with SessionLocal() as db:
        job = pull_next_queued_job(db)
        if not job:
            return None

        notes = get_case_input_notes(db, case_id=job.case_id)
        if not notes:
            advance_job_state(db, job_id=job.job_id, target_state=JobState.failed, error_code="missing_input_notes")
            return {"job_id": job.job_id, "state": "failed", "reason": "missing_input_notes"}

        try:
            inference = run_configured_inference(job.case_id, notes)
        except InferenceProviderError as exc:
            advance_job_state(db, job_id=job.job_id, target_state=JobState.failed, error_code=str(exc))
            return {"job_id": job.job_id, "state": "failed", "reason": str(exc)}

        findings_text = " ".join(inference.get("findings", []))
        qc_status, issues = evaluate_findings(findings_text)
        output = {
            "case_id": job.case_id,
            "job_id": job.job_id,
            "inference": inference,
            "qc_status": qc_status,
            "qc_issues": issues,
        }

        if qc_status == "blocked" and not settings.qc_block_fails_job:
            output["qc_status"] = "review_required"
            output["qc_issues"] = [*issues, "qc_block_downgraded_for_debug"]
            save_agent_output(db, case_id=job.case_id, payload=output)
            advance_job_state(db, job_id=job.job_id, target_state=JobState.succeeded, error_code=None)
            return {"job_id": job.job_id, "state": "succeeded", "qc_status": "review_required"}

        save_agent_output(db, case_id=job.case_id, payload=output)
        if qc_status == "blocked":
            advance_job_state(db, job_id=job.job_id, target_state=JobState.failed, error_code="qc_blocked")
            return {"job_id": job.job_id, "state": "failed", "reason": "qc_blocked"}

        advance_job_state(db, job_id=job.job_id, target_state=JobState.succeeded, error_code=None)
        return {"job_id": job.job_id, "state": "succeeded", "qc_status": qc_status}


def run_worker(*, once: bool) -> None:
    settings = get_settings()
    Base.metadata.create_all(bind=engine)

    if once:
        process_next_job()
        return

    while True:
        process_next_job()
        time.sleep(settings.worker_poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medgent workflow worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job then exit")
    args = parser.parse_args()
    run_worker(once=args.once)


if __name__ == "__main__":
    main()
