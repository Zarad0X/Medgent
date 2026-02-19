import argparse
import time

from app.core.config import get_settings
from app.db import Base, SessionLocal, engine
from app.services.inference import InferenceProviderError, run_configured_inference
from app.services.orchestrator import advance_job_state, pull_next_queued_job
from app.services.qc import evaluate_findings, flatten_qc_issues
from app.services.rag import search_docs
from app.services.workflow import get_case_input_images, get_case_input_notes, save_agent_output
from app.models import JobState


def build_rag_context(notes: str, rows: list[tuple]) -> str | None:
    if not notes or not rows:
        return None

    chunks: list[str] = []
    for idx, (doc, score) in enumerate(rows, start=1):
        chunks.append(
            (
                f"[{idx}] title={doc.title}; source={doc.source}; version={doc.source_version}; "
                f"score={score:.2f}\n{doc.content.strip()}"
            )
        )
    return "\n\n".join(chunks)


def build_rag_debug(rows: list[tuple]) -> list[dict]:
    items: list[dict] = []
    for doc, score in rows:
        items.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "source_version": doc.source_version,
                "score": round(float(score), 4),
                "snippet": doc.content[:240],
            }
        )
    return items


def process_next_job() -> dict | None:
    settings = get_settings()
    with SessionLocal() as db:
        total_start = time.perf_counter()
        job = pull_next_queued_job(db)
        if not job:
            return None

        notes = get_case_input_notes(db, case_id=job.case_id)
        images = get_case_input_images(db, case_id=job.case_id)
        if not notes and not images:
            advance_job_state(
                db,
                job_id=job.job_id,
                target_state=JobState.failed,
                error_code="missing_input_modalities",
            )
            return {"job_id": job.job_id, "state": "failed", "reason": "missing_input_modalities"}

        rag_ms = 0
        inference_ms = 0
        qc_ms = 0

        try:
            rag_start = time.perf_counter()
            rag_rows = search_docs(db, query=notes, top_k=3) if notes else []
            rag_ms = int((time.perf_counter() - rag_start) * 1000)
            rag_context = build_rag_context(notes, rag_rows)
            inference_start = time.perf_counter()
            inference = run_configured_inference(job.case_id, notes, images, rag_context=rag_context)
            inference_ms = int((time.perf_counter() - inference_start) * 1000)
        except InferenceProviderError as exc:
            advance_job_state(db, job_id=job.job_id, target_state=JobState.failed, error_code=str(exc))
            return {"job_id": job.job_id, "state": "failed", "reason": str(exc)}

        qc_start = time.perf_counter()
        findings_text = " ".join(inference.get("findings", []))
        qc_status, issues = evaluate_findings(findings_text)
        issues_flat = flatten_qc_issues(issues)
        qc_ms = int((time.perf_counter() - qc_start) * 1000)
        total_ms = int((time.perf_counter() - total_start) * 1000)
        output = {
            "case_id": job.case_id,
            "job_id": job.job_id,
            "inference": inference,
            "qc_status": qc_status,
            "qc_issues": issues,
            "qc_issues_flat": issues_flat,
            "observability": {
                "durations_ms": {
                    "rag": rag_ms,
                    "inference": inference_ms,
                    "qc": qc_ms,
                    "total": total_ms,
                },
                "inference_runtime": {
                    "run_mode": inference.get("run_mode"),
                    "model_source": inference.get("model_source"),
                    "generated_token_count": inference.get("generated_token_count"),
                    "used_fallback": inference.get("used_fallback"),
                },
            },
            "rag": {
                "query": notes if notes else None,
                "hits": build_rag_debug(rag_rows),
                "context_used": rag_context,
            },
        }

        if qc_status == "blocked" and not settings.qc_block_fails_job:
            output["qc_status"] = "review_required"
            output["qc_issues"]["safety"].append("qc_block_downgraded_for_debug")
            output["qc_issues_flat"] = flatten_qc_issues(output["qc_issues"])
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
