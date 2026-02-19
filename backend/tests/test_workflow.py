from fastapi.testclient import TestClient
from uuid import uuid4

from app.main import app
from app.worker import process_next_job

AUTH_HEADERS = {"X-API-Key": "dev-local-key"}


def process_until_job(job_id: str, max_steps: int = 10):
    seen = []
    for _ in range(max_steps):
        result = process_next_job()
        if result is None:
            break
        seen.append(result["job_id"])
        if result["job_id"] == job_id:
            return result
    return None


def test_case_and_job_idempotency_flow():
    with TestClient(app) as client:
        case_resp = client.post(
            "/api/v1/cases",
            json={"patient_pseudo_id": "p-001"},
            headers=AUTH_HEADERS,
        )
        assert case_resp.status_code == 201
        case_id = case_resp.json()["case_id"]

        payload = {
            "case_id": case_id,
            "stage": "inference",
            "idempotency_key": f"idem-{uuid4()}",
        }
        first = client.post("/api/v1/jobs", json=payload, headers=AUTH_HEADERS)
        second = client.post("/api/v1/jobs", json=payload, headers=AUTH_HEADERS)
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["job_id"] == second.json()["job_id"]

        job_id = first.json()["job_id"]
        to_running = client.post(
            f"/api/v1/jobs/{job_id}/advance",
            json={"target_state": "running"},
            headers=AUTH_HEADERS,
        )
        assert to_running.status_code == 200
        assert to_running.json()["state"] == "running"

        to_succeeded = client.post(
            f"/api/v1/jobs/{job_id}/advance",
            json={"target_state": "succeeded"},
            headers=AUTH_HEADERS,
        )
        assert to_succeeded.status_code == 200
        assert to_succeeded.json()["state"] == "succeeded"


def test_idempotency_scoped_within_case():
    idem_key = f"idem-shared-{uuid4()}"
    with TestClient(app) as client:
        case_a = client.post(
            "/api/v1/cases",
            json={"patient_pseudo_id": "p-a"},
            headers=AUTH_HEADERS,
        )
        case_b = client.post(
            "/api/v1/cases",
            json={"patient_pseudo_id": "p-b"},
            headers=AUTH_HEADERS,
        )
        assert case_a.status_code == 201
        assert case_b.status_code == 201

        job_a = client.post(
            "/api/v1/jobs",
            json={
                "case_id": case_a.json()["case_id"],
                "stage": "inference",
                "idempotency_key": idem_key,
            },
            headers=AUTH_HEADERS,
        )
        job_b = client.post(
            "/api/v1/jobs",
            json={
                "case_id": case_b.json()["case_id"],
                "stage": "inference",
                "idempotency_key": idem_key,
            },
            headers=AUTH_HEADERS,
        )
        assert job_a.status_code == 201
        assert job_b.status_code == 201
        assert job_a.json()["job_id"] != job_b.json()["job_id"]
        assert job_a.json()["case_id"] != job_b.json()["case_id"]


def test_simple_rag_qc_and_mock_inference():
    with TestClient(app) as client:
        ingest = client.post(
            "/api/v1/rag/ingest",
            json={
                "source": "hospital_sop",
                "source_version": "2026.01",
                "title": "肺结节随访",
                "content": "病灶变化评估建议结合历史影像对比。",
            },
            headers=AUTH_HEADERS,
        )
        assert ingest.status_code == 200

        search = client.post(
            "/api/v1/rag/search",
            json={"query": "病灶 变化", "top_k": 3},
            headers=AUTH_HEADERS,
        )
        assert search.status_code == 200
        assert len(search.json()["items"]) >= 1

        qc = client.post(
            "/api/v1/qc/evaluate",
            json={"findings": "病灶较前缩小，变化趋势稳定。"},
            headers=AUTH_HEADERS,
        )
        assert qc.status_code == 200
        assert qc.json()["status"] in {"pass", "review_required", "blocked"}

        infer = client.post(
            "/api/v1/inference/mock",
            json={"case_id": "c1", "notes": "右肺病灶较前变化不大，建议继续随访。"},
            headers=AUTH_HEADERS,
        )
        assert infer.status_code == 200
        assert infer.json()["case_id"] == "c1"

        configured = client.post(
            "/api/v1/inference/run",
            json={"case_id": "c2", "notes": "左肺病灶随访。"},
            headers=AUTH_HEADERS,
        )
        assert configured.status_code == 200
        assert configured.json()["case_id"] == "c2"


def test_inference_modalities_validation():
    with TestClient(app) as client:
        image_only = client.post(
            "/api/v1/inference/mock",
            json={"case_id": "i1", "images": ["/tmp/fake-image.png"]},
            headers=AUTH_HEADERS,
        )
        assert image_only.status_code == 200

        invalid = client.post(
            "/api/v1/inference/mock",
            json={"case_id": "i2", "notes": "", "images": []},
            headers=AUTH_HEADERS,
        )
        assert invalid.status_code == 422


def test_workflow_submit_and_worker_end_to_end():
    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/workflow/submit",
            json={
                "patient_pseudo_id": f"p-{uuid4()}",
                "notes": "右肺病灶较前变化不大，建议继续随访。",
                "idempotency_key": f"wf-{uuid4()}",
            },
            headers=AUTH_HEADERS,
        )
        assert submit.status_code == 201
        job_id = submit.json()["job"]["job_id"]
        assert submit.json()["job"]["state"] == "queued"

        before = client.get(f"/api/v1/workflow/jobs/{job_id}/result", headers=AUTH_HEADERS)
        assert before.status_code == 200
        assert before.json()["output"] is None

        worker_result = process_until_job(job_id)
        assert worker_result is not None
        assert worker_result["job_id"] == job_id
        assert worker_result["state"] in {"succeeded", "failed"}

        after = client.get(f"/api/v1/workflow/jobs/{job_id}/result", headers=AUTH_HEADERS)
        assert after.status_code == 200
        assert after.json()["output"] is not None
        output = after.json()["output"]
        assert "rag" in output
        assert "observability" in output
        assert "durations_ms" in output["observability"]


def test_workflow_submit_image_only(tmp_path):
    image_path = tmp_path / "image1.png"
    image_path.write_bytes(b"fake-image-bytes")

    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/workflow/submit",
            json={
                "patient_pseudo_id": f"p-{uuid4()}",
                "images": [str(image_path)],
                "idempotency_key": f"wf-{uuid4()}",
            },
            headers=AUTH_HEADERS,
        )
        assert submit.status_code == 201
        job_id = submit.json()["job"]["job_id"]

        worker_result = process_until_job(job_id)
        assert worker_result is not None
        assert worker_result["job_id"] == job_id
        assert worker_result["state"] in {"succeeded", "failed"}


def test_workflow_submit_requires_modalities():
    with TestClient(app) as client:
        submit = client.post(
            "/api/v1/workflow/submit",
            json={
                "patient_pseudo_id": f"p-{uuid4()}",
                "notes": "",
                "images": [],
                "idempotency_key": f"wf-{uuid4()}",
            },
            headers=AUTH_HEADERS,
        )
        assert submit.status_code == 422
