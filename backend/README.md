# Medgent Backend

FastAPI skeleton for the MedGemma Edge Agent architecture.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

## Worker

Process one queued job:

```bash
uv run python -m app.worker --once
```

Run continuously:

```bash
uv run python -m app.worker
```

## Test

```bash
uv run pytest
```

## Environment

- `APP_ENV` (default: `dev`)
- `DATABASE_URL` (default: `sqlite:///./medgent.db`)
- `API_PREFIX` (default: `/api/v1`)
- `API_KEY` (default: `dev-local-key`)
- `ARTIFACT_DIR` (default: `./data/artifacts`)
- `INFERENCE_PROVIDER` (`mock` or `medgemma`, default: `mock`)
- `MEDGEMMA_BASE_URL` (default: `http://127.0.0.1:9000`)
- `MEDGEMMA_TIMEOUT_SECONDS` (default: `30`)
- `WORKER_POLL_SECONDS` (default: `2`)

## Auth

All non-health APIs require header:

`X-API-Key: <API_KEY>`

## Inference Provider

- `POST /api/v1/inference/mock`: always uses mock provider.
- `POST /api/v1/inference/run`: uses configured provider from `INFERENCE_PROVIDER`.

## End-to-End Workflow

- `POST /api/v1/workflow/submit`: create case + input notes + queued job.
- `GET /api/v1/workflow/jobs/{job_id}/result`: get job status and final output payload if available.
- `GET /api/v1/inference/ping`: check if configured inference provider is reachable.

## Startup Order (Real Model)

Terminal 1 - MedGemma service:

```bash
cd backend/medgemma_server
uv run uvicorn server:app --host 0.0.0.0 --port 9000
```

Terminal 2 - Backend API:

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

Terminal 3 - Worker:

```bash
cd backend
uv run python -m app.worker
```

## E2E Smoke Test

Use bundled standard sample and run:

```bash
cd backend
./scripts/e2e_real_model.sh
```

Optional variables:
- `BASE_URL` (default `http://127.0.0.1:8000`)
- `API_KEY` (default `dev-local-key`)
- `NOTES_FILE` (default `./examples/standard_notes.txt`)

## Make Targets

```bash
cd backend
make help
make run-api
make run-worker
make run-medgemma
make e2e
```
