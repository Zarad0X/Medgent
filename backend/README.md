# Medgent Backend

FastAPI skeleton for the MedGemma Edge Agent architecture.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
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

## Auth

All non-health APIs require header:

`X-API-Key: <API_KEY>`

## Inference Provider

- `POST /api/v1/inference/mock`: always uses mock provider.
- `POST /api/v1/inference/run`: uses configured provider from `INFERENCE_PROVIDER`.
