# MedGemma Inference Service

Standalone inference service for Medgent backend.

## 1. Setup

```bash
cd backend/medgemma_server
uv sync
```

If model access is gated, log in first:

```bash
huggingface-cli login
```

## 2. Configure

Copy and edit environment:

```bash
cp .env.example .env
```

Key variables:
- `MODEL_ID` (default: `google/medgemma-4b-it`)
- `LOCAL_MODEL_DIR` (default: `/home/answer12_as/workspace/Medgent/backend/models/google-medgemma-4b-it`)
- `LOCAL_FILES_ONLY` (`true`/`false`, default: `true`)
- `DEVICE` (`auto`/`cpu`/`cuda`)
- `DTYPE` (`float16`/`bfloat16`/`float32`)
- `LOAD_IN_4BIT` (`true`/`false`, default: `true`)
- `LOAD_IN_8BIT` (`true`/`false`, default: `false`)
- `BNB_4BIT_QUANT_TYPE` (`nf4`/`fp4`, default: `nf4`)
- `BNB_4BIT_USE_DOUBLE_QUANT` (`true`/`false`, default: `true`)
- `PORT` (default: `9000`)

For RTX 4060 8GB, start with:
- `LOAD_IN_4BIT=true`
- `MAX_NEW_TOKENS=64~128`
- `TEMPERATURE=0.0`

Local-first behavior:
- If `LOCAL_MODEL_DIR` contains `config.json`, service loads from local directory first.
- If `LOCAL_MODEL_DIR` points to a parent directory, service will auto-detect the first child folder that contains `config.json`.
- If not found, it falls back to `MODEL_ID`.

## 3. Run

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 9000
```

## 4. Contract

### `POST /infer`

Request:

```json
{
  "case_id": "case-123",
  "notes": "右肺病灶较前变化不大，建议继续随访。"
}
```

Response:

```json
{
  "case_id": "case-123",
  "summary": "......",
  "findings": ["......"],
  "confidence": 0.7
}
```

## 5. Connect with backend

In `backend/.env`:

```env
INFERENCE_PROVIDER=medgemma
MEDGEMMA_BASE_URL=http://127.0.0.1:9000
MEDGEMMA_TIMEOUT_SECONDS=60
```
