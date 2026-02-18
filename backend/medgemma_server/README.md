# MedGemma Inference Service

Standalone inference service for Medgent backend.
This service uses the MedGemma-oriented path: `AutoProcessor + AutoModelForImageTextToText`.

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
- `DTYPE` (`float16`/`bfloat16`/`float32`, default: `bfloat16`)
- `LOAD_IN_4BIT` (`true`/`false`, default: `false`)
- `LOAD_IN_8BIT` (`true`/`false`, default: `false`)
- `BNB_4BIT_QUANT_TYPE` (`nf4`/`fp4`, default: `nf4`)
- `BNB_4BIT_USE_DOUBLE_QUANT` (`true`/`false`, default: `true`)
- `USE_CHAT_TEMPLATE` (`true`/`false`, default: `true`)
- `MIN_NEW_TOKENS` (default: `32`)
- `TOP_P` (default: `1.0`)
- `USE_EXPLICIT_EOS_PAD` (`true`/`false`, default: `false`)
- `PORT` (default: `9000`)

For non-quantized test:
- `LOAD_IN_4BIT=false`
- `LOAD_IN_8BIT=false`
- `MAX_NEW_TOKENS=64~128`
- `TEMPERATURE=0.0`

Important for MedGemma 4B:
- Prefer `DTYPE=bfloat16` on RTX 40xx. `float16` can produce NaN logits and pad-only outputs.

If you keep seeing `used_fallback=true`, tune in this order:
1. Keep `USE_CHAT_TEMPLATE=true`
2. Raise `MIN_NEW_TOKENS` to `48`
3. Set `TEMPERATURE=0.2` and `TOP_P=0.9`
4. If still empty, switch to `LOAD_IN_8BIT=true` (or back to 4bit) for A/B check

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
  "confidence": 0.7,
  "used_fallback": false,
  "run_mode": "native_float16",
  "model_source": "/path/to/local/model",
  "generated_token_count": 86,
  "raw_generated_text": "......",
  "raw_generated_text_with_special": "<bos>......<eos>"
}
```

## 5. Connect with backend

In `backend/.env`:

```env
INFERENCE_PROVIDER=medgemma
MEDGEMMA_BASE_URL=http://127.0.0.1:9000
MEDGEMMA_TIMEOUT_SECONDS=60
```
