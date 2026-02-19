# Medgent

Radiology follow-up copilot on MedGemma.

Medgent is an end-to-end agent pipeline for radiology follow-up analysis with:
- multimodal input (`notes`, `images`, or both),
- retrieval-augmented context (RAG),
- structured model output,
- QC checks and observability.

## Demo Screenshots

Place your hackathon screenshots under `docs/images/` and keep these 3 sections in your submission:

### 1) UI Screenshot
![UI Screenshot](docs/images/ui.png)

### 2) Structured Output Example
![Structured Output](docs/images/structured_output.png)

### 3) RAG Reference Example
![RAG Reference](docs/images/rag_reference.png)

## Quickstart (Mock Mode, 3 Commands)

Use 3 terminals.

### Terminal 1
```bash
cd backend && cp -n .env.example .env && sed -i 's/^INFERENCE_PROVIDER=.*/INFERENCE_PROVIDER=mock/' .env && make run-api
```

### Terminal 2
```bash
cd backend && make run-worker
```

### Terminal 3
```bash
cd backend && make e2e
```

Expected result:
- `job.state` becomes `succeeded` (or controlled QC outcome),
- `output` includes `inference`, `qc_status`, `rag`, and `observability`.

## Real Model Mode (MedGemma)

Use 4 terminals.

### Terminal 1 - MedGemma server
```bash
cd backend/medgemma_server
cp -n .env.example .env
uv run uvicorn server:app --host 0.0.0.0 --port 9000
```

### Terminal 2 - Backend API
```bash
cd backend
cp -n .env.example .env
sed -i 's/^INFERENCE_PROVIDER=.*/INFERENCE_PROVIDER=medgemma/' .env
uv run uvicorn app.main:app --reload --port 8000
```

### Terminal 3 - Worker
```bash
cd backend
uv run python -m app.worker
```

### Terminal 4 - Frontend
```bash
cd frontend
cp -n .env.example .env
npm run dev
```

Open:
- Frontend: `http://127.0.0.1:5173`
- Backend health: `http://127.0.0.1:8000/api/v1/health/ready`
- MedGemma health: `http://127.0.0.1:9000/health`

## Kaggle Runtime Guide

Typical Kaggle path convention:
- model files are mounted under `/kaggle/input/<dataset-or-model-name>/`

### 1) Put model files in Kaggle input
Example:
- `/kaggle/input/google-medgemma-4b-it/config.json`
- `/kaggle/input/google-medgemma-4b-it/model-*.safetensors`
- tokenizer / processor files in the same directory.

### 2) Configure MedGemma server env
In `backend/medgemma_server/.env`:

```env
MODEL_ID=google/medgemma-4b-it
LOCAL_MODEL_DIR=/kaggle/input/google-medgemma-4b-it
LOCAL_FILES_ONLY=true
DEVICE=cuda
DTYPE=bfloat16
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
```

### 3) Configure backend env
In `backend/.env`:

```env
INFERENCE_PROVIDER=medgemma
MEDGEMMA_BASE_URL=http://127.0.0.1:9000
MEDGEMMA_TIMEOUT_SECONDS=60
```

### 4) Start services in order
1. `medgemma_server`
2. `backend API`
3. `worker`
4. `frontend` (optional for demo)

## Architecture

See `architecture.md` for system design and component boundaries.

## Safety Disclaimer

This project is for research, prototyping, and hackathon demonstration only.
It is **not** a medical device and **not** a substitute for professional clinical judgment.
Do not use this system as the sole basis for diagnosis or treatment decisions.

