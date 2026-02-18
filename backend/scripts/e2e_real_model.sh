#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-dev-local-key}"
NOTES_FILE="${NOTES_FILE:-./examples/standard_notes.txt}"

if [[ ! -f "$NOTES_FILE" ]]; then
  echo "notes file not found: $NOTES_FILE"
  exit 1
fi

echo "[1/4] Ping configured inference provider"
curl -sS -H "X-API-Key: ${API_KEY}" "${BASE_URL}/api/v1/inference/ping"
echo

echo "[2/4] Submit workflow"
SUBMIT_PAYLOAD="$(
  NOTES_FILE="$NOTES_FILE" python - <<'PY'
import json
import os
from pathlib import Path

notes = Path(os.environ["NOTES_FILE"]).read_text(encoding="utf-8")
payload = {
    "patient_pseudo_id": "p-e2e-001",
    "notes": notes,
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
SUBMIT_RESP="$(
  curl -sS -X POST "${BASE_URL}/api/v1/workflow/submit" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d "$SUBMIT_PAYLOAD"
)"
echo "$SUBMIT_RESP"

JOB_ID="$(
  SUBMIT_RESP="$SUBMIT_RESP" python - <<'PY'
import json
import os

resp = os.environ["SUBMIT_RESP"]
try:
    data = json.loads(resp)
    print(data["job"]["job_id"])
except Exception:
    print("")
PY
)"
if [[ -z "$JOB_ID" ]]; then
  echo "failed to parse job_id from workflow submit response"
  exit 1
fi
echo "job_id=${JOB_ID}"

echo "[3/4] Poll workflow result"
for i in {1..30}; do
  RESP="$(curl -sS -H "X-API-Key: ${API_KEY}" "${BASE_URL}/api/v1/workflow/jobs/${JOB_ID}/result")"
  STATE="$(
    RESP="$RESP" python - <<'PY'
import json
import os

resp = os.environ["RESP"]
try:
    data = json.loads(resp)
    print(data["job"]["state"])
except Exception:
    print("")
PY
  )"
  if [[ "$STATE" == "succeeded" || "$STATE" == "failed" ]]; then
    echo "$RESP"
    break
  fi
  sleep 2
done

echo "[4/4] Done"
