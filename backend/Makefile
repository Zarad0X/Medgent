SHELL := /bin/bash

.PHONY: help run-api run-worker run-worker-once run-medgemma run-frontend frontend-install test e2e

help:
	@echo "Targets:"
	@echo "  make run-api         - start backend API server"
	@echo "  make run-worker      - start workflow worker loop"
	@echo "  make run-worker-once - process one queued job"
	@echo "  make run-medgemma    - start medgemma inference service"
	@echo "  make frontend-install- install frontend dependencies"
	@echo "  make run-frontend    - start frontend dev server"
	@echo "  make test            - run backend tests"
	@echo "  make e2e             - run end-to-end smoke script"

run-api:
	uv run uvicorn app.main:app --reload --port 8000

run-worker:
	uv run python -m app.worker

run-worker-once:
	uv run python -m app.worker --once

run-medgemma:
	cd medgemma_server && uv run uvicorn server:app --host 0.0.0.0 --port 9000

frontend-install:
	cd ../frontend && npm install

run-frontend:
	cd ../frontend && npm run dev

test:
	uv run --extra dev pytest

e2e:
	./scripts/e2e_real_model.sh
