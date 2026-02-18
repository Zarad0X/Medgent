from pathlib import Path
import uuid
import json

from fastapi import UploadFile

from app.core.config import get_settings


def save_upload_file(case_id: str, upload: UploadFile) -> tuple[str, str]:
    settings = get_settings()
    base_dir = Path(settings.artifact_dir) / case_id
    base_dir.mkdir(parents=True, exist_ok=True)

    original_name = upload.filename or "artifact.bin"
    stored_name = f"{uuid.uuid4()}_{original_name}"
    target = base_dir / stored_name
    data = upload.file.read()
    target.write_bytes(data)
    return original_name, str(target)


def save_text_file(case_id: str, *, prefix: str, file_name: str, content: str) -> str:
    settings = get_settings()
    base_dir = Path(settings.artifact_dir) / case_id
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / f"{prefix}_{uuid.uuid4()}_{file_name}"
    target.write_text(content, encoding="utf-8")
    return str(target)


def save_json_file(case_id: str, *, prefix: str, file_name: str, payload: dict) -> str:
    return save_text_file(
        case_id,
        prefix=prefix,
        file_name=file_name,
        content=json.dumps(payload, ensure_ascii=False),
    )
