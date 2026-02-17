from pathlib import Path
import uuid

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
