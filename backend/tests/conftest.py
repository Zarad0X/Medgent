import os
import sys
from uuid import uuid4
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep tests self-contained regardless of local .env values.
os.environ.setdefault("INFERENCE_PROVIDER", "mock")
os.environ.setdefault("MEDGEMMA_BASE_URL", "http://127.0.0.1:9000")
os.environ["DATABASE_URL"] = f"sqlite:///./test_medgent_{uuid4().hex}.db"
