import json
from urllib import error, request

from app.core.config import get_settings


def ping_medgemma_health() -> dict:
    settings = get_settings()
    url = f"{settings.medgemma_base_url.rstrip('/')}/health"
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=settings.medgemma_timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return {"reachable": True, "raw": ""}
            return {"reachable": True, "raw": json.loads(body)}
    except error.URLError as exc:
        return {"reachable": False, "error": str(exc)}
    except json.JSONDecodeError:
        return {"reachable": True, "raw": "non_json_response"}
