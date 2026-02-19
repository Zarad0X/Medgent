from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import public_router, router
from app.core.config import get_settings
from app.db import Base, engine

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
allow_origins = [item.strip() for item in settings.cors_allow_origins.split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(public_router, prefix=settings.api_prefix)
app.include_router(router, prefix=settings.api_prefix)

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(static_dir / "favicon.ico")
