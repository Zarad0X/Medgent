from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import public_router, router
from app.core.config import get_settings
from app.db import Base, engine

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(public_router, prefix=settings.api_prefix)
app.include_router(router, prefix=settings.api_prefix)
