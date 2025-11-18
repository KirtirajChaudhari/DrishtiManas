"""FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import health, metadata, predictions

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(metadata.router)
app.include_router(predictions.router)


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Drishti Manas API"}
