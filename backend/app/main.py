
##FastAPI entry point for EconoCausal.
from fastapi import FastAPI
from backend.app.api.routes import router

app = FastAPI(
    title="EconoCausal API",
    version="0.2.0",
    description="Causal-effect estimation API for EconoCausal.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "stage": "api-integrated"}


app.include_router(router, prefix="/api/v1", tags=["Causal Model"])
