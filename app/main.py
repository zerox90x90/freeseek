"""FastAPI app factory."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.deepseek.client import DeepSeekClient
from app.routes.anthropic import router as anthropic_router
from app.routes.openai_chat import router as openai_router
from app.routes.openai_files import router as files_router
from app.routes.openai_responses import router as responses_router
from app.routes.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ds = DeepSeekClient()
    try:
        yield
    finally:
        await app.state.ds.aclose()


app = FastAPI(title="zero-proxy", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(files_router)
app.include_router(responses_router)
app.include_router(sessions_router)


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
