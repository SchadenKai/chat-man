from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.milvus_client import create_collection, get_milvus_client
from .api.v1.main import api_router as api_router_v1
from .api.v2.main import api_router as api_router_v2


@asynccontextmanager
async def lifespan(_: FastAPI):
    vector_db = get_milvus_client()
    create_collection(client=vector_db)
    yield


app = FastAPI(
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router_v1)
app.include_router(api_router_v2)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
