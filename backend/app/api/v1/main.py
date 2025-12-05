from fastapi import APIRouter
from .chat.api import router as chat_router
from .rag.api import router as rag_router

api_router = APIRouter(prefix="/v1", tags=["v1"])
api_router.include_router(chat_router, tags=["chat"])
api_router.include_router(rag_router, tags=["rag"])