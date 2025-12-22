from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from langchain_core.language_models import BaseChatModel
from pymilvus import MilvusClient
from app.agent.react_agent.graph import call_agent
from ag_ui.encoder import EventEncoder
from ag_ui.core import RunAgentInput

from app.agent.factory import get_default_llm
from app.core.milvus_client import get_milvus_client, sample_embedding

router = APIRouter(prefix="/rag")


@router.post("/ingest")
async def send_message(file: UploadFile, vector_db : MilvusClient =Depends(get_milvus_client)): 
    """Feed raw documents to store into the vector database. This contains the whole indexing process of the RAG system"""
    file_obj = file.file
    if file_obj is None:
        raise HTTPException("The file is invalid")
    print(file_obj.read())
    return vector_db.list_collections(), vector_db.list_databases()
    