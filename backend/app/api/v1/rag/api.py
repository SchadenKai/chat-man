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
async def send_message(
    file: UploadFile, vector_db: MilvusClient = Depends(get_milvus_client)
):
    """Feed raw documents to store into the vector database. This contains the whole indexing process of the RAG system"""
    if file is None:
        raise HTTPException("The file is invalid")
    content = await file.read()
    return str(content)


@router.post("/create-database")
async def create_database(
    database_name: str, vector_db: MilvusClient = Depends(get_milvus_client)
):
    vector_db.create_database(db_name=database_name)
    return {
        "status": "ok",
        "description": f"Successfully created database name: {database_name}",
    }


@router.post("/switch-database")
async def switch_database(
    database_name: str, vector_db: MilvusClient = Depends(get_milvus_client)
):
    db_list = vector_db.list_databases()
    if database_name not in db_list:
        raise HTTPException(
            message="The database name is not existing", status_code=404
        )
    vector_db.use_database(database_name)
    return {
        "status": "ok",
        "description": f"Succesfully change database being used into {database_name}",
    }
