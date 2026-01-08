from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from langchain_core.language_models import BaseChatModel
from pymilvus import MilvusClient
from app.agent.react_agent.graph import call_agent
from ag_ui.encoder import EventEncoder
from ag_ui.core import RunAgentInput

from app.agent.factory import get_default_llm
from client.milvus_client import get_milvus_client, sample_embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

router = APIRouter(prefix="/rag")

def token_based_chunker(text: str) -> list[str]:
    chunker = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=10,
        chunk_overlap=5
    )
    return chunker.split_text(text)

def markdown_chunker(text: str) -> list[Document]:
    headers_to_split_on = [
        ("#", "Topic"),
        ("##", "Sub Topic"),
        ("###", "Sub Sub Topic")
    ]
    chunker = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return chunker.split_text(text)

def charac_doc_chunker(document: Document) -> list[Document]:
    chunk_size = 250
    chunk_overlap = 30
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return chunker.split_documents(document)

@router.post("/sample_chunker")
async def send_message(
    file: UploadFile, vector_db: MilvusClient = Depends(get_milvus_client)
):
    """Feed raw documents to store into the vector database. This contains the whole indexing process of the RAG system"""
    if file.content_type not in ["text/markdown"]:
        raise HTTPException("The file is invalid")
    content = await file.read()
    content = content.decode()

    docs_chunks = markdown_chunker(content)
    docs_chunks = charac_doc_chunker(docs_chunks)
    return {
        "total_chunks" : len(docs_chunks),
        "chunks" : docs_chunks
    }

@router.post("/ingest")
async def ingest_document(
    file: UploadFile, vector_db: MilvusClient = Depends(get_milvus_client)
):
    if file.content_type not in ["text/markdown"]:
        raise HTTPException("The file is invalid")
    content = await file.read()
    content = content.decode()

    docs_chunks = markdown_chunker(content)

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
