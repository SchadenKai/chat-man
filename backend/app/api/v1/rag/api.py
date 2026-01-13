from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

from app.services.rag.inference.retriever import retrieve_relevant_chunks
from client.milvus_client import get_milvus_client
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

from app.services.rag.indexing.bi_encoders import get_bi_encoder_model

router = APIRouter(prefix="/rag")


def token_based_chunker(text: str) -> list[str]:
    chunker = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base", chunk_size=10, chunk_overlap=5
    )
    return chunker.split_text(text)


def markdown_chunker(text: str) -> list[Document]:
    headers_to_split_on = [
        ("#", "Topic"),
        ("##", "Sub Topic"),
        ("###", "Sub Sub Topic"),
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
    return {"total_chunks": len(docs_chunks), "chunks": docs_chunks}


@router.post("/ingest")
async def ingest_document(
    file: UploadFile, vector_db: MilvusClient = Depends(get_milvus_client)
):
    if file.content_type not in ["text/markdown"]:
        raise HTTPException("The file is invalid")
    content = await file.read()
    content = content.decode()

    docs_chunks = markdown_chunker(content)


@router.post("/vector_search")
async def vector_search(
    query: str,
    encoder: OpenAIEmbeddings = Depends(get_bi_encoder_model),
    vector_db: MilvusClient = Depends(get_milvus_client),
) -> dict:
    chunks = retrieve_relevant_chunks(query=query, encoder=encoder, vector_db=vector_db)
    return {"chunks": chunks, "total": len(chunks)}
