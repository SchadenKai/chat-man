from fastapi import Depends
from pymilvus import DataType, MilvusClient
from pymilvus import model
from pymilvus.model.base import BaseEmbeddingFunction

client = MilvusClient("http://localhost:19530")


def get_milvus_client() -> MilvusClient:
    return client


def create_collection(client: MilvusClient | None = Depends(get_milvus_client)):
    if client.has_collection("demo_collection"):
        return None
    client.create_collection(
        collection_name="demo_collection",
        dimension=768,
        auto_id=True,
        primary_field_name="id",
    )

    schema = client.create_schema(
        enable_dynamic_field=True
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    


def get_embedding_model() -> BaseEmbeddingFunction:
    return model.DefaultEmbeddingFunction()


def sample_embedding(
    client: MilvusClient,
    embedding_fn: BaseEmbeddingFunction = Depends(get_embedding_model),
):
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    vectors = embedding_fn.encode_documents(docs)
    return vectors
