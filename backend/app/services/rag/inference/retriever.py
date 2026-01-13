from langchain_openai.embeddings import OpenAIEmbeddings
from pymilvus import MilvusClient
from app.core.config import settings


def retrieve_relevant_chunks(
    query: str, encoder: OpenAIEmbeddings, vector_db: MilvusClient
) -> list[list[dict]]:
    query_vector = encoder.embed_query(query)
    res = vector_db.search(
        collection_name=settings.collection_name,
        anns_field="vector",
        output_fields=["text", "dynamic_fields"],
        limit=5,
        data=[query_vector],
    )
    # res = res[0]
    # res = [
    #     {"content": hit.entity.text, "metadata": hit.entity.dynamic_fields}
    #     for hit in res
    # ]
    return res
