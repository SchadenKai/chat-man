from pymilvus import DataType, MilvusClient
from pymilvus import IndexType
from app.core.config import settings

client = MilvusClient(settings.milvus_uri)


def get_milvus_client() -> MilvusClient:
    """
    Get the Milvus client instance for dependency injection.
    """
    print("[DEBUG] Connecting with Milvus client")
    # TODO: Check if this implementation is already okay
    return client


def initial_setup(client: MilvusClient):
    """
    Initialize database and collection in vector database (Milvus) upon startup
    """
    if settings.milvus_db_name not in client.list_databases():
        client.create_database(db_name=settings.milvus_db_name)

    client.use_database(db_name=settings.milvus_db_name)

    client.drop_collection(collection_name=settings.collection_name)

    if client.has_collection(settings.collection_name):
        print("[INFO] Collection already exists. Skipping initial setup.")
        return None

    print("[INFO] Defining Schema for collection...")

    # we would just make use of the index from the dataset as the id so that we can upsert easily
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(
        field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
    # TODO: Add sparse vector field

    # TODO: Add built-in function using BM25
    # print("[INFO] Creating built-in functions")
    # schema.add_function()

    print("[INFO] Defining and applying indexing parameters")
    index = client.prepare_index_params()
    index.add_index(
        field_name="vector",
        index_type=IndexType.HNSW,
        index_name="vector_idx",
        metric_type="COSINE",
    )

    print("[INFO] Creating Collection")
    client.create_collection(
        collection_name=settings.collection_name, schema=schema, index_params=index
    )

    client.load_collection(collection_name=settings.collection_name)
    res = client.get_load_state(collection_name=settings.collection_name)
    print(f"[INFO] Successfully created and loaded collection. \nStatus: {res}")
