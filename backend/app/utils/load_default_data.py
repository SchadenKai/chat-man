from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from pandas import DataFrame
from pymilvus import MilvusClient

import kagglehub
from kagglehub import KaggleDatasetAdapter

from langchain_core.documents import Document
import ast

from app.core.config import settings


DOC_MD_TEMPLATE = """
# {food_name}

## Ingredients:
{ingredients}

## Directions:
{directions}
"""


def load_default_data(
    encoder: OpenAIEmbeddings, vector_db: MilvusClient
) -> None | Exception:
    try:
        df: DataFrame = kagglehub.dataset_load(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle="paultimothymooney/recipenlg",
            path="RecipeNLG_dataset.csv",
        )
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        return e

    raw_data = df.values.tolist()

    raw_docs: list[Document] = []
    final_docs: list[dict] = []

    for i, row in enumerate(raw_data):
        if i > 50:
            break

        title = row[1]
        link = row[4]
        source = row[5]
        ner = row[6]
        # convert list of type strings into list
        ingredients = ast.literal_eval(row[2])
        directions = ast.literal_eval(row[3])

        text_md = DOC_MD_TEMPLATE.format(
            ingredients=ingredients, directions=directions, food_name=title
        )

        doc = Document(
            page_content=text_md.replace("\n", "\\n"),
            metadata={
                "food_name": title,
                "link": link,
                "source": source,
                "raw_ner": ner,
            },
        )
        raw_docs.append(doc)

    chunker1 = MarkdownTextSplitter()
    chunked_docs_2 = chunker1.split_documents(raw_docs)

    for doc in chunked_docs_2:
        data = {
            "text": doc.page_content,
            "vector": encoder.embed_query(doc.page_content),
            **doc.metadata,
        }
        final_docs.append(data)

    try:
        vector_db.insert(collection_name=settings.collection_name, data=final_docs)
    except Exception as e:
        print(f"Something went wrong in storing vectors: {e}")
        return e
