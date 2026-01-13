from langchain_text_splitters import MarkdownTextSplitter, TextSplitter
from langchain_core.documents import Document

md_chunker = MarkdownTextSplitter()


def get_md_chunker() -> TextSplitter:
    return md_chunker


# TODO: transform this into a strategy class instead
def chunk_texts(text: str, chunker: TextSplitter) -> list[str]:
    return chunker.split_text(text)


def chunk_documents(docs: Document, chunker: TextSplitter) -> list[Document]:
    return chunker.split_documents(docs)
