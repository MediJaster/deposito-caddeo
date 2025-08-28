import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents.base import Document

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_openai import AzureOpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


def create_collection(embedding: Embeddings, documents: list[Document]):
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    client = QdrantClient(url=qdrant_url)

    if client.collection_exists(collection_name="board_games"):
        _ = client.delete_collection(collection_name="board_games")

    client.close()

    _ = QdrantVectorStore.from_documents(
        url=qdrant_url,
        documents=documents,
        embedding=embedding,
        collection_name="board_games",
    )


def get_embedding_model() -> AzureOpenAIEmbeddings:
    endpoint_url = os.getenv("AZURE_API_BASE")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION")

    if not endpoint_url or not api_key or not api_version:
        raise ValueError(
            "Please set the AZURE_API_BASE, AZURE_API_KEY, and AZURE_API_VERSION environment variables."
        )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint_url,
        api_key=api_key,
        api_version=api_version,
        model="text-embedding-ada-002",
        chunk_size=1,
    )

    return embeddings


DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "documents")


def load_pdf_from_folder() -> list[Document]:
    loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    documents = loader.load()

    return documents


def split_docs(documents: list[Document]) -> list[Document]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(documents)

    return docs


def main():
    documents = load_pdf_from_folder()
    docs = split_docs(documents)

    embedding = get_embedding_model()

    create_collection(embedding=embedding, documents=docs)


if __name__ == "__main__":
    main()
