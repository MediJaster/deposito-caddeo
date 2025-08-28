import os
from typing import Type

from crewai.tools import BaseTool
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from contextlib import contextmanager

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


class RetrievalToolInput(BaseModel):
    """Input schema for RAGTool."""

    query: str = Field(..., description="The query to search in the vector store.")


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


@contextmanager
def get_qdrant_vectorstore():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    embeddings = get_embedding_model()
    vectorstore = QdrantVectorStore(
        client=client, embedding=embeddings, collection_name="board_games"
    )

    try:
        yield vectorstore
    finally:
        client.close()


class RetrievalTool(BaseTool):
    name: str = "Retrieval Tool"
    description: str = "A tool that retrieves relevant information from a Qdrant vector store based on a given query."
    args_schema: Type[BaseModel] = RetrievalToolInput

    def _run(self, query: str) -> str:
        with get_qdrant_vectorstore() as qdrant:
            result = qdrant.similarity_search(query, k=5)

        result_str = "\n---\n".join([doc.page_content for doc in result])

        return result_str
