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
    """
    Restituisce il modello di embedding Azure OpenAI configurato via variabili d'ambiente.

    Parameters
    ----------
    None

    Returns
    -------
    AzureOpenAIEmbeddings
        Istanza configurata con `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION`.

    Raises
    ------
    ValueError
        Se una tra `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION` non è impostata.

    Notes
    -----
    Complessità temporale: O(1).
    Complessità spaziale: O(1).

    Examples
    --------
    >>> import os
    >>> from rag_or_web_flow.tools.rag_tool import get_embedding_model
    >>> saved = {k: os.environ.pop(k, None) for k in ("AZURE_API_BASE","AZURE_API_KEY","AZURE_API_VERSION")}
    >>> try:
    ...     get_embedding_model()
    ... except ValueError as e:
    ...     "AZURE_API_BASE" in str(e) or "AZURE_API_KEY" in str(e) or "AZURE_API_VERSION" in str(e)
    ... finally:
    ...     _ = [os.environ.__setitem__(k, v) for k, v in saved.items() if v is not None]
    True
    """
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
        """
        Esegue una ricerca di similarità su Qdrant e restituisce i contenuti dei documenti.

        Parameters
        ----------
        query : str
            Query da cercare nella collezione Qdrant. Unità: adimensionale.
            Range: stringa non vuota.

        Returns
        -------
        str
            Concatenazione dei contenuti dei documenti trovati, separati da "---".
            Unità: adimensionale.

        Raises
        ------
        ValueError
            Se `query` è vuota o composta solo da spazi.
        ValueError
            Se le variabili d'ambiente richieste da `get_embedding_model` non sono impostate.

        Notes
        -----
        Complessità temporale: O(k) con k = numero di documenti richiesti (qui k = 5).
        Complessità spaziale: O(k).

        Examples
        --------
        >>> from rag_or_web_flow.tools.rag_tool import RetrievalTool
        >>> # L'esempio richiede un'istanza Qdrant raggiungibile e le variabili d'ambiente settate.
        >>> isinstance(RetrievalTool()._run("test"), str)  # doctest: +SKIP
        True
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        with get_qdrant_vectorstore() as qdrant:
            result = qdrant.similarity_search(query, k=5)

        result_str = "\n---\n".join([doc.page_content for doc in result])

        return result_str
