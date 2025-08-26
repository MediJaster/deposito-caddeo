from __future__ import annotations

import httpx
from langchain_openai.chat_models.azure import AzureChatOpenAI
import os
from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from dotenv import load_dotenv


# =========================
# Configurazione
# =========================

load_dotenv()

FAISS_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "faiss_index_example")


@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = FAISS_PERSIST_DIR

    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Retriever (MMR)
    search_type: str = "mmr"  # "mmr" o "similarity"
    k: int = 4  # risultati finali
    fetch_k: int = 20  # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3  # 0 = diversificazione massima, 1 = pertinenza massima

    # Embedding
    embedding_deployment_name: str = "text-embedding-ada-002"

    # LLM
    llm_deployment_name: str = "gpt-4.1"


SETTINGS = Settings()

# =========================
# Componenti di base
# =========================

httpx_client = httpx.Client(http2=True, verify=False)


def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """
    Restituisce un modello di embedding locale e gratuito (Hugging Face).
    """
    endpoint_url = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    return AzureOpenAIEmbeddings(
        http_client=httpx_client,
        azure_endpoint=endpoint_url,
        model=settings.embedding_deployment_name,
        api_key=api_key,
        api_version="2023-05-15",
    )


def get_llm_from_azure_openai(settings: Settings) -> AzureChatOpenAI:
    endpoint_url = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT_URL e AZURE_OPENAI_API_KEY devono essere impostate per Azure OpenAI."
        )

    return AzureChatOpenAI(
        http_client=httpx_client,
        azure_endpoint=endpoint_url,
        model=settings.llm_deployment_name,
        api_key=api_key,
        api_version="2025-01-01-preview",
    )


def simulate_corpus() -> list[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"},
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"},
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"},
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"},
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"},
        ),
    ]
    return docs


def load_real_documents_from_folder(folder_path: str) -> list[Document]:
    """
    Carica documenti reali da file di testo (es. .txt, .md) all'interno di una cartella.
    Ogni file viene letto e convertito in un oggetto Document con metadato 'source'.
    """
    folder = Path(folder_path)
    documents: list[Document] = []

    if not folder.exists() or not folder.is_dir():
        raise ValueError(
            f"La cartella '{folder_path}' non esiste o non è una directory."
        )

    for file_path in folder.glob("**/*"):
        if file_path.suffix.lower() not in [".txt", ".md"]:
            continue  # ignora file non supportati

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        # Aggiunge il metadato 'source' per citazioni (es. nome del file)
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    return documents


def split_markdown_on_dashes(docs: list[Document]) -> list[Document]:
    """
    Splits each Document in the input list at every '---' using LangChain's MarkdownHeaderTextSplitter.
    Returns a flat list of Document objects.
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("---", "section")])
    split_docs = []
    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        # preserve original metadata
        for s in splits:
            s.metadata = dict(doc.metadata)
        split_docs.extend(splits)

    print(split_docs)
    print(f"N Docs: {len(split_docs)}")
    return split_docs


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "---",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(
    chunks: list[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str
) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(
    settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: list[Document]
) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir, embeddings, allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    # chunks = split_markdown_on_dashes(docs=docs)
    # print(chunks)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda,
            },
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: list[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm: AzureChatOpenAI, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano."
        "Usa esclusivamente il CONTENUTO fornito nel contesto."
        "Se l'informazione non è presente, dichiara che non è disponibile."
        "Includi citazioni tra parentesi quadre nel formato [source:...]."
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Domanda:\n{question}\n\n"
                "Contesto (estratti selezionati):\n{context}\n\n"
                "Istruzioni:\n"
                "1) Rispondi solo con informazioni contenute nel contesto.\n"
                "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
                "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'",
            ),
        ]
    )

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)


# RAGAS
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,  # "precision@k" sui chunk recuperati
    context_recall,  # copertura dei chunk rilevanti
    faithfulness,  # ancoraggio della risposta al contesto
    answer_relevancy,  # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)


def get_contexts_for_question(retriever, question: str, k: int) -> list[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]


def build_ragas_dataset(
    questions: list[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


def run_ragas(settings, chain, llm, retriever):
    # 5) Esempi di domande
    questions = [
        "Che cos'è una pipeline RAG e quali sono le sue fasi principali?",
        "A cosa serve FAISS e quali capacità offre?",
        "Cos'è MMR e perché è utile durante il retrieval?",
        "Quale dimensione hanno gli embedding prodotti da all-MiniLM-L6-v2?",
    ]

    # (opzionale) ground truth sintetica per correctness
    ground_truth = {
        questions[
            0
        ]: "Indicizzazione (caricamento, splitting, embedding, storage) e retrieval + generazione.",
        questions[
            1
        ]: "Libreria per ricerca di similarità e clustering di vettori densi (ANN/NNN) scalabile.",
        questions[
            2
        ]: "Bilancia pertinenza e diversità per ridurre ridondanza e coprire aspetti differenti.",
        questions[3]: "384",
    }

    # 6) Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 7) Scegli le metriche
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,  # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=get_embeddings(settings),  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = [
        "user_input",
        "response",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")


# =========================
# Esecuzione dimostrativa
# =========================


def main():
    settings = SETTINGS

    # 1) Componenti
    embeddings = get_embeddings(settings)
    llm = get_llm_from_azure_openai(settings)

    # 2) Dati simulati e indicizzazione (load or build)
    docs = simulate_corpus()
    # docs = load_real_documents_from_folder(
    #     os.path.join(os.path.dirname(__file__), "documents")
    # )
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    # 5) Esempi di domande
    questions = [
        "Dimmi la formula chimica dell'acqua",
        "Il pianeta più vicino al Sole?",
        "Che lingua parlano nella nazione dove si trova Parigi?",
        'Chi è l\'autore della "Divina Commedia"?',
    ]

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = rag_answer(q, chain)
        print(ans)
        print()

    run_ragas(settings, chain, llm, retriever)


if __name__ == "__main__":
    main()
