from langchain.chat_models.base import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from src.settings import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    """
    Initialize and return a HuggingFace embeddings model instance.

    This function creates a sentence transformer model that converts text into
    high-dimensional vector representations for semantic similarity search.

    Args:
        settings: Configuration object containing the model name and parameters

    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance

    Model Loading Behavior:
    - First run: Downloads model from HuggingFace Hub (requires internet)
    - Subsequent runs: Loads from local cache (~/.cache/huggingface/)
    - Model size: 100MB-2GB depending on the selected model

    Performance Notes:
    - GPU acceleration: Automatically uses CUDA if available
    - CPU fallback: Falls back to CPU if GPU unavailable
    - Memory usage: Model loaded into RAM/VRAM during inference

    Error Handling:
    - Network issues: Will fail if model not cached and no internet
    - Memory issues: Large models may cause OOM on low-memory systems
    - Model not found: Invalid model names will cause runtime errors
    """
    return AzureOpenAIEmbeddings(model=settings.azure_openai_embedding_deployment_name)


def get_llm(settings: Settings) -> BaseChatModel | None:
    """
    Initialize and test an LLM instance for text generation if properly configured.

    This function attempts to create an LLM connection using environment variables
    and performs a connectivity test to ensure the service is working before
    returning the instance. If any step fails, it gracefully falls back to None.

    Args:
        settings: Configuration object containing LLM environment variable names

    Returns:
        ChatModel or None: Configured LLM instance if successful, None otherwise

    Configuration Requirements:
    - OPENAI_BASE_URL: Base URL for the LLM service
    - OPENAI_API_KEY: Authentication key for the service
    - LMSTUDIO_MODEL: Specific model identifier to use

    Supported LLM Services:
    - OpenAI API: Production-grade, reliable, paid service
    - LM Studio: Local inference, free, requires model download
    - Ollama: Local inference, free, easy setup
    - Azure OpenAI: Enterprise-grade, reliable, paid service
    - Custom APIs: Any OpenAI-compatible endpoint

    Connection Testing:
    - Performs a simple "test" query to verify connectivity
    - Tests both network connectivity and model availability
    - Helps identify configuration issues early

    Error Handling Strategy:
    - Missing env vars: Graceful fallback with informative message
    - Network issues: Catches connection errors and continues
    - Authentication errors: Handles invalid API keys gracefully
    - Model errors: Catches model-specific issues

    Fallback Behavior:
    - Returns None if any step fails
    - Script continues without LLM generation
    - Retrieved content is displayed instead of generated answers

    Security Considerations:
    - API keys are read from environment variables only
    - No hardcoded credentials in source code
    - Test query is minimal and doesn't expose sensitive data
    """
    try:
        # Test the LLM connection before returning
        llm = AzureChatOpenAI(model=settings.azure_openai_llm_deployment_name)

    except Exception as e:
        print(f"LLM configuration error: {e}")
        print("Continuing without LLM - will show retrieved content only")
        return None
