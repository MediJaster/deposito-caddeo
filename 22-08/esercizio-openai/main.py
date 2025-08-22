import os
import ssl

import httpx
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from openai import AzureOpenAI


class AzureOpenAISettings(BaseModel):
    API_KEY: str
    ENDPOINT_URL: str
    DEPLOYMENT_NAME: str


class Settings(BaseSettings):
    AZURE_OPENAI: AzureOpenAISettings

    class Config:
        env_file = ".env"
        case_sensitive = False
        separator = "_"


settings = Settings()


def main() -> None:
    # Required to avoid SSL errors when behind ZScaler proxy
    ctx = ssl.create_default_context(
        cafile=os.environ.get("SSL_CERT_FILE"),
    )

    httpx_client = httpx.Client(http2=True, verify=ctx)

    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI.ENDPOINT_URL,
        api_key=settings.AZURE_OPENAI.API_KEY,
        api_version="2025-01-01-preview",
        http_client=httpx_client,
    )

    message_list = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": "Write a Python function that computes the Fibonacci sequence up to n.",
        },
    ]

    # Generate the completion
    completion = client.chat.completions.create(
        model=settings.AZURE_OPENAI.DEPLOYMENT_NAME,
        messages=message_list,
        temperature=0.7,
    )

    print(completion.to_json())


if __name__ == "__main__":
    main()
