"""Client de LLM compartilhado pelos nos do Agente de Modelagem.

Le a API key do OpenRouter de um arquivo `.env` local (nao versionado) via
python-dotenv e expoe uma instancia de `ChatOpenAI` (apontada para a API
OpenAI-compativel do OpenRouter) pronta para uso com
`.with_structured_output(SchemaPydantic)`.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY nao encontrada. Defina-a no arquivo .env na raiz do projeto."
        )

    model = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
    return ChatOpenAI(model=model, api_key=api_key, base_url=_OPENROUTER_BASE_URL)
