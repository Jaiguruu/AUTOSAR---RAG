from typing import Optional, List, Dict, Any
import re
from phi.assistant import Assistant
from phi.llm.ollama import Ollama
from phi.llm.openai import OpenAIChat
from phi.llm.openrouter import OpenRouter
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.vectordb.qdrant import Qdrant
from phi.knowledge import AssistantKnowledge
from qdrant_client import QdrantClient

# Compatibility patch for qdrant-client >= 1.10+ where .search was replaced by .query_points
if not hasattr(QdrantClient, "search"):
    def _compat_search(
        self,
        collection_name: str,
        query_vector: Any,
        limit: int = 5,
        with_vectors: bool = True,
        with_payload: bool = True,
        query_filter: Any = None,
        **kwargs: Any,
    ):
        res = self.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
            query_filter=query_filter,
            **kwargs,
        )
        return res.points

    QdrantClient.search = _compat_search  # type: ignore

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


def get_rag_assistant(
    llm_model: str = "qwen2.5-coder:1.5b",
    embeddings_model: str = "all-minilm:l6-v2",
    instructions: Optional[str] = None,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    debug_mode: bool = False,
) -> Assistant:
    """Initialize local RAG Assistant configured with Ollama or cloud providers."""

    # 1. Configure the Embedder and its dimension
    if "liquid/lfm" in embeddings_model or "openrouter" in embeddings_model:
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key required for OpenRouter embeddings.")
        embedder = OpenAIEmbedder(
            model=embeddings_model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            dimensions=1024,
        )
    elif "all-minilm" in embeddings_model.lower():
        embedder = OllamaEmbedder(
            model=embeddings_model,
            dimensions=384,
        )
    elif "nomic-embed-text" in embeddings_model.lower():
        embedder = OllamaEmbedder(
            model="nomic-embed-text",
            dimensions=768,
        )
    elif "phi3" in embeddings_model.lower():
        embedder = OllamaEmbedder(
            model="phi3",
            dimensions=3072,
        )
    elif "llama3" in embeddings_model.lower():
        embedder = OllamaEmbedder(
            model="llama3",
            dimensions=4096,
        )
    else:
        # Default fallback for custom local ollama embedding models
        embedder = OllamaEmbedder(
            model=embeddings_model,
            dimensions=384,
        )

    # 2. Vector DB configuration (isolate collections per embedding model to avoid dimension mismatches)
    clean_embed_slug = re.sub(r"[^a-zA-Z0-9_]", "_", embeddings_model)
    collection_name = f"autosar_rag_{clean_embed_slug}"
    qdrant_url = "http://localhost:6333"
    api_key = "123456"

    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        api_key=api_key,
        embedder=embedder,
    )

    # Automatically ensure collection exists in Qdrant
    try:
        if not vector_db.exists():
            vector_db.create()
    except Exception:
        pass

    knowledge = AssistantKnowledge(
        vector_db=vector_db,
        num_documents=3,
    )

    # 3. LLM Configuration (qwen2.5-coder:1.5b via local Ollama prioritized)
    if llm_model.startswith("openrouter/"):
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key not provided.")
        llm = OpenRouter(model=llm_model, api_key=openrouter_api_key, stop="</answer>")
    elif llm_model == "gpt-3.5-turbo":
        llm = OpenAIChat(model="gpt-3.5-turbo", stop="</answer>")
    else:
        # Local Ollama model (qwen2.5-coder:1.5b, llama3, etc.)
        llm = Ollama(model=llm_model)

    # 4. Instructions
    default_instructions = [
        "You are an expert AUTOSAR software architect and engineering assistant.",
        "Provide direct, accurate, and concise answers based on the provided AUTOSAR documentation.",
        "Do not invent information outside the context or AUTOSAR specifications.",
        "Always reply in the same language as the user's question.",
    ]
    if instructions:
        if "\n" in instructions:
            default_instructions.extend([line.strip() for line in instructions.split("\n") if line.strip()])
        else:
            default_instructions.append(instructions.strip())

    return Assistant(
        name="local_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=llm,
        storage=PgAssistantStorage(table_name="local_rag_assistant", db_url=db_url),
        knowledge_base=knowledge,
        description="Fast local AUTOSAR Assistant powered by Qwen2.5-Coder",
        instructions=default_instructions,
        add_references_to_prompt=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )