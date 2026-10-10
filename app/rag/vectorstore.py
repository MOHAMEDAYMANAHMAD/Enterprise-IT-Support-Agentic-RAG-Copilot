from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from app.core.config import get_settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


settings = get_settings()


_embeddings = None
_vectorstore = None
_client = None

EMBEDDING_DIMENSIONS = {
    "gemini-embedding-001": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}



def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384
    if "gemini-embedding-001" in normalized:
            return 1536

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Qdrant. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )



def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.gemini_api_key,
            output_dimensionality=1536
        )
    return _embeddings




def get_client() -> QdrantClient:
    global _client
    if _client is None:
        kwargs = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client



def ensure_collection():
    client = get_client()
    desired_dimension = get_embedding_dimension()

    if not client.collection_exists(settings.qdrant_collection_name):
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=desired_dimension,
                distance=Distance.COSINE,
            ),
        )
    return client




def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        client = ensure_collection()
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection_name,
            embedding=get_embeddings(),
        )
    return _vectorstore



def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})



def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
