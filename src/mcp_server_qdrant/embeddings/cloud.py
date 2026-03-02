from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# Known cloud inference model dimensions.
# Full list available in Qdrant Cloud Console > Cluster Detail > Inference tab.
CLOUD_MODEL_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-minilm-l6-v2": 384,
    "mxbai/embed-large-v1": 1024,
    "qdrant/clip-vit-b-32-vision": 512,
    "qdrant/clip-vit-b-32-text": 512,
}


class CloudInferenceProvider(EmbeddingProvider):
    """
    Cloud inference provider — embeddings are generated server-side by Qdrant.
    This provider holds model config (name, vector size) but never embeds locally.
    The QdrantConnector uses models.Document() objects instead when this provider is active.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        normalized = model_name.lower()
        if normalized not in CLOUD_MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown cloud inference model: {model_name}. "
                f"Known models: {list(CLOUD_MODEL_DIMENSIONS.keys())}. "
                f"You may also check the Qdrant Cloud Console for the full list."
            )
        self._vector_size = CLOUD_MODEL_DIMENSIONS[normalized]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "CloudInferenceProvider does not embed locally. "
            "Use models.Document() objects for server-side inference."
        )

    async def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError(
            "CloudInferenceProvider does not embed locally. "
            "Use models.Document() objects for server-side inference."
        )

    def get_vector_name(self) -> str:
        return "dense"

    def get_vector_size(self) -> int:
        return self._vector_size
