import os

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# Native models hosted directly by Qdrant Cloud Inference (no external key).
# Full list available in Qdrant Cloud Console > Cluster Detail > Inference tab.
CLOUD_MODEL_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-minilm-l6-v2": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "qdrant/clip-vit-b-32-vision": 512,
    "qdrant/clip-vit-b-32-text": 512,
}

# External providers Qdrant Cloud proxies (model name prefixed with the slug).
# Qdrant never stores the provider key — it must travel in each Document's
# `options` on every request. `dim_param` is the provider-specific name for the
# output-dimension knob; `input_type` flags whether the provider accepts an
# asymmetric search_document/search_query hint in options. Verified live against
# the cluster for cohere/embed-v4.0.
EXTERNAL_PROVIDERS: dict[str, dict] = {
    "openai/": {"key_env": "OPENAI_API_KEY", "dim_param": "dimensions", "input_type": False},
    "cohere/": {"key_env": "COHERE_API_KEY", "dim_param": "output_dimension", "input_type": True},
    "jinaai/": {"key_env": "JINA_API_KEY", "dim_param": "dimensions", "input_type": False},
    "openrouter/": {"key_env": "OPENROUTER_API_KEY", "dim_param": None, "input_type": False},
}

# Default output dimensionality for known external models, used when no explicit
# EMBEDDING_OUTPUT_DIMENSION is configured.
EXTERNAL_MODEL_DEFAULT_DIMENSIONS: dict[str, int] = {
    "cohere/embed-v4.0": 1536,
    "openai/text-embedding-3-large": 3072,
    "jinaai/jina-embeddings-v3": 1024,
}


class CloudInferenceProvider(EmbeddingProvider):
    """
    Cloud inference provider — embeddings are generated server-side by Qdrant
    Cloud. This provider holds model config (name, vector size, per-request
    options) but never embeds locally; QdrantConnector sends models.Document()
    objects for server-side inference.

    Two flavours, distinguished by the model name:
      * native Qdrant-hosted models (e.g. mxbai-embed-large-v1) — no key, no options
      * external-provider models (cohere/…, openai/…, jinaai/…, openrouter/…) —
        Qdrant proxies the provider's API, so the provider key plus
        provider-specific params (output dimension, input_type) travel in
        Document.options on every request.
    """

    def __init__(self, model_name: str, output_dimension: int | None = None):
        self.model_name = model_name
        self._output_dimension = output_dimension
        normalized = model_name.lower()

        prefix = next(
            (p for p in EXTERNAL_PROVIDERS if normalized.startswith(p)),
            None,
        )
        self._external = EXTERNAL_PROVIDERS[prefix] if prefix else None

        if self._external is not None:
            size = output_dimension or EXTERNAL_MODEL_DEFAULT_DIMENSIONS.get(normalized)
            if size is None:
                raise ValueError(
                    f"External model {model_name!r} has no known default dimension; "
                    f"set EMBEDDING_OUTPUT_DIMENSION explicitly."
                )
            self._vector_size = size
        else:
            if normalized not in CLOUD_MODEL_DIMENSIONS:
                raise ValueError(
                    f"Unknown cloud inference model: {model_name}. "
                    f"Known native models: {list(CLOUD_MODEL_DIMENSIONS)}; "
                    f"known external prefixes: {list(EXTERNAL_PROVIDERS)}. "
                    f"You may also check the Qdrant Cloud Console for the full list."
                )
            self._vector_size = CLOUD_MODEL_DIMENSIONS[normalized]

    def _api_key(self) -> str:
        assert self._external is not None
        env_name = self._external["key_env"]
        key = os.environ.get(env_name)
        if not key:
            raise ValueError(
                f"Model {self.model_name!r} is proxied through an external provider "
                f"but {env_name} is not set in the environment."
            )
        return key

    def document_options(self, input_type: str | None) -> dict | None:
        """Per-request options for a models.Document, or None for native models.

        External providers need their API key on every request; providers that
        support it also take an output-dimension knob and an input_type
        (search_document for stored chunks, search_query for lookups) for
        asymmetric retrieval.
        """
        if self._external is None:
            return None
        options: dict = {"api_key": self._api_key()}
        dim_param = self._external["dim_param"]
        if dim_param and self._output_dimension is not None:
            options[dim_param] = self._output_dimension
        if self._external["input_type"] and input_type is not None:
            options["input_type"] = input_type
        return options

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
