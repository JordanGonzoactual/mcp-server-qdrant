import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_qdrant.embeddings.cloud import CloudInferenceProvider, CLOUD_MODEL_DIMENSIONS
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.qdrant import Entry, QdrantConnector, SPARSE_VECTOR_NAME
from mcp_server_qdrant.settings import EmbeddingProviderSettings


class TestCloudInferenceProvider:
    def test_initialization_valid_model(self):
        provider = CloudInferenceProvider("sentence-transformers/all-minilm-l6-v2")
        assert provider.model_name == "sentence-transformers/all-minilm-l6-v2"
        assert provider.get_vector_size() == 384

    def test_initialization_large_model(self):
        provider = CloudInferenceProvider("mxbai/embed-large-v1")
        assert provider.get_vector_size() == 1024

    def test_initialization_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown cloud inference model"):
            CloudInferenceProvider("unknown/model")

    def test_get_vector_name(self):
        provider = CloudInferenceProvider("mxbai/embed-large-v1")
        assert provider.get_vector_name() == "dense"

    @pytest.mark.asyncio
    async def test_embed_documents_raises(self):
        provider = CloudInferenceProvider("mxbai/embed-large-v1")
        with pytest.raises(NotImplementedError):
            await provider.embed_documents(["test"])

    @pytest.mark.asyncio
    async def test_embed_query_raises(self):
        provider = CloudInferenceProvider("mxbai/embed-large-v1")
        with pytest.raises(NotImplementedError):
            await provider.embed_query("test")

    def test_all_known_models(self):
        for model, dim in CLOUD_MODEL_DIMENSIONS.items():
            provider = CloudInferenceProvider(model)
            assert provider.get_vector_size() == dim


class TestCloudInferenceFactory:
    def test_create_cloud_provider(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "cloud")
        monkeypatch.setenv("EMBEDDING_MODEL", "mxbai/embed-large-v1")
        settings = EmbeddingProviderSettings()
        provider = create_embedding_provider(settings)
        assert isinstance(provider, CloudInferenceProvider)
        assert provider.get_vector_size() == 1024


class TestCloudInferenceSettings:
    def test_cloud_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "cloud")
        monkeypatch.setenv("EMBEDDING_MODEL", "mxbai/embed-large-v1")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.CLOUD
        assert settings.model_name == "mxbai/embed-large-v1"

    def test_sparse_model_default(self):
        settings = EmbeddingProviderSettings()
        assert settings.sparse_model is None

    def test_sparse_model_from_env(self, monkeypatch):
        monkeypatch.setenv("SPARSE_MODEL", "qdrant/bm25")
        settings = EmbeddingProviderSettings()
        assert settings.sparse_model == "qdrant/bm25"


class TestQdrantConnectorCloudInference:
    """Test QdrantConnector cloud inference code paths using mocks."""

    @pytest.fixture
    def cloud_provider(self):
        return CloudInferenceProvider("mxbai/embed-large-v1")

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)
        client.create_collection = AsyncMock()
        client.create_payload_index = AsyncMock()
        client.upsert = AsyncMock()
        client.query_points = AsyncMock()
        return client

    @pytest.fixture
    def connector_dense_only(self, cloud_provider, mock_client):
        """Cloud connector with dense only (no sparse)."""
        with patch("mcp_server_qdrant.qdrant.AsyncQdrantClient", return_value=mock_client):
            connector = QdrantConnector(
                qdrant_url="https://fake.cloud.qdrant.io",
                qdrant_api_key="fake-key",
                collection_name="test_cloud",
                embedding_provider=cloud_provider,
                cloud_inference=True,
            )
        return connector

    @pytest.fixture
    def connector_hybrid(self, cloud_provider, mock_client):
        """Cloud connector with dense + sparse hybrid search."""
        with patch("mcp_server_qdrant.qdrant.AsyncQdrantClient", return_value=mock_client):
            connector = QdrantConnector(
                qdrant_url="https://fake.cloud.qdrant.io",
                qdrant_api_key="fake-key",
                collection_name="test_hybrid",
                embedding_provider=cloud_provider,
                cloud_inference=True,
                sparse_model="qdrant/bm25",
            )
        return connector

    @pytest.mark.asyncio
    async def test_ensure_collection_dense_only(self, connector_dense_only, mock_client):
        """Collection creation without sparse vectors."""
        await connector_dense_only._ensure_collection_exists("test_cloud")

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert "dense" in call_kwargs["vectors_config"]
        assert call_kwargs["vectors_config"]["dense"].size == 1024
        assert call_kwargs.get("sparse_vectors_config") is None

    @pytest.mark.asyncio
    async def test_ensure_collection_hybrid(self, connector_hybrid, mock_client):
        """Collection creation with both dense and sparse vectors."""
        await connector_hybrid._ensure_collection_exists("test_hybrid")

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]

        # Dense vector config
        assert "dense" in call_kwargs["vectors_config"]
        assert call_kwargs["vectors_config"]["dense"].size == 1024

        # Sparse vector config
        sparse_config = call_kwargs["sparse_vectors_config"]
        assert sparse_config is not None
        assert SPARSE_VECTOR_NAME in sparse_config

    @pytest.mark.asyncio
    async def test_store_cloud_dense_only(self, connector_dense_only, mock_client):
        """Store uses Document objects for cloud inference (dense only)."""
        mock_client.collection_exists = AsyncMock(return_value=True)

        from qdrant_client import models

        entry = Entry(content="test content", metadata={"key": "val"})
        await connector_dense_only.store(entry)

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        point = call_kwargs["points"][0]

        # Should use Document object for dense vector
        assert isinstance(point.vector["dense"], models.Document)
        assert point.vector["dense"].text == "test content"
        assert point.vector["dense"].model == "mxbai/embed-large-v1"

        # No sparse vector
        assert SPARSE_VECTOR_NAME not in point.vector

    @pytest.mark.asyncio
    async def test_store_cloud_hybrid(self, connector_hybrid, mock_client):
        """Store uses Document objects for both dense and sparse vectors."""
        mock_client.collection_exists = AsyncMock(return_value=True)

        from qdrant_client import models

        entry = Entry(content="test content")
        await connector_hybrid.store(entry)

        mock_client.upsert.assert_called_once()
        point = mock_client.upsert.call_args[1]["points"][0]

        # Dense vector
        assert isinstance(point.vector["dense"], models.Document)
        assert point.vector["dense"].model == "mxbai/embed-large-v1"

        # Sparse vector
        assert isinstance(point.vector[SPARSE_VECTOR_NAME], models.Document)
        assert point.vector[SPARSE_VECTOR_NAME].model == "qdrant/bm25"

    @pytest.mark.asyncio
    async def test_search_cloud_dense_only(self, connector_dense_only, mock_client):
        """Search uses Document query for cloud inference (dense only)."""
        mock_client.collection_exists = AsyncMock(return_value=True)

        # Mock search results
        mock_point = MagicMock()
        mock_point.payload = {"document": "found content", "metadata": {"k": "v"}}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points = AsyncMock(return_value=mock_result)

        from qdrant_client import models

        results = await connector_dense_only.search("test query")

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args[1]

        # Dense-only: no prefetch, direct Document query
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None
        assert isinstance(call_kwargs["query"], models.Document)
        assert call_kwargs["using"] == "dense"

        assert len(results) == 1
        assert results[0].content == "found content"

    @pytest.mark.asyncio
    async def test_search_cloud_hybrid(self, connector_hybrid, mock_client):
        """Search uses prefetch + RRF fusion for hybrid search."""
        mock_client.collection_exists = AsyncMock(return_value=True)

        mock_point = MagicMock()
        mock_point.payload = {"document": "hybrid result", "metadata": None}
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        mock_client.query_points = AsyncMock(return_value=mock_result)

        from qdrant_client import models

        results = await connector_hybrid.search("test query", limit=5)

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args[1]

        # Should have prefetch with both dense and sparse
        assert "prefetch" in call_kwargs
        prefetch = call_kwargs["prefetch"]
        assert len(prefetch) == 2

        # Dense prefetch
        assert isinstance(prefetch[0].query, models.Document)
        assert prefetch[0].using == "dense"

        # Sparse prefetch
        assert isinstance(prefetch[1].query, models.Document)
        assert prefetch[1].using == SPARSE_VECTOR_NAME

        # RRF fusion
        assert isinstance(call_kwargs["query"], models.FusionQuery)

        assert len(results) == 1
        assert results[0].content == "hybrid result"

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection(self, connector_hybrid, mock_client):
        """Search returns empty list for nonexistent collection."""
        mock_client.collection_exists = AsyncMock(return_value=False)
        results = await connector_hybrid.search("test")
        assert results == []
