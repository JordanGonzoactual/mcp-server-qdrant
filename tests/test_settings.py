import pytest

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
        QdrantSettings()

    def test_minimal_config(self, monkeypatch):
        """Test loading minimal configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")

        settings = QdrantSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    def test_full_config(self, monkeypatch):
        """Test loading full configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        monkeypatch.setenv("COLLECTION_NAME", "my_memories")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "15")
        monkeypatch.setenv("QDRANT_READ_ONLY", "1")

        settings = QdrantSettings()
        assert settings.location == "http://qdrant.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.search_limit == 15
        assert settings.read_only is True

    def test_local_path_config(self, monkeypatch):
        """Test loading local path configuration from environment variables."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        settings = QdrantSettings()
        assert settings.local_path == "/path/to/local/qdrant"

    def test_local_path_is_exclusive_with_url(self, monkeypatch):
        """Test that local path cannot be set if Qdrant URL is provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        with pytest.raises(ValueError):
            QdrantSettings()

        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        with pytest.raises(ValueError):
            QdrantSettings()


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_values(self, monkeypatch):
        """Test loading custom values from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom_model")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "custom_model"


class TestQdrantCollectionsSetting:
    def test_qdrant_collections_none_when_unset(self, monkeypatch):
        """When QDRANT_COLLECTIONS is not set, field should be None."""
        monkeypatch.delenv("QDRANT_COLLECTIONS", raising=False)
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        settings = QdrantSettings()
        assert settings.qdrant_collections is None

    def test_qdrant_collections_single_entry(self, monkeypatch):
        """Parse a single name:collection pair."""
        monkeypatch.setenv("QDRANT_COLLECTIONS", "research:KnowledgeMap")
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        settings = QdrantSettings()
        assert settings.qdrant_collections == {"research": "KnowledgeMap"}

    def test_qdrant_collections_multiple_entries(self, monkeypatch):
        """Parse multiple comma-separated name:collection pairs."""
        monkeypatch.setenv(
            "QDRANT_COLLECTIONS",
            "research:KnowledgeMap,memory:mem-claude-code-system",
        )
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        settings = QdrantSettings()
        assert settings.qdrant_collections == {
            "research": "KnowledgeMap",
            "memory": "mem-claude-code-system",
        }

    def test_qdrant_collections_strips_whitespace(self, monkeypatch):
        """Whitespace around names and colons should be stripped."""
        monkeypatch.setenv(
            "QDRANT_COLLECTIONS",
            " research : KnowledgeMap , memory : mem-claude-code-system ",
        )
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        settings = QdrantSettings()
        assert settings.qdrant_collections == {
            "research": "KnowledgeMap",
            "memory": "mem-claude-code-system",
        }

    def test_qdrant_collections_conflict_with_collection_name(self, monkeypatch):
        """Setting both QDRANT_COLLECTIONS and COLLECTION_NAME should raise ValueError."""
        monkeypatch.setenv("QDRANT_COLLECTIONS", "research:KnowledgeMap")
        monkeypatch.setenv("COLLECTION_NAME", "some_collection")
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        with pytest.raises(ValueError, match="Cannot set both"):
            QdrantSettings()

    def test_legacy_collection_name_still_works(self, monkeypatch):
        """COLLECTION_NAME env var must still work when QDRANT_COLLECTIONS is not set."""
        monkeypatch.setenv("COLLECTION_NAME", "legacy_collection")
        monkeypatch.delenv("QDRANT_COLLECTIONS", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        settings = QdrantSettings()
        assert settings.collection_name == "legacy_collection"
        assert settings.qdrant_collections is None

    def test_both_none_when_neither_set(self, monkeypatch):
        """When neither env var is set, both fields should be None."""
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        monkeypatch.delenv("QDRANT_COLLECTIONS", raising=False)
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        settings = QdrantSettings()
        assert settings.collection_name is None
        assert settings.qdrant_collections is None


class TestToolSettings:
    def test_default_values(self):
        """Test that default values are set correctly when no env vars are provided."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_store_description(self, monkeypatch):
        """Test loading custom store description from environment variable."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_find_description(self, monkeypatch):
        """Test loading custom find description from environment variable."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == "Custom find description"

    def test_all_custom_values(self, monkeypatch):
        """Test loading all custom values from environment variables."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"

    def test_get_find_description_returns_default_when_no_override(self, monkeypatch):
        """get_find_description falls back to tool_find_description when no override env var is set."""
        monkeypatch.delenv("TOOL_FIND_DESCRIPTION_MEMORY", raising=False)
        settings = ToolSettings()
        assert settings.get_find_description("memory") == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_get_find_description_returns_override_when_env_var_set(self, monkeypatch):
        """get_find_description returns override value when TOOL_FIND_DESCRIPTION_MEMORY is set."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION_MEMORY", "Find memories in the memory collection")
        settings = ToolSettings()
        assert settings.get_find_description("memory") == "Find memories in the memory collection"

    def test_get_find_description_label_uppercased_for_env_lookup(self, monkeypatch):
        """get_find_description uses uppercase label when looking up the env var."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION_RESEARCH", "Search the research collection")
        settings = ToolSettings()
        assert settings.get_find_description("research") == "Search the research collection"

    def test_get_store_description_returns_default_when_no_override(self, monkeypatch):
        """get_store_description falls back to tool_store_description when no override env var is set."""
        monkeypatch.delenv("TOOL_STORE_DESCRIPTION_MEMORY", raising=False)
        settings = ToolSettings()
        assert settings.get_store_description("memory") == DEFAULT_TOOL_STORE_DESCRIPTION

    def test_get_store_description_returns_override_when_env_var_set(self, monkeypatch):
        """get_store_description returns override when TOOL_STORE_DESCRIPTION_RESEARCH is set."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION_RESEARCH", "Store documents in the research collection")
        settings = ToolSettings()
        assert settings.get_store_description("research") == "Store documents in the research collection"

    def test_get_find_description_does_not_affect_other_labels(self, monkeypatch):
        """Override for one label does not affect another label's find description."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION_MEMORY", "Custom memory find")
        monkeypatch.delenv("TOOL_FIND_DESCRIPTION_RESEARCH", raising=False)
        settings = ToolSettings()
        assert settings.get_find_description("memory") == "Custom memory find"
        assert settings.get_find_description("research") == DEFAULT_TOOL_FIND_DESCRIPTION
