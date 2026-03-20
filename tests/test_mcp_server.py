from unittest.mock import MagicMock

import pytest

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import QdrantSettings, ToolSettings



ENV_VARS_TO_CLEAN = [
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "COLLECTION_NAME",
    "QDRANT_COLLECTIONS",
    "QDRANT_LOCAL_PATH",
    "QDRANT_READ_ONLY",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "SPARSE_MODEL",
    "TOOL_STORE_DESCRIPTION",
    "TOOL_FIND_DESCRIPTION",
    "QDRANT_CHANNEL_ENABLED",
]


@pytest.fixture
def clean_env(monkeypatch):
    for var in ENV_VARS_TO_CLEAN:
        monkeypatch.delenv(var, raising=False)


def make_mock_embedding_provider():
    provider = MagicMock()
    provider.model_name = "test-model"
    provider.get_vector_name.return_value = "dense"
    provider.get_vector_size.return_value = 1024
    return provider


def clean_env(monkeypatch):
    for var in ENV_VARS_TO_CLEAN:
        monkeypatch.delenv(var, raising=False)


def get_tool_names(server: QdrantMCPServer) -> list[str]:
    return [t.name for t in server._tool_manager._tools.values()]


class TestSetupToolsLegacySingleCollection:
    def test_legacy_single_collection_registers_find_and_store(self, monkeypatch):
        """When COLLECTION_NAME is set, registers qdrant-find and qdrant-store (3 tools)."""
        clean_env(monkeypatch)
        monkeypatch.setenv("COLLECTION_NAME", "KnowledgeMap")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-find" in tool_names
        assert "qdrant-store" in tool_names
        assert len(tool_names) == 3

    def test_legacy_registers_update_tool(self, monkeypatch):
        """COLLECTION_NAME -> qdrant-update registered."""
        clean_env(monkeypatch)
        monkeypatch.setenv("COLLECTION_NAME", "KnowledgeMap")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-update" in tool_names


class TestSetupToolsMultiCollection:
    def test_multi_collection_registers_labeled_tools(self, monkeypatch):
        """When QDRANT_COLLECTIONS is set, registers qdrant-find-{label} and qdrant-store-{label} per entry."""
        clean_env(monkeypatch)
        monkeypatch.setenv(
            "QDRANT_COLLECTIONS", "research:KnowledgeMap,memory:mem-claude-code-system"
        )

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-find-research" in tool_names
        assert "qdrant-store-research" in tool_names
        assert "qdrant-find-memory" in tool_names
        assert "qdrant-store-memory" in tool_names
        assert len(tool_names) == 6

    def test_multi_collection_does_not_register_generic_tools(self, monkeypatch):
        """When QDRANT_COLLECTIONS is set, generic qdrant-find and qdrant-store are NOT registered."""
        clean_env(monkeypatch)
        monkeypatch.setenv("QDRANT_COLLECTIONS", "research:KnowledgeMap")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-find" not in tool_names
        assert "qdrant-store" not in tool_names


class TestUpdateMetadataTool:
    def test_multi_collection_registers_update_tools(self, monkeypatch):
        """QDRANT_COLLECTIONS -> qdrant-update-{name} tools registered."""
        clean_env(monkeypatch)
        monkeypatch.setenv(
            "QDRANT_COLLECTIONS",
            "research:KnowledgeMap,memory:mem-claude-code-system",
        )

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-update-research" in tool_names
        assert "qdrant-update-memory" in tool_names

    def test_legacy_registers_update_tool(self, monkeypatch):
        """COLLECTION_NAME -> qdrant-update registered."""
        clean_env(monkeypatch)
        monkeypatch.setenv("COLLECTION_NAME", "KnowledgeMap")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-update" in tool_names

    def test_read_only_skips_update_tool(self, monkeypatch):
        """read_only -> no update tools."""
        clean_env(monkeypatch)
        monkeypatch.setenv("QDRANT_COLLECTIONS", "research:KnowledgeMap")
        monkeypatch.setenv("QDRANT_READ_ONLY", "true")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-update-research" not in tool_names


class TestSetupToolsMultiCollectionReadOnly:
    def test_multi_collection_read_only_skips_store_tools(self, monkeypatch):
        """When QDRANT_COLLECTIONS + QDRANT_READ_ONLY=true, only qdrant-find-{label} tools are registered."""
        clean_env(monkeypatch)
        monkeypatch.setenv("QDRANT_COLLECTIONS", "research:KnowledgeMap")
        monkeypatch.setenv("QDRANT_READ_ONLY", "true")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-find-research" in tool_names
        assert "qdrant-store-research" not in tool_names
        assert "qdrant-update-research" not in tool_names
        assert len(tool_names) == 1


class TestSetupToolsNoConfig:
    def test_no_config_registers_generic_find_and_store(self, monkeypatch):
        """When neither COLLECTION_NAME nor QDRANT_COLLECTIONS is set, registers generic qdrant-find and qdrant-store."""
        clean_env(monkeypatch)

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-find" in tool_names
        assert "qdrant-store" in tool_names
        assert len(tool_names) == 3


def create_server() -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider=make_mock_embedding_provider(),
    )


class TestChannelCapability:
    def test_channel_capability_declared_when_enabled(self, clean_env, monkeypatch):
        monkeypatch.setenv("QDRANT_CHANNEL_ENABLED", "true")
        monkeypatch.setenv("COLLECTION_NAME", "test")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
        monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        server = create_server()
        assert server._qdrant_settings.channel_enabled is True

    def test_channel_settings_stored(self, clean_env, monkeypatch):
        monkeypatch.setenv("COLLECTION_NAME", "test")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
        monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        server = create_server()
        assert server._qdrant_settings.channel_enabled is False
