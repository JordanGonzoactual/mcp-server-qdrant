from unittest.mock import AsyncMock, MagicMock, patch

import anyio
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


def get_tool_names(server: QdrantMCPServer) -> list[str]:
    return [t.name for t in server._tool_manager._tools.values()]


def create_server() -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider=make_mock_embedding_provider(),
    )


class TestSetupToolsLegacySingleCollection:
    def test_legacy_single_collection_registers_find_and_store(self, clean_env, monkeypatch):
        """When COLLECTION_NAME is set, registers qdrant-find and qdrant-store (3 tools)."""
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

    def test_legacy_registers_update_tool(self, clean_env, monkeypatch):
        """COLLECTION_NAME -> qdrant-update registered."""
        monkeypatch.setenv("COLLECTION_NAME", "KnowledgeMap")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-update" in tool_names


class TestSetupToolsMultiCollection:
    def test_multi_collection_registers_labeled_tools(self, clean_env, monkeypatch):
        """When QDRANT_COLLECTIONS is set, registers qdrant-find-{label} and qdrant-store-{label} per entry."""
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

    def test_multi_collection_does_not_register_generic_tools(self, clean_env, monkeypatch):
        """When QDRANT_COLLECTIONS is set, generic qdrant-find and qdrant-store are NOT registered."""
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
    def test_multi_collection_registers_update_tools(self, clean_env, monkeypatch):
        """QDRANT_COLLECTIONS -> qdrant-update-{name} tools registered."""
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

    def test_legacy_registers_update_tool(self, clean_env, monkeypatch):
        """COLLECTION_NAME -> qdrant-update registered."""
        monkeypatch.setenv("COLLECTION_NAME", "KnowledgeMap")

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-update" in tool_names

    def test_read_only_skips_update_tool(self, clean_env, monkeypatch):
        """read_only -> no update tools."""
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
    def test_multi_collection_read_only_skips_store_tools(self, clean_env, monkeypatch):
        """When QDRANT_COLLECTIONS + QDRANT_READ_ONLY=true, only qdrant-find-{label} tools are registered."""
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
    def test_no_config_registers_generic_find_and_store(self, clean_env):
        """When neither COLLECTION_NAME nor QDRANT_COLLECTIONS is set, registers generic qdrant-find and qdrant-store."""
        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        tool_names = get_tool_names(server)
        assert "qdrant-find" in tool_names
        assert "qdrant-store" in tool_names
        assert len(tool_names) == 3


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


class TestRunStdioAsyncChannelIntegration:
    """Tests that run_stdio_async wires ChannelOrchestrator when channel is enabled."""

    @pytest.mark.asyncio
    async def test_journal_watcher_started_when_channel_enabled(
        self, clean_env, monkeypatch, tmp_path
    ):
        """When channel_enabled and channel_journal_path are set, check_journal is called."""
        journal_file = tmp_path / "test_journal.jsonl"
        journal_file.write_text("")

        monkeypatch.setenv("QDRANT_CHANNEL_ENABLED", "true")
        monkeypatch.setenv(
            "QDRANT_CHANNEL_JOURNAL_PATH", str(journal_file)
        )
        monkeypatch.setenv(
            "QDRANT_COLLECTIONS", "memory:mem-collection"
        )

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        check_journal_calls = []

        async def fake_check_journal():
            check_journal_calls.append(1)

        mock_orchestrator = MagicMock()
        mock_orchestrator.check_journal = fake_check_journal

        fake_read_stream = MagicMock()
        fake_write_stream = MagicMock()

        async def fake_mcp_run(rs, ws, opts, **kwargs):
            # Simulate a brief server run then exit
            await anyio.sleep(0.02)

        with patch(
            "mcp_server_qdrant.mcp_server.ChannelOrchestrator",
            return_value=mock_orchestrator,
        ) as MockOrch:
            with patch.object(server._mcp_server, "run", side_effect=fake_mcp_run):
                with patch(
                    "mcp_server_qdrant.mcp_server.stdio_server"
                ) as mock_stdio:
                    mock_stdio.return_value.__aenter__ = AsyncMock(
                        return_value=(fake_read_stream, fake_write_stream)
                    )
                    mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
                    await server.run_stdio_async()

        assert MockOrch.called
        call_kwargs = MockOrch.call_args
        assert call_kwargs.kwargs["connector"] is server.qdrant_connector
        assert call_kwargs.kwargs["collections"] == {"memory": "mem-collection"}
        assert call_kwargs.kwargs["journal_path"] == str(journal_file)

    @pytest.mark.asyncio
    async def test_journal_watcher_not_started_when_channel_disabled(
        self, clean_env, monkeypatch
    ):
        """When channel_enabled is False, ChannelOrchestrator is not instantiated."""
        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        async def fake_mcp_run(rs, ws, opts, **kwargs):
            await anyio.sleep(0)

        with patch(
            "mcp_server_qdrant.mcp_server.ChannelOrchestrator"
        ) as MockOrch:
            with patch.object(server._mcp_server, "run", side_effect=fake_mcp_run):
                with patch(
                    "mcp_server_qdrant.mcp_server.stdio_server"
                ) as mock_stdio:
                    mock_stdio.return_value.__aenter__ = AsyncMock(
                        return_value=(MagicMock(), MagicMock())
                    )
                    mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
                    await server.run_stdio_async()

        MockOrch.assert_not_called()

    @pytest.mark.asyncio
    async def test_journal_watcher_not_started_when_no_journal_path(
        self, clean_env, monkeypatch
    ):
        """When channel_enabled but no journal_path, ChannelOrchestrator is not instantiated."""
        monkeypatch.setenv("QDRANT_CHANNEL_ENABLED", "true")
        monkeypatch.setenv("QDRANT_COLLECTIONS", "memory:mem-collection")
        # No QDRANT_CHANNEL_JOURNAL_PATH set

        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=make_mock_embedding_provider(),
        )

        async def fake_mcp_run(rs, ws, opts, **kwargs):
            await anyio.sleep(0)

        with patch(
            "mcp_server_qdrant.mcp_server.ChannelOrchestrator"
        ) as MockOrch:
            with patch.object(server._mcp_server, "run", side_effect=fake_mcp_run):
                with patch(
                    "mcp_server_qdrant.mcp_server.stdio_server"
                ) as mock_stdio:
                    mock_stdio.return_value.__aenter__ = AsyncMock(
                        return_value=(MagicMock(), MagicMock())
                    )
                    mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
                    await server.run_stdio_async()

        MockOrch.assert_not_called()
