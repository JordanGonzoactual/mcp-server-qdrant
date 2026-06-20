from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import CollectionScopeError, QdrantMCPServer
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


class _StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding provider so the in-memory connector can
    round-trip without downloading a real model."""

    def __init__(self, size: int = 8):
        self._size = size

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._size
        for i, ch in enumerate(text):
            vec[i % self._size] += (ord(ch) % 17) / 17.0
        return vec

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self._vector(doc) for doc in documents]

    async def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    def get_vector_name(self) -> str:
        return "dense"

    def get_vector_size(self) -> int:
        return self._size


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.debug = AsyncMock()
    return ctx


class TestPerConnectionCollectionScope:
    """CCS-73: a shared HTTP server scopes each connection to the collection
    set carried in the X-Qdrant-Collections header, denying access to any
    collection outside that set while keeping the env path functional."""

    def _build_server(self, monkeypatch) -> QdrantMCPServer:
        # Two labeled collections: scope A -> ColA, scope B -> ColB.
        monkeypatch.setenv("QDRANT_COLLECTIONS", "a:ColA,b:ColB")
        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=_StubEmbeddingProvider(),
        )
        # Point the connector at a local in-memory Qdrant so permitted paths
        # round-trip without any external endpoint.
        from qdrant_client import AsyncQdrantClient

        server.qdrant_connector._client = AsyncQdrantClient(location=":memory:")
        return server

    def _tool_fn(self, server: QdrantMCPServer, name: str):
        return server._tool_manager._tools[name].fn

    @pytest.mark.asyncio
    async def test_cross_scope_read_denied_own_scope_permitted(
        self, clean_env, monkeypatch
    ):
        server = self._build_server(monkeypatch)
        find_b = self._tool_fn(server, "qdrant-find-b")  # bound to ColB

        # Connection scoped to set A (ColA) must NOT reach ColB.
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "ColA"},
        ):
            with pytest.raises(CollectionScopeError):
                await find_b(ctx=_make_ctx(), query="anything")

        # Same connection IS permitted to read its own collection (ColA).
        find_a = self._tool_fn(server, "qdrant-find-a")
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "ColA"},
        ):
            result = await find_a(ctx=_make_ctx(), query="anything")
        # Empty in-memory collection -> None, but the scope gate let it through.
        assert result is None

    @pytest.mark.asyncio
    async def test_cross_scope_write_denied_own_scope_permitted(
        self, clean_env, monkeypatch
    ):
        server = self._build_server(monkeypatch)
        store_b = self._tool_fn(server, "qdrant-store-b")  # bound to ColB

        # Connection scoped to set A cannot write into ColB.
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "ColA"},
        ):
            with pytest.raises(CollectionScopeError):
                await store_b(ctx=_make_ctx(), information="secret")

        # Same connection CAN write + read back its own collection (ColA).
        store_a = self._tool_fn(server, "qdrant-store-a")
        find_a = self._tool_fn(server, "qdrant-find-a")
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "ColA"},
        ):
            await store_a(ctx=_make_ctx(), information="hello world")
            result = await find_a(ctx=_make_ctx(), query="hello world")
        assert result is not None
        assert any("hello world" in line for line in result)

    @pytest.mark.asyncio
    async def test_header_overrides_env_scope_for_that_request(
        self, clean_env, monkeypatch
    ):
        """Header naming a collection NOT in the env set still scopes the
        request to exactly that header set (per-connection override)."""
        server = self._build_server(monkeypatch)
        find_a = self._tool_fn(server, "qdrant-find-a")  # bound to ColA

        # Header scopes to ColB only; ColA must be denied for this request.
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "ColB"},
        ):
            with pytest.raises(CollectionScopeError):
                await find_a(ctx=_make_ctx(), query="anything")

    @pytest.mark.asyncio
    async def test_absent_header_falls_back_to_env_scope(
        self, clean_env, monkeypatch
    ):
        """Backward compat (stdio): no HTTP header -> env-derived scope applies,
        and a collection in the env set is permitted."""
        server = self._build_server(monkeypatch)
        find_a = self._tool_fn(server, "qdrant-find-a")  # ColA is in env scope

        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={},
        ):
            # Does not raise: ColA is within the env-derived scope {ColA, ColB}.
            result = await find_a(ctx=_make_ctx(), query="anything")
        assert result is None


class TestPerConnectionCollectionRouting:
    """A shared HTTP server routes a labeled tool to a DIFFERENT collection per
    connection, resolved at request time from the X-Qdrant-Collections header's
    ``label:collection`` mapping — so one process serves many repos without one
    process per repo."""

    def _build_server(self, monkeypatch) -> QdrantMCPServer:
        monkeypatch.setenv("QDRANT_COLLECTIONS", "a:ColA,b:ColB")
        server = QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=_StubEmbeddingProvider(),
        )
        from qdrant_client import AsyncQdrantClient

        server.qdrant_connector._client = AsyncQdrantClient(location=":memory:")
        return server

    def _tool_fn(self, server: QdrantMCPServer, name: str):
        return server._tool_manager._tools[name].fn

    @pytest.mark.asyncio
    async def test_label_routes_to_header_named_collection(
        self, clean_env, monkeypatch
    ):
        """Header ``a:ColX`` makes the ``a`` tool target ColX (not the env
        default ColA); a connection routing ``a:ColA`` sees nothing, proving
        the write landed in the per-connection collection."""
        server = self._build_server(monkeypatch)
        store_a = self._tool_fn(server, "qdrant-store-a")
        find_a = self._tool_fn(server, "qdrant-find-a")

        # Connection routes label 'a' -> ColX (a collection outside the env map).
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "a:ColX"},
        ):
            await store_a(ctx=_make_ctx(), information="routed payload")
            routed = await find_a(ctx=_make_ctx(), query="routed payload")
        assert routed is not None
        assert any("routed payload" in line for line in routed)

        # A different connection routing 'a' -> the env default ColA sees
        # nothing: the write went to ColX, so routing is per-connection.
        with patch(
            "mcp_server_qdrant.mcp_server.get_http_headers",
            return_value={"x-qdrant-collections": "a:ColA"},
        ):
            default = await find_a(ctx=_make_ctx(), query="routed payload")
        assert default is None


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
