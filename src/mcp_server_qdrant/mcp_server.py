import json
import logging
from typing import Annotated, Any, Optional

import anyio
from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_http_headers
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.stdio import stdio_server
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.channel import ChannelOrchestrator
from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import (
    make_partial_function,
    make_routed_function,
)
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.reranker import VoyageReranker
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    RerankSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)

# Per-connection collection scoping (CCS-73).
# A shared HTTP server reads each connection's active collections from this
# header instead of relying solely on the process-level QDRANT_COLLECTIONS env
# var. The grammar mirrors that env var: comma-separated `label:collection`
# pairs (e.g. `research:KnowledgeMap,memory:mem-foo`). The header both ROUTES
# each labeled tool to that connection's collection AND scopes enforcement to
# the collections named. When the header is absent the server falls back to the
# env-derived map/scope, preserving stdio backward compat.
COLLECTION_SCOPE_HEADER = "x-qdrant-collections"


class CollectionScopeError(ValueError):
    """Raised when a request targets a collection outside its active scope."""


def parse_collection_map(raw: str | None) -> dict[str, str] | None:
    """Parse the per-connection collection header into a {label: collection} map.

    Grammar mirrors the env ``QDRANT_COLLECTIONS`` var: comma-separated
    ``label:collection`` pairs. A bare token without a colon is treated as
    ``name:name`` so a flat collection-name list still yields a usable map
    (label == collection). Returns None when no value is supplied (absent/blank
    header), signalling that the env-derived map/scope should apply instead.
    """
    if not raw:
        return None
    mapping: dict[str, str] = {}
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            label, _, collection = token.partition(":")
            label, collection = label.strip(), collection.strip()
            if label and collection:
                mapping[label] = collection
        else:
            mapping[token] = token
    return mapping or None


def parse_collection_scope(raw: str | None) -> set[str] | None:
    """The set of collections a header value permits (its map's values)."""
    mapping = parse_collection_map(raw)
    return set(mapping.values()) if mapping else None


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        rerank_settings: Optional[RerankSettings] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self._qdrant_settings = qdrant_settings
        self.rerank_settings = rerank_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        is_cloud = (
            self.embedding_provider_settings is not None
            and self.embedding_provider_settings.provider_type
            == EmbeddingProviderType.CLOUD
        )

        reranker = None
        if rerank_settings and rerank_settings.enabled and rerank_settings.api_key:
            reranker = VoyageReranker(
                api_key=rerank_settings.api_key,
                model=rerank_settings.model,
                base_url=rerank_settings.base_url,
            )

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
            cloud_inference=is_cloud,
            sparse_model=(
                self.embedding_provider_settings.sparse_model
                if is_cloud and self.embedding_provider_settings
                else None
            ),
            reranker=reranker,
            rerank_candidate_limit=(
                rerank_settings.candidate_limit if rerank_settings else 40
            ),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def _env_collection_map(self) -> dict[str, str] | None:
        """The {label: collection} map derived from process-level env config.

        Used as the fallback when no per-connection header is present (e.g.
        stdio transport). Returns None when nothing is configured, meaning every
        collection is permitted (legacy single/no-collection behavior unchanged).
        """
        if self.qdrant_settings.qdrant_collections:
            return dict(self.qdrant_settings.qdrant_collections)
        if self.qdrant_settings.collection_name:
            name = self.qdrant_settings.collection_name
            return {name: name}
        return None

    def _active_collection_map(self) -> dict[str, str] | None:
        """Resolve the active {label: collection} map for the current request.

        Reads the per-connection ``X-Qdrant-Collections`` header at request time.
        When present it overrides the env-derived map FOR THAT REQUEST; when
        absent the env-derived map applies (backward compat for stdio).
        """
        header_map = parse_collection_map(
            get_http_headers().get(COLLECTION_SCOPE_HEADER)
        )
        if header_map is not None:
            return header_map
        return self._env_collection_map()

    def _active_collection_scope(self) -> set[str] | None:
        """The set of collections the current request may touch (None = unrestricted)."""
        active_map = self._active_collection_map()
        if active_map is None:
            return None
        return set(active_map.values())

    def _resolve_collection_for_label(self, label: str) -> str | None:
        """Resolve a labeled tool to its target collection for the current request.

        Per-connection ROUTING: the header map wins, so ONE shared server can
        serve many repos — each connection's ``memory`` tool resolves to that
        connection's own collection. Falls back to the env-configured collection
        for the label when the header omits it (stdio / single-process use).
        """
        active_map = self._active_collection_map()
        if active_map and label in active_map:
            return active_map[label]
        if self.qdrant_settings.qdrant_collections:
            return self.qdrant_settings.qdrant_collections.get(label)
        return None

    def _enforce_collection_scope(self, collection_name: str | None) -> None:
        """Deny operations targeting a collection outside the active scope."""
        scope = self._active_collection_scope()
        if scope is None:
            return
        if collection_name is None or collection_name not in scope:
            raise CollectionScopeError(
                f"Access to collection {collection_name!r} is denied: it is not "
                f"within the active collection scope {sorted(scope)!r}."
            )

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            self._enforce_collection_scope(collection_name)

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            self._enforce_collection_scope(collection_name)

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def update_metadata(
            ctx: Context,
            point_ids: Annotated[
                list[str], Field(description="List of point IDs to update")
            ],
            metadata: Annotated[
                Metadata,
                Field(
                    description="Metadata fields to set or overwrite on the points. Merged with existing metadata."
                ),
            ],
            collection_name: Annotated[
                str, Field(description="The collection containing the points")
            ],
        ) -> str:
            """
            Update metadata on existing points without re-embedding.
            """
            await ctx.debug(f"Updating metadata on {len(point_ids)} points")

            self._enforce_collection_scope(collection_name)

            payload_update = {f"metadata.{k}": v for k, v in metadata.items()}
            count = await self.qdrant_connector.update_payload(
                point_ids, payload_update, collection_name=collection_name
            )
            return f"Updated metadata on {count} point(s)"

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if self.qdrant_settings.qdrant_collections:
            self._register_multi_collection_tools(
                find, store, update_metadata, filterable_conditions
            )
            return

        find_foo = find
        store_foo = store
        update_foo = update_metadata

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            update_foo = make_partial_function(
                update_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                update_foo,
                name="qdrant-update",
                description="Update metadata on existing points in Qdrant.",
            )

    def _register_multi_collection_tools(
        self, find, store, update_metadata, filterable_conditions
    ):
        """Register labeled find/store/update tools for each entry in qdrant_collections.

        Each tool's target collection is bound to its LABEL and resolved per
        request (``_resolve_collection_for_label``), so one shared HTTP server
        routes a connection's ``memory`` tool to that connection's own
        collection rather than a single process-global one.
        """

        def route(label):
            return lambda label=label: self._resolve_collection_for_label(label)

        for label in self.qdrant_settings.qdrant_collections:
            find_foo = make_routed_function(find, "collection_name", route(label))
            if len(filterable_conditions) > 0:
                find_foo = wrap_filters(find_foo, filterable_conditions)
            elif not self.qdrant_settings.allow_arbitrary_filter:
                find_foo = make_partial_function(find_foo, {"query_filter": None})

            self.tool(
                find_foo,
                name=f"qdrant-find-{label}",
                description=self.tool_settings.get_find_description(label),
            )

            if not self.qdrant_settings.read_only:
                store_foo = make_routed_function(store, "collection_name", route(label))
                self.tool(
                    store_foo,
                    name=f"qdrant-store-{label}",
                    description=self.tool_settings.get_store_description(label),
                )

                update_foo = make_routed_function(
                    update_metadata, "collection_name", route(label)
                )
                self.tool(
                    update_foo,
                    name=f"qdrant-update-{label}",
                    description=f"Update metadata on existing points in the {label} collection.",
                )

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport, injecting channel capability when enabled."""
        experimental: dict[str, dict[str, Any]] | None = None
        if self._qdrant_settings.channel_enabled:
            experimental = {"claude/channel": {}}

        init_options = self._mcp_server.create_initialization_options(
            NotificationOptions(tools_changed=True),
            experimental_capabilities=experimental,
        )

        async with stdio_server() as (read_stream, write_stream):
            logger.info(f"Starting MCP server {self.name!r} with transport 'stdio'")

            if self._should_run_channel_watcher():
                await self._run_with_channel_watcher(
                    read_stream, write_stream, init_options
                )
            else:
                await self._mcp_server.run(read_stream, write_stream, init_options)

    def _should_run_channel_watcher(self) -> bool:
        settings = self._qdrant_settings
        return bool(
            settings.channel_enabled
            and settings.channel_journal_path
            and settings.qdrant_collections
        )

    async def _run_with_channel_watcher(
        self, read_stream: Any, write_stream: Any, init_options: Any
    ) -> None:
        """Run the MCP server alongside a background journal watcher task."""
        settings = self._qdrant_settings

        class _WriteStreamProxy:
            def __init__(self, ws: Any) -> None:
                self._write_stream = ws

        orchestrator = ChannelOrchestrator(
            connector=self.qdrant_connector,
            collections={
                k: v for k, v in (settings.qdrant_collections or {}).items()
                if k == "memory"
            },
            session=_WriteStreamProxy(write_stream),
            journal_path=settings.channel_journal_path,  # type: ignore[arg-type]
            similarity_threshold=settings.channel_similarity_threshold,
            cooldown_seconds=settings.channel_cooldown_seconds,
            max_per_session=settings.channel_max_per_session,
            project_filter=settings.channel_project_filter,
            suppress_tags=settings.channel_suppress_tags,
            episode_max_age_days=settings.channel_episode_max_age_days,
        )

        async def _watch_journal() -> None:
            while True:
                await anyio.sleep(5)
                try:
                    await orchestrator.check_journal()
                except Exception:
                    logger.exception("Channel journal watcher error (suppressed)")

        async with anyio.create_task_group() as tg:
            tg.start_soon(_watch_journal)
            await self._mcp_server.run(read_stream, write_stream, init_options)
            tg.cancel_scope.cancel()
