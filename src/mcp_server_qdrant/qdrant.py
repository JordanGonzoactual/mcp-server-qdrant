import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.reranker import VoyageReranker
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]

SPARSE_VECTOR_NAME = "sparse"


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None
    score: float | None = None
    id: str | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    :param cloud_inference: Whether to use Qdrant Cloud server-side inference with Document() objects.
    :param sparse_model: The sparse embedding model for hybrid search (e.g., "qdrant/bm25").
                         Only used when cloud_inference is True.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
        cloud_inference: bool = False,
        sparse_model: str | None = None,
        reranker: VoyageReranker | None = None,
        rerank_candidate_limit: int = 40,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._cloud_inference = cloud_inference
        self._sparse_model = sparse_model
        self._reranker = reranker
        self._rerank_candidate_limit = rerank_candidate_limit

        client_kwargs: dict[str, Any] = {
            "location": qdrant_url,
            "api_key": qdrant_api_key,
            "path": qdrant_local_path,
        }
        if cloud_inference:
            client_kwargs["cloud_inference"] = True

        self._client = AsyncQdrantClient(**client_kwargs)
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        payload = {"document": entry.content, METADATA_PATH: entry.metadata}

        if self._cloud_inference:
            # Use Document objects for server-side embedding
            dense_model = self._embedding_provider.model_name
            dense_name = self._embedding_provider.get_vector_name()
            dense_options = self._embedding_provider.document_options("search_document")
            vector: dict[str, Any] = {
                dense_name: models.Document(
                    text=entry.content, model=dense_model, options=dense_options
                )
            }
            if self._sparse_model:
                vector[SPARSE_VECTOR_NAME] = models.Document(
                    text=entry.content, model=self._sparse_model
                )

            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        else:
            # Local embedding path (FastEmbed)
            embeddings = await self._embedding_provider.embed_documents([entry.content])
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={vector_name: embeddings[0]},
                        payload=payload,
                    )
                ],
            )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.
        :param score_threshold: Minimum score to include in results. For hybrid/RRF search,
                                scores are typically 0.01-0.06; for cosine similarity, 0.0-1.0.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # When a reranker is active, fetch a larger candidate set to feed it,
        # then truncate to `limit` after reranking.
        fetch_limit = (
            max(limit, self._rerank_candidate_limit) if self._reranker else limit
        )

        if self._cloud_inference and self._sparse_model:
            # Hybrid search: dense + sparse with Reciprocal Rank Fusion
            dense_model = self._embedding_provider.model_name
            dense_name = self._embedding_provider.get_vector_name()
            dense_options = self._embedding_provider.document_options("search_query")

            search_results = await self._client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(
                            text=query, model=dense_model, options=dense_options
                        ),
                        using=dense_name,
                        limit=fetch_limit,
                    ),
                    models.Prefetch(
                        query=models.Document(
                            text=query, model=self._sparse_model
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=fetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=fetch_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
        elif self._cloud_inference:
            # Cloud inference, dense only (no sparse model configured)
            dense_model = self._embedding_provider.model_name
            dense_name = self._embedding_provider.get_vector_name()
            dense_options = self._embedding_provider.document_options("search_query")

            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=models.Document(
                    text=query, model=dense_model, options=dense_options
                ),
                using=dense_name,
                limit=fetch_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
        else:
            # Local embedding path (FastEmbed)
            query_vector = await self._embedding_provider.embed_query(query)
            vector_name = self._embedding_provider.get_vector_name()

            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=fetch_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

        entries = [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
                score=result.score,
                id=str(result.id) if result.id is not None else None,
            )
            for result in search_results.points
        ]

        # Final reranking stage: reorder the fused candidates by cross-encoder
        # relevance and truncate to the requested limit. Falls back to fusion
        # order on any reranker failure.
        if self._reranker is not None and entries:
            order = await self._reranker.rerank(
                query, [entry.content for entry in entries], top_k=limit
            )
            entries = [entries[index] for index in order]

        return entries

    async def update_payload(
        self,
        point_ids: list[str],
        payload: dict[str, Any],
        *,
        collection_name: str | None = None,
    ) -> int:
        """
        Update payload fields on existing points.
        :param point_ids: List of point IDs to update.
        :param payload: Payload fields to set/overwrite (merged with existing payload).
        :param collection_name: The collection containing the points.
        :return: Number of points updated.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        await self._client.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=point_ids,
        )
        return len(point_ids)

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()

            sparse_config = None
            if self._sparse_model:
                sparse_config = {
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }

            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config=sparse_config,
            )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
