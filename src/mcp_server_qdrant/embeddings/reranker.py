import logging

import httpx

logger = logging.getLogger(__name__)


class VoyageReranker:
    """
    Reorders a candidate set by relevance using the Voyage rerank API.

    Applied as a final stage after dense+sparse fusion: the fused candidates are
    handed to the cross-encoder reranker, which scores each against the query
    directly. On any failure (network, auth, malformed response) it falls back
    to the original fusion order so search never breaks because of reranking.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-2.5",
        base_url: str = "https://api.voyageai.com/v1/rerank",
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Return candidate indices ordered best-first, truncated to top_k.

        Indices refer to positions in the input ``documents`` list. On failure,
        returns the original order truncated to top_k.
        """
        keep = min(top_k, len(documents))
        if keep <= 0:
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._base_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "query": query,
                        "documents": documents,
                        "model": self._model,
                        "top_k": keep,
                    },
                )
                response.raise_for_status()
                data = response.json()["data"]
            order = [item["index"] for item in data]
            if not order:
                raise ValueError("reranker returned no results")
            return order[:keep]
        except Exception:
            logger.warning(
                "Voyage rerank failed; falling back to fusion order", exc_info=True
            )
            return list(range(keep))
