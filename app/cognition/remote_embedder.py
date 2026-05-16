"""
RemoteEmbedder — calls the PC embedding server over LAN.

Falls back to the local EmbeddingService if the remote server is unreachable.
Toggle via EMBED_SERVER_URL in config/constants.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Timeout for embedding requests (seconds)
_TIMEOUT = 10.0


class RemoteEmbedder:
    """Embedding client that calls the external embed_server.py on the PC."""

    def __init__(self, server_url: str):
        self._url = server_url.rstrip("/")
        self._healthy: Optional[bool] = None

    async def health_check(self) -> bool:
        """Check if the remote embedding server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._url}/health")
                self._healthy = r.status_code == 200
                return self._healthy
        except Exception:
            self._healthy = False
            return False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the remote server. Raises on failure."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(f"{self._url}/embed", json={"texts": texts})
            r.raise_for_status()
            return r.json()["embeddings"]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding for non-async contexts."""
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.post(f"{self._url}/embed", json={"texts": texts})
            r.raise_for_status()
            return r.json()["embeddings"]
