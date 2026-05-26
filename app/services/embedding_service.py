"""
Embedding Service — Shared embedding engine for the RAG pipeline.

Uses sentence-transformers with BAAI/bge-small-en-v1.5 (384-dim vectors).
Handles text chunking and batch embedding.

Why bge-small over all-MiniLM-L6-v2:
  - Same 384 dimensions (matches existing FLOAT[384] schema)
  - +6 MTEB points on retrieval tasks
  - Supports instruction-prefixed queries for better retrieval

Usage:
    from app.services.embedding_service import embedder
    vec = embedder.embed_text("NVDA earnings beat expectations")
    vecs = embedder.embed_batch(["text1", "text2", "text3"])
    chunks = embedder.chunk_text("very long article...", max_tokens=512)
"""

import logging
import re

logger = logging.getLogger(__name__)

# Model config
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 512  # tokens (~2048 chars)
DEFAULT_CHUNK_OVERLAP = 51  # ~10% overlap
CHARS_PER_TOKEN = 4  # rough approximation


import threading

class EmbeddingService:
    """Lazy-loading embedding service using sentence-transformers."""

    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        from app.config import settings
        # We assume the embedding server is OpenAI-compatible (e.g., vLLM or TEI)
        self.api_url = settings.EMBEDDING_SERVER_URL
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import httpx
                    limits = httpx.Limits(max_connections=50, max_keepalive_connections=10)
                    self._client = httpx.Client(timeout=60.0, limits=limits)
        return self._client

    def close(self):
        """Close the persistent client connection pool."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return EMBEDDING_DIM

    def embed_text(self, text: str, prefix: str = "") -> list[float]:
        """Embed a single text string via HTTP API."""
        return self.embed_batch([text], prefix=prefix, show_progress=False)[0]

    def embed_batch(
        self,
        texts: list[str],
        prefix: str = "",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed a batch of texts efficiently via HTTP API."""
        if not texts:
            return []
            
        if prefix:
            texts = [prefix + t for t in texts]

        if show_progress and len(texts) > batch_size:
            logger.info(f"Embedding {len(texts)} texts via API at {self.api_url}")

        results = []
        client = self._get_client()
        
        # Check if we should bypass the local API call and use Prism directly
        from app.config import settings
        bypass_local = ("localhost" in self.api_url or "127.0.0.1" in self.api_url) and settings.PRISM_ENABLED
        
        # Chunk into batches so we don't overload the API payload size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            success = False
            
            if not bypass_local:
                try:
                    # OpenAI compatible request
                    payload = {
                        "model": self._model_name,
                        "input": batch,
                    }
                    response = client.post(self.api_url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    if "data" in data:
                        # OpenAI format
                        sorted_data = sorted(data["data"], key=lambda x: x["index"])
                        embeddings = [item["embedding"] for item in sorted_data]
                        results.extend(embeddings)
                        success = True
                    else:
                        # Fallback for simple custom servers that just return a list of lists
                        results.extend(data)
                        success = True
                        
                except Exception as e:
                    logger.info(f"Failed to fetch embeddings from primary API (will fallback to Prism): {e}")
                    
            if not success:
                # Try fallback to Prism
                try:
                    logger.info("Attempting fallback to Prism embedding gateway...")
                    fallback_embeddings = []
                    for text in batch:
                        # Call Prism's /embed endpoint
                        url = f"{settings.PRISM_URL}/embed"
                        payload = {
                            "provider": "lm-studio",
                            "text": text
                        }
                        headers = {
                            "x-project": settings.PRISM_PROJECT,
                            "x-username": settings.PRISM_USERNAME,
                            "Content-Type": "application/json"
                        }
                        resp = client.post(url, json=payload, headers=headers)
                        resp.raise_for_status()
                        res_data = resp.json()
                        emb = res_data["embedding"]
                        
                        # Adjust dimension to EMBEDDING_DIM (384)
                        if len(emb) > EMBEDDING_DIM:
                            emb = emb[:EMBEDDING_DIM]
                        elif len(emb) < EMBEDDING_DIM:
                            emb = emb + [0.0] * (EMBEDDING_DIM - len(emb))
                        fallback_embeddings.append(emb)
                        
                    results.extend(fallback_embeddings)
                    logger.info(f"Successfully retrieved fallback embeddings from Prism for batch of size {len(batch)}.")
                except Exception as fallback_err:
                    logger.critical(f"Prism embedding fallback also failed: {fallback_err}")
                    # Return zero vectors as fallback so the pipeline doesn't crash
                    results.extend([[0.0] * EMBEDDING_DIM for _ in batch])

        return results


    def chunk_text(
        self,
        text: str,
        max_tokens: int = DEFAULT_CHUNK_SIZE,
        overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[str]:
        """Split text into overlapping chunks respecting sentence boundaries.

        Uses recursive character splitting with sentence-boundary awareness.
        Each chunk is approximately max_tokens tokens (~max_tokens*4 chars).

        Args:
            text: Input text to chunk.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Number of overlapping tokens between chunks.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        max_chars = max_tokens * CHARS_PER_TOKEN
        overlap_chars = overlap_tokens * CHARS_PER_TOKEN

        # If text fits in one chunk, return as-is
        if len(text) <= max_chars:
            return [text.strip()]

        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_len = len(sentence)

            # If single sentence exceeds max, split by characters
            if sent_len > max_chars:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0

                # Hard split the long sentence
                for i in range(0, sent_len, max_chars - overlap_chars):
                    chunk = sentence[i : i + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)
                continue

            # Adding this sentence would exceed limit
            if current_len + sent_len + 1 > max_chars and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Overlap: keep last sentences that fit in overlap window
                overlap_text = ""
                overlap_sents = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) + 1 <= overlap_chars:
                        overlap_sents.insert(0, s)
                        overlap_text = " ".join(overlap_sents)
                    else:
                        break

                current_chunk = overlap_sents
                current_len = len(overlap_text)

            current_chunk.append(sentence)
            current_len += sent_len + 1

        # Flush remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]


# Module-level singleton
embedder = EmbeddingService()
"""
Global embedding service instance. Import and use directly:

    from app.services.embedding_service import embedder
    vec = embedder.embed_text("some text")
"""
