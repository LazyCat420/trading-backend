import pytest
from unittest.mock import MagicMock, patch
from app.services.embedding_service import EmbeddingService

def test_embed_batch_fallback_to_prism():
    """Test that if the primary API fails, it successfully falls back to Prism."""
    # Mock settings
    with patch("app.config.settings") as mock_settings:
        mock_settings.EMBEDDING_SERVER_URL = "http://failed-url:8001/embed"
        mock_settings.PRISM_URL = "http://prism-url:7777"
        mock_settings.PRISM_PROJECT = "test-project"
        mock_settings.PRISM_USERNAME = "test-user"
        
        service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
        
        # Mock httpx client
        mock_client = MagicMock()
        
        # Set up side effect: first call (primary server) raises exception
        # Second call (Prism fallback) returns mock response with 768-dim vector
        def mock_post(url, *args, **kwargs):
            if "failed-url" in url:
                raise Exception("Connection Refused")
            elif "prism-url" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "embedding": [0.5] * 768,
                    "dimensions": 768,
                    "provider": "lm-studio",
                    "model": "test-model"
                }
                return resp
            raise Exception(f"Unexpected URL: {url}")
            
        mock_client.post.side_effect = mock_post
        
        with patch.object(service, "_get_client", return_value=mock_client):
            res = service.embed_batch(["hello", "world"], batch_size=1, show_progress=False)
            
            assert len(res) == 2
            # Should slice to 384 dimensions
            assert len(res[0]) == 384
            assert len(res[1]) == 384
            assert res[0][0] == 0.5
            
            # Verify mock_client.post was called for failed-url and prism-url
            assert mock_client.post.call_count == 4 # 2 for failed-url, 2 for prism-url
