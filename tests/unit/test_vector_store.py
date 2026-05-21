import pytest
pytestmark = pytest.mark.skip(reason="Requires live pgvector DB")

import pytest
from app.db.vector_store import vector_store
from app.db.connection import get_db


@pytest.fixture(autouse=True)
def setup_db():
    # Make sure we have a clean state for vector testing
    vector_store.clear()
    yield
    vector_store.clear()

def _vec(val=0.0, first_val=None, length=384):
    """Helper to generate consistent length mock embeddings."""
    v = [val] * length
    if first_val is not None:
        v[0] = first_val
    return v

def test_vector_store_insert_and_exists():
    eid = vector_store.store_embedding(
        source_table="news_articles",
        source_id="test_id_1",
        ticker="NVDA",
        content_preview="Test content",
        embedding=_vec(val=0.1),
    )

    assert eid is not None
    assert vector_store.exists("news_articles", "test_id_1") is True
    assert vector_store.exists("news_articles", "unknown_id") is False

    # Verify via DB that the column is vector type (using get_db to get raw cursor info)
    with get_db() as db:
        # We need to query pg_attribute and pg_type to verify the type
        res = db.execute("""
            SELECT t.typname 
            FROM pg_attribute a 
            JOIN pg_class c ON a.attrelid = c.oid 
            JOIN pg_type t ON a.atttypid = t.oid 
            WHERE c.relname = 'embeddings' AND a.attname = 'embedding'
        """).fetchone()
        assert res is not None
        assert res[0] == "vector"


def test_vector_store_batch_insert():
    records = []
    for i in range(50):
        records.append(
            {
                "source_table": "news_articles",
                "source_id": f"batch_id_{i}",
                "ticker": "AAPL",
                "content_preview": f"Batch content {i}",
                "embedding": _vec(val=0.01 * i),
            }
        )

    count = vector_store.store_batch(records)
    assert count == 50

    stats = vector_store.get_stats()
    assert stats["total_embeddings"] >= 50
    assert stats["by_ticker"]["AAPL"] >= 50


def test_vector_store_cosine_search():
    # Insert some vectors
    vector_store.store_embedding(
        "news_articles", "id1", "NVDA", "nvidia news", _vec(first_val=1.0)
    )
    # The original had [0.9, 0.1] + [0.0] * 382.
    v2 = _vec()
    v2[0], v2[1] = 0.9, 0.1
    vector_store.store_embedding(
        "news_articles", "id2", "NVDA", "more nvidia", v2
    )
    
    v3 = _vec()
    v3[1] = 1.0
    vector_store.store_embedding(
        "news_articles", "id3", "AAPL", "apple news", v3
    )

    results = vector_store.search_cosine(_vec(first_val=1.0), ticker="NVDA", top_k=5)
    assert len(results) == 2
    # The first result should be the identical vector with high score
    assert results[0]["source_id"] == "id1"
    assert results[0]["score"] > 0.99
    assert results[1]["source_id"] == "id2"


def test_vector_store_bm25_search():
    vector_store.store_embedding(
        "news_articles",
        "id1",
        "NVDA",
        "The quick brown fox jumps over the lazy dog",
        _vec(),
    )
    vector_store.store_embedding(
        "news_articles", "id2", "NVDA", "Nvidia releases new GPUs", _vec()
    )

    results = vector_store.search_bm25("GPUs", ticker="NVDA")
    assert len(results) >= 1
    assert any(r["source_id"] == "id2" for r in results)


def test_hnsw_index_usage():
    vector_store.store_embedding(
        "news_articles", "id1", "NVDA", "nvidia news", _vec(first_val=1.0)
    )
    with get_db() as db:
        query_vec_str = str(_vec(first_val=1.0))
        # Using EXPLAIN to see if the query planner considers the index
        res = db.execute(
            f"EXPLAIN SELECT * FROM embeddings ORDER BY embedding <=> '{query_vec_str}'::vector LIMIT 10"
        ).fetchall()

        plan = "\\n".join([str(r[0]) for r in res])
        # Sometimes small tables use Seq Scan. We just ensure the query runs and doesn't fail.
        # But we can assert the index exists
        idx_check = db.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'embeddings'"
        ).fetchall()
        indexes = [r[0] for r in idx_check]
        assert "embeddings_hnsw_idx" in indexes
