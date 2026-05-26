import pytest
from unittest.mock import MagicMock, patch
import asyncio
import time

from app.pipeline.data.data_janitor import run_data_janitor


@pytest.mark.asyncio
async def test_janitor_concurrency(monkeypatch, mock_db):
    """Verify that evaluate_relevance is called concurrently for all items."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.pipeline.data.data_janitor.get_db", mock_get_db)
    db = mock_db

    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "news_articles" in query and "SELECT" in query:
            cursor.fetchall.return_value = [
                ("n1", "Long enough content to trigger evaluation 12345"),
                ("n2", "Long enough content to trigger evaluation 12345"),
            ]
        elif "reddit_posts" in query and "SELECT" in query and "LIMIT 20" in query:
            cursor.fetchall.return_value = [
                ("r1", "Long enough content to trigger evaluation 12345"),
                ("r2", "Long enough content to trigger evaluation 12345"),
            ]
        elif (
            "youtube_transcripts" in query and "SELECT" in query and "LIMIT 10" in query
        ):
            cursor.fetchall.return_value = [
                ("y1", "Long enough content to trigger evaluation 12345"),
                ("y2", "Long enough content to trigger evaluation 12345"),
            ]
        else:
            cursor.fetchall.return_value = []
        return cursor

    db.execute.side_effect = mock_execute

    call_count = 0
    active_tasks = 0
    max_active_tasks = 0

    async def mock_eval(*args, **kwargs):
        nonlocal call_count, active_tasks, max_active_tasks
        call_count += 1
        active_tasks += 1
        max_active_tasks = max(max_active_tasks, active_tasks)
        await asyncio.sleep(0.01) # Yield to event loop to allow concurrent tasks to start
        active_tasks -= 1
        return {"status": "discarded", "reason": "Test discard"}

    monkeypatch.setattr("app.pipeline.data.data_janitor.evaluate_relevance", mock_eval)

    # Run the janitor
    metrics = await run_data_janitor(emit=MagicMock())

    # Verify concurrency without relying on wall-clock time
    assert call_count == 6
    assert max_active_tasks > 1, f"Tasks were not concurrent, max active: {max_active_tasks}"

    # Check that it updated the metrics correctly
    assert metrics["scanned"] == 6
    assert metrics["discarded"] == 6


@pytest.mark.asyncio
async def test_janitor_db_updates(monkeypatch, mock_db):
    """Verify that the database is correctly updated with discarded items."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.pipeline.data.data_janitor.get_db", mock_get_db)
    db = mock_db

    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "news_articles" in query and "SELECT" in query:
            cursor.fetchall.return_value = [
                ("n1", "Long enough content to trigger evaluation 12345")
            ]
        elif "reddit_posts" in query and "SELECT" in query and "LIMIT 20" in query:
            cursor.fetchall.return_value = []
        elif (
            "youtube_transcripts" in query and "SELECT" in query and "LIMIT 10" in query
        ):
            cursor.fetchall.return_value = [
                ("y1", "Long enough content to trigger evaluation 12345")
            ]
        else:
            cursor.fetchall.return_value = []
        return cursor

    db.execute.side_effect = mock_execute

    async def mock_eval(*args, **kwargs):
        # Discard everything
        return {"status": "discarded", "reason": "Spam"}

    monkeypatch.setattr("app.pipeline.data.data_janitor.evaluate_relevance", mock_eval)

    await run_data_janitor(emit=MagicMock())

    # Check that execute was called with UPDATE statements
    update_calls = [c for c in db.execute.call_args_list if "UPDATE" in str(c)]
    assert len(update_calls) == 2

    # Check first update (news)
    news_call = update_calls[0]
    assert "UPDATE news_articles" in news_call.args[0]
    assert news_call.args[1] == ["discarded", "Spam", "n1"]

    # Check second update (yt)
    yt_call = update_calls[1]
    assert "UPDATE youtube_transcripts" in yt_call.args[0]
    assert yt_call.args[1] == ["discarded", "Spam", "y1"]


@pytest.mark.asyncio
async def test_janitor_exception_handling(monkeypatch, mock_db):
    """Verify that exceptions in evaluate_relevance do not crash the gather."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.pipeline.data.data_janitor.get_db", mock_get_db)
    db = mock_db

    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "news_articles" in query and "SELECT" in query:
            cursor.fetchall.return_value = [
                ("n1", "Long enough content to trigger evaluation 12345"),
                ("n2", "Long enough content to trigger evaluation 12345"),
            ]
        else:
            cursor.fetchall.return_value = []
        return cursor

    db.execute.side_effect = mock_execute

    call_count = 0

    async def mock_eval(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated network failure")
        return {"status": "discarded", "reason": "Test discard"}

    monkeypatch.setattr("app.pipeline.data.data_janitor.evaluate_relevance", mock_eval)

    metrics = await run_data_janitor(emit=MagicMock())

    assert metrics["scanned"] == 1
    assert metrics["discarded"] == 1

from app.agents.janitor_agent import run_janitor_cleanup

@pytest.mark.integration
@pytest.mark.asyncio
async def test_janitor_agent_archive_old_data(real_db):
    """Verify that the janitor agent moves old news to data_archive and deletes them in the real DB."""
    from contextlib import contextmanager
    import datetime

    @contextmanager
    def mock_get_db():
        yield real_db

    # Insert old records to test archiving (older than 14 days)
    old_dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=20)
    
    real_db.execute(
        "INSERT INTO news_articles (id, ticker, title, publisher, url, published_at, summary) "
        "VALUES ('n_old_1', 'AAPL', 'Old Title', 'Pub', 'url1', %s, 'Old Summary')",
        [old_dt]
    )
    real_db.execute(
        "INSERT INTO reddit_posts (id, ticker, subreddit, title, body, score, upvote_ratio, comment_count, created_utc) "
        "VALUES ('r_old_1', 'MSFT', 'wallstreetbets', 'Old Post', 'Old Body', 10, 1.0, 5, %s)",
        [old_dt]
    )

    with patch("app.agents.janitor_agent.get_db", new=mock_get_db):
        await run_janitor_cleanup()

    # Verify that the records were moved to data_archive
    archives = real_db.execute("SELECT source_table, source_id, ticker FROM data_archive").fetchall()
    assert len(archives) == 2
    
    # Verify that they were deleted from source tables
    news_count = real_db.execute("SELECT COUNT(*) FROM news_articles WHERE id = 'n_old_1'").fetchone()[0]
    assert news_count == 0
    
    reddit_count = real_db.execute("SELECT COUNT(*) FROM reddit_posts WHERE id = 'r_old_1'").fetchone()[0]
    assert reddit_count == 0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_janitor_agent_prune_quarantine_and_debug_data(real_db):
    """Verify that the janitor agent prunes expired quarantine and stale debug data."""
    from contextlib import contextmanager
    import datetime

    @contextmanager
    def mock_get_db():
        yield real_db

    # Create tables if they don't exist in the test db
    real_db.execute("""
    CREATE TABLE IF NOT EXISTS ticker_quarantine (
        ticker VARCHAR(20) PRIMARY KEY,
        reason VARCHAR(50) NOT NULL,
        details TEXT,
        quarantined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """)
    real_db.execute("""
    CREATE TABLE IF NOT EXISTS rejected_symbols (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(50) NOT NULL,
        reason TEXT,
        source VARCHAR(50),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Insert old records for pruning
    old_dt_quar = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10) # > 7 days
    new_dt_quar = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=2) # < 7 days
    
    old_dt_debug = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=40) # > 30 days
    new_dt_debug = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10) # < 30 days

    real_db.execute(
        "INSERT INTO ticker_quarantine (ticker, reason, details, quarantined_at) VALUES ('QOLD', 'test', 'test', %s)",
        [old_dt_quar]
    )
    real_db.execute(
        "INSERT INTO ticker_quarantine (ticker, reason, details, quarantined_at) VALUES ('QNEW', 'test', 'test', %s)",
        [new_dt_quar]
    )

    real_db.execute(
        "INSERT INTO rejected_symbols (symbol, reason, source, created_at) VALUES ('R_OLD', 'test', 'test', %s)",
        [old_dt_debug]
    )
    real_db.execute(
        "INSERT INTO rejected_symbols (symbol, reason, source, created_at) VALUES ('R_NEW', 'test', 'test', %s)",
        [new_dt_debug]
    )

    with patch("app.agents.janitor_agent.get_db", new=mock_get_db):
        from app.agents.janitor_agent import run_janitor_cleanup
        await run_janitor_cleanup()

    # Verify that the old quarantine entry was deleted, but the new one remains
    quar_count = real_db.execute("SELECT COUNT(*) FROM ticker_quarantine").fetchone()[0]
    assert quar_count == 1
    rem_quar = real_db.execute("SELECT ticker FROM ticker_quarantine").fetchone()[0]
    assert rem_quar == 'QNEW'

    # Verify that the old debug entry was deleted, but the new one remains
    debug_count = real_db.execute("SELECT COUNT(*) FROM rejected_symbols").fetchone()[0]
    assert debug_count == 1
    rem_debug = real_db.execute("SELECT symbol FROM rejected_symbols").fetchone()[0]
    assert rem_debug == 'R_NEW'



@pytest.mark.integration
@pytest.mark.asyncio
async def test_janitor_agent_purges_traces_stats_approvals(real_db):
    """JAN-03, JAN-04, JAN-05, JAN-06, JAN-07: Verify trace, stat, approval purges and run log."""
    from contextlib import contextmanager
    import datetime
    from app.agents.janitor_agent import run_janitor_cleanup

    @contextmanager
    def mock_get_db():
        yield real_db

    # Create tables if they don't exist
    real_db.execute("""
    CREATE TABLE IF NOT EXISTS agent_traces (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS agent_loop_stats (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS pending_approvals (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS janitor_run_log (
        id SERIAL PRIMARY KEY,
        run_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        details TEXT
    );
    """)

    old_dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10)
    new_dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)

    # Insert old and new records
    real_db.execute("INSERT INTO agent_traces (id, created_at) VALUES (%s, %s)", ["trace_old", old_dt])
    real_db.execute("INSERT INTO agent_traces (id, created_at) VALUES (%s, %s)", ["trace_new", new_dt])
    real_db.execute("INSERT INTO agent_loop_stats (id, created_at) VALUES (%s, %s)", ["stat_old", old_dt])
    real_db.execute("INSERT INTO agent_loop_stats (id, created_at) VALUES (%s, %s)", ["stat_new", new_dt])
    real_db.execute("INSERT INTO pending_approvals (id, created_at) VALUES (%s, %s)", ["app_old", old_dt])
    real_db.execute("INSERT INTO pending_approvals (id, created_at) VALUES (%s, %s)", ["app_new", new_dt])

    with patch("app.agents.janitor_agent.get_db", new=mock_get_db):
        await run_janitor_cleanup()

    # Verify old records are deleted
    traces = real_db.execute("SELECT COUNT(*) FROM agent_traces").fetchone()[0]
    assert traces == 1

    stats = real_db.execute("SELECT COUNT(*) FROM agent_loop_stats").fetchone()[0]
    assert stats == 1

    approvals = real_db.execute("SELECT COUNT(*) FROM pending_approvals").fetchone()[0]
    assert approvals == 1

    # Verify run log was written
    logs = real_db.execute("SELECT details FROM janitor_run_log ORDER BY run_time DESC LIMIT 1").fetchall()
    assert len(logs) == 1
    import json
    details = json.loads(logs[0][0])
    assert details["traces_deleted"] > 0
    assert details["stats_deleted"] > 0
    assert details["approvals_deleted"] > 0
    assert details["status"] == "success"
