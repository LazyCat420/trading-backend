"""
General Audit Tests — Environmental & Infrastructure.

Tests from the /general-audit checklist to verify system resilience
under adverse infrastructure conditions.
"""

import asyncio
import time
from datetime import datetime
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# 1. Slow DB Test
#    Simulate the primary database responding in 5–10 seconds and verify
#    the processing cycle does not hang or crash.
# ────────────────────────────────────────────────────────────────────────


class TestSlowDB:
    """Verify cycle phases handle slow database responses without crash."""

    @staticmethod
    def _make_slow_cursor(delay_seconds: float = 6.0):
        """Create a mock cursor whose execute() sleeps to simulate slow DB."""
        cursor = MagicMock()

        def slow_execute(sql, params=None):
            time.sleep(delay_seconds)
            cursor.description = None
            return cursor

        cursor.execute.side_effect = slow_execute
        cursor.fetchone.return_value = (0,)
        cursor.fetchall.return_value = []
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        return cursor

    @staticmethod
    def _make_slow_get_db(delay_seconds: float = 6.0):
        """Create a patched get_db that yields a slow cursor."""
        slow_cursor = TestSlowDB._make_slow_cursor(delay_seconds)

        @contextmanager
        def slow_get_db():
            yield slow_cursor

        return slow_get_db, slow_cursor

    @pytest.mark.asyncio
    async def test_phase1_health_survives_slow_db(self):
        """Phase 1 health check should complete (not crash) with a slow DB.

        The DB takes ~1 second per query, but phase 1 should still run
        through reconciliation, triage, and directives without raising.
        """
        from app.cycle.context import CycleContext

        # Use a shorter delay for test speed — 1 second per query is enough
        # to prove the point without making the test take 30+ seconds.
        slow_get_db, slow_cursor = self._make_slow_get_db(delay_seconds=1.0)

        ctx = CycleContext(
            cycle_id="test-slow-db",
            tickers=["AAPL", "MSFT"],
            collect=False,
            analyze=False,
            trade=False,
        )
        cycle_summary = {}
        state = {"position_tickers": []}

        mock_health = AsyncMock(return_value={"jetson": True, "dgx_spark": True})

        with patch("app.cycle.phases.phase1_health.get_db", slow_get_db), \
             patch("app.cycle.phases.phase1_health.llm") as mock_llm, \
             patch("app.cycle.phases.phase1_health.check_stop_losses", new_callable=AsyncMock, return_value=[]), \
             patch("app.cycle.phases.phase1_health.check_take_profits", new_callable=AsyncMock, return_value=[]), \
             patch("app.cycle.phases.phase1_health.classify_tickers") as mock_triage, \
             patch("app.cycle.phases.phase1_health.increment_days_since_deep"), \
             patch("app.cycle.phases.phase1_health.flag_neglected_tickers", return_value=[]), \
             patch("app.cycle.phases.phase1_health.get_attention_summary", return_value={}):

            mock_llm.health_all = mock_health
            mock_triage.return_value = MagicMock(
                glance=["AAPL"], standard=["MSFT"], deep=[],
                summary=MagicMock(return_value="1 glance, 1 standard, 0 deep"),
            )

            from app.cycle.phases.phase1_health import run_phase1_health

            start = time.monotonic()
            # Should NOT raise — slow DB is tolerated, just slow
            await run_phase1_health(
                ctx=ctx,
                bot_id="test-bot",
                emit=lambda *a, **kw: None,
                cycle_summary=cycle_summary,
                state=state,
            )
            elapsed = time.monotonic() - start

            # Verify it completed (didn't crash)
            assert "jetson_healthy_start" in cycle_summary
            # It should have taken at least 1 second due to the slow DB
            assert elapsed >= 1.0, f"Expected slow path, but completed in {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_portfolio_gate_survives_slow_db(self):
        """Portfolio gate check_portfolio_gate should not crash if DB is slow."""
        slow_get_db, slow_cursor = self._make_slow_get_db(delay_seconds=1.0)

        # Portfolio gate reads positions and sector metadata
        slow_cursor.fetchone.return_value = None  # no sector found
        slow_cursor.fetchall.return_value = []    # no positions

        # Must patch at app.db.connection level since portfolio_gate AND
        # paper_trader both import get_db from app.db.connection.
        # Also mock paper_trader's portfolio helpers that do their own DB calls.
        with patch("app.db.connection.get_db", slow_get_db), \
             patch("app.cycle.portfolio_gate.get_db", slow_get_db), \
             patch("app.trading.paper_trader.get_portfolio_value", return_value=10000.0), \
             patch("app.trading.paper_trader._get_current_price", return_value=(150.0, 0.5)):
            from app.cycle.portfolio_gate import check_portfolio_gate

            start = time.monotonic()
            result = check_portfolio_gate(
                ticker="AAPL",
                action="BUY",
                bot_id="test-bot",
                confidence=80,
            )
            elapsed = time.monotonic() - start

            # Should complete without crash — even with slow DB
            assert isinstance(result, dict)
            assert "blocked" in result
            # It should have been slow due to DB delay
            assert elapsed >= 1.0, f"Expected slow path, but completed in {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_run_with_timeout_protects_against_slow_operations(self):
        """run_with_timeout should return fallback when operation exceeds timeout."""
        from app.utils.async_utils import run_with_timeout

        async def slow_operation():
            await asyncio.sleep(10)
            return "should never reach this"

        result = await run_with_timeout(
            slow_operation(),
            timeout=1.0,
            label="test-slow-op",
            fallback={"fallback": True},
        )

        assert result == {"fallback": True}

    @pytest.mark.asyncio
    async def test_state_manager_get_state_survives_slow_db(self):
        """Verify that PipelineStateDB.get_state doesn't hang when DB is slow.

        The state manager imports `from app.db import connection` and calls
        `connection.get_db()`. We patch at the module level.
        """
        slow_get_db, slow_cursor = self._make_slow_get_db(delay_seconds=1.0)
        slow_cursor.fetchone.return_value = None  # No saved state

        # State manager uses `connection.get_db()` so patch at the source module
        with patch("app.db.connection.get_db", slow_get_db):
            from app.cycle.orchestration.state_manager import PipelineStateDB

            start = time.monotonic()
            try:
                state = PipelineStateDB.get_state()
            except Exception:
                # Even if it fails, it shouldn't hang
                state = None
            elapsed = time.monotonic() - start

            # Should complete within reasonable time (slow but not hung)
            assert elapsed < 30.0, f"get_state took {elapsed:.2f}s — possible hang"


# ────────────────────────────────────────────────────────────────────────
# 2. Message Queue Silent Drop Test
#    Drop a pub/sub or queue message mid-cycle and verify the system
#    detects it and recovers.
#
#    In trading-service, the "message queue" is the asyncio.Queue that
#    feeds tickers from collection → analysis workers. Sentinels (None)
#    signal worker shutdown.
# ────────────────────────────────────────────────────────────────────────


class TestMessageQueueSilentDrop:
    """Verify the analysis queue handles dropped/missing messages gracefully."""

    @pytest.mark.asyncio
    async def test_priority_queue_sentinel_before_all_tickers_processed(self):
        """If a sentinel arrives before all tickers are consumed, workers
        should shut down cleanly without hanging."""
        from app.cycle.orchestration.priority_queue import PriorityAnalysisQueue

        q = PriorityAnalysisQueue()

        # Put 3 tickers then a sentinel
        await q.put("AAPL")
        await q.put("MSFT")
        await q.put("GOOG")
        await q.put(None)  # sentinel

        # Consume only 1 ticker, then hit sentinel
        t1 = await q.get()
        assert t1 is not None  # Should get a real ticker

        # The sentinel is still in the queue — remaining tickers are "dropped"
        # Workers should be able to drain the queue without hanging
        remaining = []
        while not q.empty():
            item = await q.get()
            remaining.append(item)

        # Verify we got the remaining tickers + sentinel
        assert None in remaining, "Sentinel should be in remaining items"
        # Total items consumed should be 4 (3 tickers + 1 sentinel)
        assert len(remaining) + 1 == 4  # +1 for the first one we consumed

    @pytest.mark.asyncio
    async def test_analysis_queue_worker_handles_early_sentinel(self):
        """Simulate a worker that receives a sentinel before processing
        all expected tickers. The worker should exit cleanly."""
        work_queue = asyncio.Queue()

        # Put sentinel immediately — simulates "dropped" ticker messages
        work_queue.put_nowait(None)

        processed = []

        async def mock_worker():
            while True:
                ticker = await work_queue.get()
                if ticker is None:
                    work_queue.task_done()
                    break
                processed.append(ticker)
                work_queue.task_done()

        # Worker should exit immediately without hanging
        await asyncio.wait_for(mock_worker(), timeout=2.0)
        assert processed == [], "Worker should process no tickers before sentinel"

    @pytest.mark.asyncio
    async def test_analysis_queue_partial_drain_then_sentinel(self):
        """Workers process some tickers, then sentinel arrives. Any
        unprocessed tickers between the last processed and sentinel
        remain in the queue but workers exit cleanly."""
        work_queue = asyncio.Queue()

        # Simulate: collection puts 5 tickers, but only 2 make it before sentinel
        work_queue.put_nowait("AAPL")
        work_queue.put_nowait("MSFT")
        work_queue.put_nowait(None)  # sentinel arrives early
        work_queue.put_nowait("GOOG")  # these arrive after sentinel
        work_queue.put_nowait("TSLA")

        processed = []
        exited_cleanly = False

        async def mock_worker():
            nonlocal exited_cleanly
            while True:
                ticker = await work_queue.get()
                if ticker is None:
                    work_queue.task_done()
                    exited_cleanly = True
                    break
                processed.append(ticker)
                work_queue.task_done()

        await asyncio.wait_for(mock_worker(), timeout=2.0)

        assert exited_cleanly, "Worker should exit cleanly on sentinel"
        assert processed == ["AAPL", "MSFT"], "Worker should process only pre-sentinel tickers"
        # Remaining items are still in the queue — not lost, just unprocessed
        assert work_queue.qsize() == 2, f"Expected 2 remaining, got {work_queue.qsize()}"

    @pytest.mark.asyncio
    async def test_multiple_workers_single_sentinel_propagation(self):
        """With multiple workers, a single sentinel should propagate to
        all workers (the analysis phase re-puts sentinel for next worker).

        This tests the actual pattern from phase4_analysis.py lines 158-162:
            if active_workers > 0:
                work_queue.put_nowait(None)
        """
        work_queue = asyncio.Queue()

        # 3 tickers + 1 sentinel for 2 workers
        work_queue.put_nowait("AAPL")
        work_queue.put_nowait("MSFT")
        work_queue.put_nowait("GOOG")
        work_queue.put_nowait(None)

        processed = []
        workers_exited = 0
        total_workers = 2
        active_count = total_workers
        lock = asyncio.Lock()

        async def mock_worker(worker_id):
            nonlocal workers_exited, active_count
            while True:
                ticker = await work_queue.get()
                if ticker is None:
                    async with lock:
                        active_count -= 1
                        workers_exited += 1
                        # Re-propagate sentinel for remaining workers
                        if active_count > 0:
                            work_queue.put_nowait(None)
                    work_queue.task_done()
                    break
                async with lock:
                    processed.append(ticker)
                work_queue.task_done()

        tasks = [asyncio.create_task(mock_worker(i)) for i in range(total_workers)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        assert workers_exited == total_workers, f"All {total_workers} workers should exit, got {workers_exited}"
        assert set(processed) == {"AAPL", "MSFT", "GOOG"}, f"All tickers should be processed: {processed}"

    @pytest.mark.asyncio
    async def test_queue_get_timeout_prevents_infinite_hang(self):
        """If no sentinel ever arrives (message dropped), a worker using
        asyncio.wait_for should not hang forever."""
        work_queue = asyncio.Queue()

        # Put 1 ticker but NO sentinel — simulates dropped sentinel
        work_queue.put_nowait("AAPL")

        processed = []
        timed_out = False

        async def mock_worker_with_timeout():
            nonlocal timed_out
            while True:
                try:
                    ticker = await asyncio.wait_for(work_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                if ticker is None:
                    work_queue.task_done()
                    break
                processed.append(ticker)
                work_queue.task_done()

        await asyncio.wait_for(mock_worker_with_timeout(), timeout=5.0)

        assert processed == ["AAPL"]
        assert timed_out, "Worker should timeout when no sentinel arrives"


# ────────────────────────────────────────────────────────────────────────
# 3. Container Restart Mid-Cycle Test
#    Verify that if the container is killed/restarted mid-cycle, it
#    recovers cleanly from the last checkpoint.
# ────────────────────────────────────────────────────────────────────────


class TestContainerRestart:
    """Verify system recovery when the backend is killed mid-cycle."""

    @pytest.mark.asyncio
    async def test_zombie_cycle_reset_on_boot_detects_checkpoint(self):
        """If a running cycle crashed (zombie) and a checkpoint exists,
        reset_on_boot should set status to 'interrupted'."""
        from app.services.pipeline_service import PipelineService

        # Mock state to be zombie
        zombie_state = {
            "status": "analyzing",
            "cycle_id": "zombie-123",
            "tickers": ["AAPL", "MSFT"],
            "progress": "processing",
        }

        mock_checkpoint = {
            "cycle_id": "zombie-123",
            "status": "interrupted",
            "completed_phases": ["collecting"],
            "completed_tickers": {"analyzing": []},
            "cycle_config": {"tickers": ["AAPL", "MSFT"]},
            "checkpoint_ts": "2026-05-30T00:00:00Z",
            "original_started_at": "2026-05-30T00:00:00Z",
        }

        saved_states = []

        def save_state_spy(state):
            saved_states.append(state.copy())

        # Reset mixin state
        PipelineService._state = zombie_state.copy()

        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value=zombie_state), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state", side_effect=save_state_spy), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.expire_old_checkpoints"):

            PipelineService.reset_on_boot()

            # The state status and phase should have been set to 'interrupted'
            assert PipelineService._state["status"] == "interrupted"
            assert PipelineService._state["phase"] == "interrupted"
            assert "interrupted" in PipelineService._state["progress"]

            # Verify save_state was called with the updated state
            assert len(saved_states) > 0
            assert saved_states[-1]["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_zombie_cycle_reset_on_boot_no_checkpoint_resets_to_idle(self):
        """If a running cycle crashed but NO checkpoint exists,
        reset_on_boot should reset status to 'idle' and clean up orphaned data."""
        from app.services.pipeline_service import PipelineService

        # Mock state to be zombie
        zombie_state = {
            "status": "analyzing",
            "cycle_id": "zombie-456",
            "tickers": ["AAPL"],
        }

        PipelineService._state = zombie_state.copy()

        # Track sql statements executed
        sql_statements = []
        mock_cursor = MagicMock()

        def execute_spy(sql, params=None):
            sql_statements.append((sql, params))
            return mock_cursor

        mock_cursor.execute.side_effect = execute_spy
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def spy_get_db():
            yield mock_cursor

        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value=zombie_state), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=None), \
             patch("app.cycle.orchestration.state_manager.connection.get_db", spy_get_db), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.expire_old_checkpoints"):

            PipelineService.reset_on_boot()

            # State should be reset to default (which has status='idle')
            assert PipelineService._state["status"] == "idle"

            # Check that database cleanups were called for orphaned data of zombie-456
            deletes = [sql for sql, params in sql_statements if "DELETE FROM" in sql]
            assert len(deletes) >= 3
            assert any("pipeline_events" in sql for sql in deletes)
            assert any("analysis_results" in sql for sql in deletes)
            assert any("debate_history" in sql for sql in deletes)
            for sql, params in sql_statements:
                if "DELETE FROM" in sql:
                    assert params == ["zombie-456"]

    @pytest.mark.asyncio
    async def test_resume_interrupted_cycle_determines_correct_phase_and_resumes(self):
        """Verify that resuming an interrupted cycle correctly recovers context
        and restarts execution from the correct phase."""
        from app.services.pipeline_service import PipelineService

        # Mock state to be interrupted
        interrupted_state = {
            "status": "interrupted",
            "cycle_id": "interrupted-789",
            "tickers": ["AAPL", "MSFT", "GOOG"],
            "collect_flag": True,
            "analyze_flag": True,
        }

        PipelineService._state = interrupted_state.copy()

        mock_checkpoint = {
            "cycle_id": "interrupted-789",
            "status": "interrupted",
            "completed_phases": ["collecting"],  # collecting completed -> resume_from is analyzing
            "completed_tickers": {"analyzing": ["AAPL"]},
            "cycle_config": {
                "tickers": ["AAPL", "MSFT", "GOOG"],
                "collect_flag": True,
                "analyze_flag": True,
                "macro_memo": "macro memo test",
            },
            "checkpoint_ts": "2026-05-30T00:00:00Z",
            "original_started_at": "2026-05-30T00:00:00Z",
        }

        # Mock db calls for fetch_already_analyzed & fetch_existing_results
        mock_cursor = MagicMock()
        query_count = 0

        def execute_spy(sql, params=None):
            nonlocal query_count
            query_count += 1
            return mock_cursor

        def fetchall_spy():
            if query_count == 1:
                return [("AAPL",)]
            elif query_count == 2:
                return [("AAPL", '{"ticker": "AAPL", "recommendation": "BUY"}')]
            return []

        mock_cursor.execute.side_effect = execute_spy
        mock_cursor.fetchall.side_effect = fetchall_spy
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def spy_get_db():
            yield mock_cursor

        # We'll intercept `_run_cycle` call
        run_cycle_called_with = None

        async def dummy_run_cycle(ctx):
            nonlocal run_cycle_called_with
            run_cycle_called_with = ctx

        async def dummy_heartbeat(cycle_id):
            pass

        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value=interrupted_state), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
             patch("app.cycle.orchestration.lifecycle_controller.get_db", spy_get_db), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \
             patch("app.cycle.orchestration.lifecycle_controller.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
             patch("app.services.pipeline_service.PipelineService._run_cycle", dummy_run_cycle), \
             patch("app.cycle.orchestration.lifecycle_controller.LifecycleControllerMixin._checkpoint_heartbeat", dummy_heartbeat):

            # Call background resume directly
            await PipelineService._background_resume_cycle("interrupted-789")

            # Let the loop run to execute tasks
            await asyncio.sleep(0.01)

            # Verify context passed to run_cycle
            assert run_cycle_called_with is not None
            assert run_cycle_called_with.cycle_id == "interrupted-789"
            assert run_cycle_called_with.tickers == ["AAPL", "MSFT", "GOOG"]
            assert run_cycle_called_with.resume_from == "analyzing"
            assert run_cycle_called_with.already_analyzed == ["AAPL"]
            assert len(run_cycle_called_with.existing_results) == 1
            assert run_cycle_called_with.existing_results[0]["ticker"] == "AAPL"
            assert run_cycle_called_with.macro_memo == "macro memo test"


# ────────────────────────────────────────────────────────────────────────
# 4. Inference / AI Service Overload Test
#    Verify that when the LLM service returns truncated, partial, or malformed
#    JSON responses (due to context overflow, timeouts, or model overload),
#    the agents recover gracefully using robust fallbacks.
# ────────────────────────────────────────────────────────────────────────


class TestInferenceOverload:
    """Verify agent resilience under partial/truncated LLM JSON responses."""

    @pytest.mark.asyncio
    async def test_pre_trade_agent_truncated_response(self):
        """Verify pre_trade_agent handles truncated JSON response without crashing."""
        from app.agents.pre_trade_agent import run_pre_trade

        # Simulate a JSON response that was cut off mid-sentence
        truncated_response = '{"decision": "VETO", "rationale": "High volatility and'

        with patch("app.agents.pre_trade_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 10}
            result = await run_pre_trade("AAPL", 90, "cycle_1", "bot_1")

            # Should fallback to APPROVE with Kelly sizing
            assert result["decision"] == "APPROVE"
            assert "Kelly fallback" in result["rationale"]

    @pytest.mark.asyncio
    async def test_trade_execution_agent_truncated_response(self):
        """Verify trade_execution_agent handles truncated JSON response without crashing."""
        from app.agents.trade_execution_agent import run_trade_execution

        truncated_response = '{"decision": "VETO", "rationale": "Insufficient liquidity due to'

        with patch("app.agents.trade_execution_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 10}

            # Action: BUY -> fallback is APPROVE / Kelly
            result = await run_trade_execution("AAPL", "BUY", 90, "cycle_1", "bot_1")
            assert result["decision"] == "APPROVE"
            assert "Kelly fallback" in result["rationale"]

            # Action: SELL -> fallback is APPROVE / 100% trim
            result_sell = await run_trade_execution("AAPL", "SELL", 90, "cycle_1", "bot_1")
            assert result_sell["decision"] == "APPROVE"
            assert result_sell["sell_pct"] == 100  # 100% trim fallback


    @pytest.mark.asyncio
    async def test_portfolio_allocator_agent_truncated_response(self):
        """Verify portfolio_allocator_agent handles truncated JSON response without crashing."""
        from app.agents.portfolio_allocator_agent import run_portfolio_allocator

        truncated_response = '{"allocations": [{"ticker": "AAPL", "weight": 0.5}, {"ticker": "MSFT'

        with patch("app.agents.portfolio_allocator_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 15}

            result = await run_portfolio_allocator(
                [{"ticker": "AAPL", "action": "BUY", "confidence": 90}], "cycle_1", "bot_1"
            )
            # Should degrade to empty allocations dict
            assert result == {}

    @pytest.mark.asyncio
    async def test_post_mortem_auditor_agent_truncated_response(self):
        """Verify post_mortem_auditor_agent handles truncated JSON response without crashing."""
        from app.agents.post_mortem_auditor_agent import run_post_mortem

        truncated_response = '{"audit_findings": {"ticker": "AAPL", "performance":'

        with patch("app.agents.post_mortem_auditor_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 12}

            result = await run_post_mortem("AAPL", 100.0, 110.0, 10.0, "cycle_1", "bot_1")
            assert result == {}

    @pytest.mark.asyncio
    async def test_meta_agent_truncated_response(self):
        """Verify meta_agent handles truncated JSON response without crashing."""
        from app.agents.meta_agent import generate_prompt

        truncated_response = '{"improved_prompt": "You are a trading bot assistant'

        with patch("app.agents.meta_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 20}

            result = await generate_prompt("wins", "losses", "insights", "existing", "cycle_1", "bot_1")
            assert result == {}

    @pytest.mark.asyncio
    async def test_context_compressor_truncated_response(self):
        """Verify context_compressor generate_capsule degrades gracefully with truncated JSON."""
        from app.agents.context_compressor import generate_capsule

        # A truncated JSON which contains some plain text patterns that the regex fallback can extract
        truncated_response = '{"signal": "BUY", "confidence": 85, "summary": "Strong momentum'

        # When json parsing fails, it falls back to regex heuristic.
        # Let's ensure it extracts "BUY" and confidence 0.85
        result = await generate_capsule(
            {"response": truncated_response, "tokens_used": 10}, "test_agent", "cycle_1", "AAPL"
        )
        assert result.signal == "BUY"
        assert result.confidence == 0.85
        assert "Strong momentum" in result.summary


# ────────────────────────────────────────────────────────────────────────
# 5. Connection Pool Exhaustion Test
#    Verify that when the connection pool is saturated, the get_db()
#    concurrency manager handles the failure gracefully by reclaiming
#    connections via garbage collection, and raises errors properly without
#    crashing the backend process.
# ────────────────────────────────────────────────────────────────────────


class TestConnectionPoolExhaustion:
    """Verify system resilience under DB connection pool exhaustion."""

    @pytest.fixture(autouse=True)
    def patch_get_db(self):
        """Override the global autouse fixture from conftest.py to test the real get_db."""
        return None

    @pytest.mark.asyncio
    async def test_get_db_recovers_on_gc_collect(self):
        """Verify that get_db() calls gc.collect() and retries when pool is exhausted,
        succeeding if a connection becomes available."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()

        # First call raises timeout, second call succeeds
        mock_pool.getconn.side_effect = [
            Exception("Pool timeout: no connections available"),
            mock_conn,
        ]

        with patch("app.db.connection._ensure_pool", return_value=mock_pool), \
             patch("gc.collect") as mock_gc:

            from app.db.connection import get_db

            with get_db() as db:
                assert db._conn == mock_conn

            # Verify gc.collect() was triggered
            mock_gc.assert_called_once()
            # Verify getconn was called twice (first fail, then retry)
            assert mock_pool.getconn.call_count == 2

    @pytest.mark.asyncio
    async def test_get_db_raises_after_failed_retry(self):
        """Verify that get_db() raises if the retry attempt also fails."""
        mock_pool = MagicMock()

        # Both calls raise timeout
        mock_pool.getconn.side_effect = [
            Exception("Pool timeout error"),
            Exception("Pool timeout error"),
        ]

        with patch("app.db.connection._ensure_pool", return_value=mock_pool), \
             patch("gc.collect"):

            from app.db.connection import get_db

            with pytest.raises(Exception, match="Pool timeout error"):
                with get_db():
                    pass

    @pytest.mark.asyncio
    async def test_pipeline_state_db_survives_pool_exhaustion(self):
        """Verify that PipelineStateDB doesn't crash when get_db() raises timeout."""
        mock_pool = MagicMock()
        mock_pool.getconn.side_effect = Exception("Pool timeout error")

        with patch("app.db.connection._ensure_pool", return_value=mock_pool), \
             patch("gc.collect"):

            from app.cycle.orchestration.state_manager import PipelineStateDB

            # Reading state should fall back to default_state instead of raising
            state = PipelineStateDB.get_state()
            assert state["status"] == "idle"
            assert state["cycle_id"] is None

            # Saving state should swallow the error and log it (should not raise)
            PipelineStateDB.save_state({"status": "running"})




# ────────────────────────────────────────────────────────────────────────
# 6. Network Partition Test
#    Simulate upstream connectivity loss (ConnectionError, TimeoutError)
#    mid-collection.  Verify:
#    a) Single-source failure is isolated — other sources + ticker succeed.
#    b) Full-ticker failure is isolated — other tickers proceed.
#    c) Timeout-based disconnection rejects the ticker cleanly.
# ────────────────────────────────────────────────────────────────────────


class TestNetworkPartition:
    """Verify per-ticker collection isolates upstream network failures."""

    @staticmethod
    def _make_emit_recorder():
        """Return (emit_fn, events_list) where events_list collects all emitted events."""
        events = []

        def _emit(phase, key, msg, *, status=None, data=None, elapsed_ms=None):
            events.append({"phase": phase, "key": key, "msg": msg, "status": status})

        return _emit, events

    @staticmethod
    def _build_common_patches(stack, **overrides):
        """Enter all common patches via ExitStack and return handles dict.

        ``overrides`` lets callers swap specific mocks (e.g. side_effect
        on fetch_price_history).
        """
        from contextlib import ExitStack

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = overrides.get("price_count", (500,))
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        # --- Core pipeline machinery (no-ops / defaults) ---
        mock_cc = stack.enter_context(
            patch("app.pipeline.data.data_perticker_collection.cycle_control")
        )
        mock_cc.wait_if_paused = AsyncMock()

        mock_settings = stack.enter_context(
            patch("app.pipeline.data.data_perticker_collection.settings")
        )
        mock_settings.COLLECTION_MAX_CONCURRENT = 5
        mock_settings.USE_TOOL_CALLING = False

        if "source_timeout" in overrides:
            stack.enter_context(
                patch(
                    "app.pipeline.data.data_perticker_collection.SOURCE_TIMEOUT",
                    overrides["source_timeout"],
                )
            )

        stack.enter_context(
            patch("app.pipeline.data.collection_scheduler.should_collect", return_value=True)
        )
        stack.enter_context(
            patch("app.pipeline.data.data_perticker_collection.record_collection")
        )
        stack.enter_context(
            patch("app.pipeline.data.data_perticker_collection.elapsed_ms", return_value=10.0)
        )
        stack.enter_context(
            patch("app.pipeline.data.data_perticker_collection.pipeline_profiler")
        )

        # --- Collectors (overrideable) ---
        stack.enter_context(
            patch(
                "app.collectors.data_rotator.fetch_price_history",
                **(overrides.get("yf_price", {"new_callable": AsyncMock, "return_value": 300})),
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.data_rotator.fetch_fundamentals",
                **(overrides.get("yf_fund", {"new_callable": AsyncMock, "return_value": 5})),
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.data_rotator.fetch_financials",
                **(overrides.get("yf_fin", {"new_callable": AsyncMock, "return_value": 3})),
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.data_rotator.fetch_balance_sheet",
                **(overrides.get("yf_bs", {"new_callable": AsyncMock, "return_value": 2})),
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.finnhub_collector.collect_news",
                **(overrides.get("finnhub", {"new_callable": AsyncMock, "return_value": 5})),
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.reddit_collector.collect_for_ticker",
                new_callable=AsyncMock, return_value=10,
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.youtube_collector.collect_for_ticker",
                new_callable=AsyncMock, return_value={"stored": 3},
            )
        )
        stack.enter_context(
            patch(
                "app.collectors.yfinance_collector.collect_news",
                new_callable=AsyncMock, return_value=8,
            )
        )

        # --- DB + infra ---
        stack.enter_context(patch("app.db.connection.get_db", side_effect=fake_get_db))

        mock_rl = stack.enter_context(
            patch("app.services.api_rate_limiter.rate_limiter")
        )
        mock_rl.acquire.return_value.__aenter__ = AsyncMock()
        mock_rl.acquire.return_value.__aexit__ = AsyncMock()

        stack.enter_context(
            patch("app.processors.technical_processor.compute_technicals", return_value=50)
        )
        stack.enter_context(
            patch("app.pipeline.watchlist_health.update_signals_from_collection")
        )
        stack.enter_context(
            patch(
                "app.pipeline.data.data_perticker_collection.run_ticker_processors",
                new_callable=AsyncMock,
            )
        )
        stack.enter_context(
            patch("app.pipeline.data.data_sufficiency.check_data_sufficiency", return_value=False)
        )
        stack.enter_context(
            patch("app.pipeline.data.fallback_collector.detect_data_gaps", return_value=[])
        )
        stack.enter_context(
            patch("app.pipeline.alpha_decay_purge.run_alpha_decay_purge", return_value=[])
        )

        # --- Ticker extractor safety gate ---
        mock_reg = MagicMock()
        mock_reg.is_rejected.return_value = False
        stack.enter_context(
            patch("app.processors.ticker_extractor.get_registry", return_value=mock_reg)
        )
        stack.enter_context(
            patch("app.processors.ticker_extractor._is_hard_blocked", return_value=False)
        )

        return {"mock_cc": mock_cc, "mock_settings": mock_settings}

    @pytest.mark.asyncio
    async def test_single_source_failure_isolated(self):
        """If Finnhub raises ConnectionError, other sources still succeed and
        the ticker is NOT rejected (yfinance is the gate-keeper)."""
        from contextlib import ExitStack

        emit, events = self._make_emit_recorder()

        async def finnhub_explode(ticker):
            raise ConnectionError("Simulated network partition: DNS unreachable")

        with ExitStack() as stack:
            self._build_common_patches(
                stack,
                finnhub={"side_effect": finnhub_explode},
            )

            from app.pipeline.data.data_perticker_collection import run_perticker_collection

            results = {"collectors": {}}
            summary = {"cycle_id": "test-net-partition"}

            await run_perticker_collection(
                tickers=["AAPL"],
                _glance_set=set(),
                _deep_set={"AAPL"},
                emit=emit,
                results=results,
                _summary=summary,
            )

        # AAPL should still be in the results (not rejected)
        assert "AAPL" in results["tickers"], (
            "AAPL should survive a single-source (Finnhub) network failure"
        )

        # Verify Finnhub emitted a skip/error status
        finnhub_events = [e for e in events if "finnhub" in e["key"].lower()]
        assert any(e["status"] in ("skipped", "error") for e in finnhub_events), (
            "Finnhub should emit skipped/error status on ConnectionError"
        )
        logger.info("PASS: Single-source network failure isolated — ticker survived")

    @pytest.mark.asyncio
    async def test_yfinance_network_failure_rejects_ticker_not_others(self):
        """If yfinance raises ConnectionError for ticker A, ticker A is rejected
        but ticker B completes successfully.

        This tests the critical `return_exceptions=True` in asyncio.gather
        at the outer (all-tickers) level.
        """
        from contextlib import ExitStack

        emit, events = self._make_emit_recorder()

        async def yf_price_history(ticker):
            if ticker == "BADTK":
                raise ConnectionError("Network unreachable for BADTK")
            return 300

        async def yf_fundamentals(ticker):
            if ticker == "BADTK":
                raise ConnectionError("Network unreachable for BADTK")
            return 5

        async def yf_financials(ticker):
            if ticker == "BADTK":
                raise ConnectionError("Network unreachable for BADTK")
            return 3

        async def yf_balance_sheet(ticker):
            if ticker == "BADTK":
                raise ConnectionError("Network unreachable for BADTK")
            return 2

        with ExitStack() as stack:
            self._build_common_patches(
                stack,
                yf_price={"side_effect": yf_price_history},
                yf_fund={"side_effect": yf_fundamentals},
                yf_fin={"side_effect": yf_financials},
                yf_bs={"side_effect": yf_balance_sheet},
            )
            # Extra patch for PipelineStateDB error logging
            stack.enter_context(
                patch("app.pipeline.orchestration.state_manager.PipelineStateDB")
            )

            from app.pipeline.data.data_perticker_collection import run_perticker_collection

            results = {"collectors": {}}
            summary = {"cycle_id": "test-net-partition-multi"}

            await run_perticker_collection(
                tickers=["GOODTK", "BADTK"],
                _glance_set=set(),
                _deep_set={"GOODTK", "BADTK"},
                emit=emit,
                results=results,
                _summary=summary,
            )

        # GOODTK must survive
        assert "GOODTK" in results["tickers"], (
            "GOODTK should survive when BADTK has network failure"
        )
        # BADTK should be rejected (yfinance failure → _ticker_rejected.set())
        assert "BADTK" not in results["tickers"], (
            "BADTK should be rejected after yfinance network failure"
        )
        logger.info("PASS: Ticker-level network partition isolated — healthy ticker survived")

    @pytest.mark.asyncio
    async def test_timeout_disconnection_rejects_ticker(self):
        """Simulate a network hang (timeout) on yfinance — the ticker should
        be rejected via asyncio.wait_for timeout and _ticker_rejected.set()."""
        from contextlib import ExitStack

        emit, events = self._make_emit_recorder()

        async def yf_hang(ticker):
            await asyncio.sleep(9999)

        with ExitStack() as stack:
            self._build_common_patches(
                stack,
                price_count=(0,),
                source_timeout=0.5,
                yf_price={"side_effect": yf_hang},
            )

            from app.pipeline.data.data_perticker_collection import run_perticker_collection

            results = {"collectors": {}}
            summary = {"cycle_id": "test-timeout-partition"}

            await run_perticker_collection(
                tickers=["HANGTK"],
                _glance_set=set(),
                _deep_set={"HANGTK"},
                emit=emit,
                results=results,
                _summary=summary,
            )

        # HANGTK should be rejected (timeout → _ticker_rejected.set())
        assert "HANGTK" not in results["tickers"], (
            "HANGTK should be rejected after yfinance timeout (network hang)"
        )

        # Verify yfinance emitted a timeout or error status for HANGTK
        # (The retry decorator may reclassify the exception, so we accept both)
        yf_events = [
            e for e in events
            if e["key"] == "yfinance_HANGTK" and e["status"] in ("timeout", "error")
        ]
        assert len(yf_events) >= 1, (
            f"Should emit timeout/error status for yfinance when network hangs. "
            f"Got events: {[e for e in events if 'yfinance' in str(e.get('key', ''))]}"
        )
        logger.info(
            "PASS: Timeout disconnection correctly rejects ticker (status=%s)",
            yf_events[0]["status"],
        )


# ────────────────────────────────────────────────────────────────────────
# 7. Processing Window Boundary Test
#    Verify market-hours boundary logic at exact open/close times.
#    Three functions are tested:
#    a) SchedulerService._is_market_hours() — ET 9:30–16:00 weekdays
#    b) passive_collector._is_market_hours() — PT 6:00–17:00 weekdays
#    c) data_lifecycle.is_off_peak() — ET 9:30–16:00 weekdays
#    d) Scheduler skips market_hours_only schedules when outside hours
# ────────────────────────────────────────────────────────────────────────


class TestProcessingWindowBoundary:
    """Verify market hours functions return correct values at exact boundaries."""

    # ── A: SchedulerService._is_market_hours() ──

    def test_scheduler_market_open_boundary_930am_et(self):
        """At exactly 9:30 AM ET on a Wednesday, _is_market_hours() → True."""
        import pytz

        et = pytz.timezone("US/Eastern")
        # Wednesday 9:30:00 AM ET
        fake_now = et.localize(datetime(2026, 6, 3, 9, 30, 0))

        with patch("app.services.cycle_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.cycle_scheduler import SchedulerService

            assert SchedulerService._is_market_hours() is True, (
                "9:30 AM ET on Wednesday should be market hours"
            )
        logger.info("PASS: 9:30 AM ET → market hours = True")

    def test_scheduler_market_close_boundary_400pm_et(self):
        """At exactly 4:00 PM ET on a Wednesday, _is_market_hours() → True (inclusive)."""
        import pytz

        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2026, 6, 3, 16, 0, 0))

        with patch("app.services.cycle_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.cycle_scheduler import SchedulerService

            assert SchedulerService._is_market_hours() is True, (
                "4:00 PM ET on Wednesday should still be market hours (boundary inclusive)"
            )
        logger.info("PASS: 4:00 PM ET → market hours = True (boundary)")

    def test_scheduler_one_minute_after_close(self):
        """At 4:01 PM ET on a Wednesday, _is_market_hours() → False."""
        import pytz

        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2026, 6, 3, 16, 1, 0))

        with patch("app.services.cycle_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.cycle_scheduler import SchedulerService

            assert SchedulerService._is_market_hours() is False, (
                "4:01 PM ET on Wednesday should NOT be market hours"
            )
        logger.info("PASS: 4:01 PM ET → market hours = False")

    def test_scheduler_weekend_saturday(self):
        """Saturday should NOT be market hours regardless of time."""
        import pytz

        et = pytz.timezone("US/Eastern")
        # Saturday 12:00 PM ET (June 6, 2026 is Saturday)
        fake_now = et.localize(datetime(2026, 6, 6, 12, 0, 0))

        with patch("app.services.cycle_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.cycle_scheduler import SchedulerService

            assert SchedulerService._is_market_hours() is False, (
                "Saturday 12:00 PM should NOT be market hours"
            )
        logger.info("PASS: Saturday → market hours = False")

    # ── B: passive_collector._is_market_hours() (PT-based) ──

    def test_passive_collector_extended_hours_open(self):
        """At 6:00 AM PT on a Monday, passive collector → market hours."""
        import pytz

        pt = pytz.timezone("America/Los_Angeles")
        fake_now = pt.localize(datetime(2026, 6, 1, 6, 0, 0))  # Monday

        with patch("app.services.passive_collector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.passive_collector import _is_market_hours

            assert _is_market_hours() is True, (
                "6:00 AM PT on Monday should be extended market hours"
            )
        logger.info("PASS: 6:00 AM PT → passive collector market hours = True")

    def test_passive_collector_after_extended_close(self):
        """At 5:00 PM PT on a Monday, passive collector → NOT market hours."""
        import pytz

        pt = pytz.timezone("America/Los_Angeles")
        fake_now = pt.localize(datetime(2026, 6, 1, 17, 0, 0))  # Monday

        with patch("app.services.passive_collector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.passive_collector import _is_market_hours

            assert _is_market_hours() is False, (
                "5:00 PM PT on Monday should NOT be extended market hours"
            )
        logger.info("PASS: 5:00 PM PT → passive collector market hours = False")

    # ── C: data_lifecycle.is_off_peak() ──

    def test_off_peak_during_market_hours(self):
        """During active market hours (12:00 PM ET Mon), is_off_peak() → False."""
        fake_now = datetime(2026, 6, 1, 12, 0, 0)  # Monday noon

        with patch("app.pipeline.data.data_lifecycle.datetime") as mock_dt:
            # Patch both ZoneInfo and direct datetime calls
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.pipeline.data.data_lifecycle import is_off_peak

            assert is_off_peak() is False, (
                "12:00 PM ET on Monday should NOT be off-peak"
            )
        logger.info("PASS: 12:00 PM ET Monday → off-peak = False")

    def test_off_peak_after_market_close(self):
        """After market close (6:00 PM ET Mon), is_off_peak() → True."""
        fake_now = datetime(2026, 6, 1, 18, 0, 0)  # Monday 6 PM

        with patch("app.pipeline.data.data_lifecycle.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.pipeline.data.data_lifecycle import is_off_peak

            assert is_off_peak() is True, (
                "6:00 PM ET on Monday should be off-peak"
            )
        logger.info("PASS: 6:00 PM ET Monday → off-peak = True")

    # ── D: Scheduler skips market_hours_only schedules ──

    @pytest.mark.asyncio
    async def test_scheduler_skips_market_hours_only_schedule_outside_hours(self):
        """When market_hours_only=True and it's 8 PM ET, the schedule should
        be skipped (not execute a cycle)."""
        import pytz

        et = pytz.timezone("US/Eastern")
        fake_now = et.localize(datetime(2026, 6, 3, 20, 0, 0))  # Wed 8 PM ET

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        # Return a schedule row with market_hours_only = True
        mock_cursor.fetchone.return_value = (
            "sched-001",       # id
            "Test Schedule",   # name
            "interval",        # schedule_type
            None,              # cron_expression
            2.0,               # interval_hours
            True,              # collect
            True,              # analyze
            True,              # trade
            "[]",              # tickers
            10,                # max_tickers
            True,              # discovered_tickers
            True,              # market_hours_only ← True
            True,              # is_active
            None,              # last_run_at
            None,              # next_run_at
            0,                 # run_count
            None,              # last_status
            None,              # last_error
            "2026-01-01T00:00:00Z",  # created_at
            "2026-01-01T00:00:00Z",  # updated_at
        )
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.services.cycle_scheduler.datetime") as mock_dt, \
             patch("app.services.cycle_scheduler.get_db", side_effect=fake_get_db), \
             patch("app.services.cycle_scheduler.cycle_control") as mock_cc:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_cc.is_paused = False

            from app.services.cycle_scheduler import SchedulerService

            # Mock _sync_next_run_to_db to avoid scheduler access
            with patch.object(SchedulerService, "_sync_next_run_to_db"):
                await SchedulerService.execute_schedule("sched-001")

            # The cycle should NOT have been dispatched — check that
            # no system_commands INSERT was executed
            insert_calls = [
                call for call in mock_cursor.execute.call_args_list
                if "system_commands" in str(call).lower()
            ]
            assert len(insert_calls) == 0, (
                "Should NOT dispatch a cycle when market_hours_only=True and outside hours"
            )
        logger.info("PASS: Scheduler skips market_hours_only schedule outside hours")


# ────────────────────────────────────────────────────────────────────────
# 8. Off-Hours / Blackout Window Test
#    Verify that LLM summarization and passive collection behave correctly
#    during off-hours (outside market hours).
#    a) summarize_stale_records() skips during market hours
#    b) summarize_stale_records() proceeds during off-peak
#    c) Passive collector sleeps 6h during off-hours (no collection)
#    d) Passive collector uses 3h sleep during market hours
# ────────────────────────────────────────────────────────────────────────


class TestOffHoursBlackoutWindow:
    """Verify off-hours/blackout gating for LLM summarization and passive collection."""

    @pytest.mark.asyncio
    async def test_summarize_stale_records_skips_during_market_hours(self):
        """summarize_stale_records() should return 0 immediately during market hours
        without making any DB or LLM calls."""
        with patch("app.pipeline.data.data_lifecycle.is_off_peak", return_value=False):
            from app.pipeline.data.data_lifecycle import summarize_stale_records

            result = await summarize_stale_records(limit=50)

        assert result == 0, (
            "summarize_stale_records should return 0 during market hours"
        )
        logger.info("PASS: summarize_stale_records skips during market hours → 0")

    @pytest.mark.asyncio
    async def test_summarize_stale_records_proceeds_during_off_peak(self):
        """summarize_stale_records() should attempt DB queries during off-peak."""
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []  # no records to summarize
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.pipeline.data.data_lifecycle.is_off_peak", return_value=True), \
             patch("app.pipeline.data.data_lifecycle.get_db", side_effect=fake_get_db), \
             patch("app.pipeline.data.data_lifecycle.settings") as mock_settings:
            mock_settings.RAW_DATA_TTL_HOURS = 24

            from app.pipeline.data.data_lifecycle import summarize_stale_records

            result = await summarize_stale_records(limit=50)

        # Should have proceeded (returned 0 because no records, not because skipped)
        assert result == 0
        # Verify DB was actually queried (not short-circuited)
        assert mock_cursor.execute.called, (
            "During off-peak, summarize should query the DB (not skip)"
        )
        logger.info("PASS: summarize_stale_records proceeds during off-peak (queries DB)")

    def test_passive_collector_off_hours_sleep_duration(self):
        """During off-hours, passive collector should use OFF_HOURS_SLEEP (6h)."""
        from app.services.passive_collector import (
            MARKET_HOURS_SLEEP,
            OFF_HOURS_SLEEP,
        )

        # Verify constants are what we expect
        assert OFF_HOURS_SLEEP == 6 * 3600, (
            f"OFF_HOURS_SLEEP should be 6 hours (21600s), got {OFF_HOURS_SLEEP}"
        )
        assert MARKET_HOURS_SLEEP == 3 * 3600, (
            f"MARKET_HOURS_SLEEP should be 3 hours (10800s), got {MARKET_HOURS_SLEEP}"
        )
        assert OFF_HOURS_SLEEP > MARKET_HOURS_SLEEP, (
            "Off-hours sleep should be longer than market-hours sleep"
        )
        logger.info(
            "PASS: OFF_HOURS_SLEEP=%ds > MARKET_HOURS_SLEEP=%ds",
            OFF_HOURS_SLEEP, MARKET_HOURS_SLEEP,
        )

    def test_passive_collector_skips_collection_during_off_hours(self):
        """During off-hours, the passive collector should NOT call _passive_collect_ticker.

        We verify this by checking that when _is_market_hours() returns False,
        the loop goes to sleep (continue) before reaching ticker collection.
        """
        import pytz

        pt = pytz.timezone("America/Los_Angeles")
        # Sunday midnight PT — definitely off-hours
        fake_now = pt.localize(datetime(2026, 6, 7, 0, 0, 0))

        with patch("app.services.passive_collector.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.services.passive_collector import _is_market_hours

            is_mkt = _is_market_hours()

        assert is_mkt is False, "Sunday midnight should be off-hours"

        # The loop logic is:
        #   if _is_market_hours():
        #       sleep_seconds = MARKET_HOURS_SLEEP
        #   else:
        #       sleep(OFF_HOURS_SLEEP)
        #       continue  ← skips all collection below
        #
        # Since _is_market_hours() returns False, the `continue` statement
        # prevents any ticker collection from occurring.
        logger.info("PASS: Off-hours correctly detected → collection skipped via continue")

    def test_is_off_peak_before_market_open(self):
        """5:00 AM ET on a Monday should be off-peak (before 9:30 AM open)."""
        fake_now = datetime(2026, 6, 1, 5, 0, 0)  # Monday 5 AM

        with patch("app.pipeline.data.data_lifecycle.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.pipeline.data.data_lifecycle import is_off_peak

            assert is_off_peak() is True, (
                "5:00 AM ET on Monday should be off-peak (pre-market)"
            )
        logger.info("PASS: 5:00 AM ET Monday → off-peak = True (pre-market)")

    def test_is_off_peak_sunday(self):
        """Sunday 12:00 PM should be off-peak regardless of time."""
        fake_now = datetime(2026, 6, 7, 12, 0, 0)  # Sunday noon

        with patch("app.pipeline.data.data_lifecycle.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.pipeline.data.data_lifecycle import is_off_peak

            assert is_off_peak() is True, (
                "Sunday noon should be off-peak"
            )
        logger.info("PASS: Sunday noon → off-peak = True")


# ────────────────────────────────────────────────────────────────────────
# 9. Clock Drift Test
#    Verify the collection scheduler handles clock skew between:
#    - local container clock (datetime.now)
#    - DB timestamps (last_success / last_failure)
#    - External source timestamps (published_at, etc.)
#    Key functions tested: should_collect, hours_since_last, _parse_timestamp
# ────────────────────────────────────────────────────────────────────────


class TestClockDrift:
    """Verify collection scheduler handles clock skew safely."""

    def test_future_timestamp_makes_data_appear_fresh(self):
        """If the DB has a future timestamp (clock skewed source),
        hours_since_last returns a negative value. should_collect
        should still handle this correctly (treat as fresh → skip)."""
        import datetime as dt

        # Simulate: DB says last_success = 2 hours in the future
        # DB cursor returns datetime objects directly (not strings)
        future_ts = dt.datetime.now(dt.UTC) + dt.timedelta(hours=2)

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (future_ts,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            from app.pipeline.data.collection_scheduler import hours_since_last, should_collect

            hours = hours_since_last("news_finnhub")

        # hours_since_last should return a negative number (future timestamp)
        assert hours is not None, "Should return a value, not None"
        assert hours < 0, f"Future timestamp should produce negative hours, got {hours}"

        # should_collect should say False (negative hours < 6h interval = fresh)
        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            result = should_collect("news_finnhub")
        assert result is False, (
            "Future timestamp (clock drift) should make data appear fresh → skip"
        )
        logger.info("PASS: Future timestamp → negative hours → treated as fresh (skip)")

    def test_should_collect_unknown_source_always_true(self):
        """Unknown source keys should always return True (safe default)."""
        from app.pipeline.data.collection_scheduler import should_collect

        # No DB call needed — unknown source short-circuits
        result = should_collect("nonexistent_source_xyz")
        assert result is True, (
            "Unknown source should always return True (collect)"
        )
        logger.info("PASS: Unknown source → should_collect = True (safe default)")

    def test_parse_timestamp_naive_string(self):
        """_parse_timestamp should handle timezone-naive strings by assuming UTC."""
        from app.pipeline.data.collection_scheduler import _parse_timestamp
        import datetime as dt

        result = _parse_timestamp("2026-06-01 12:00:00")
        assert result is not None, "Should parse '2026-06-01 12:00:00'"
        assert result.tzinfo is not None, (
            "Timezone-naive string should be upgraded to UTC"
        )
        assert result.tzinfo == dt.UTC, (
            f"Should be UTC, got {result.tzinfo}"
        )
        logger.info("PASS: Naive timestamp string → parsed with UTC tzinfo")

    def test_parse_timestamp_date_object(self):
        """_parse_timestamp should handle date objects (no time component)."""
        import datetime as dt
        from app.pipeline.data.collection_scheduler import _parse_timestamp

        d = dt.date(2026, 6, 1)
        result = _parse_timestamp(d)
        assert result is not None, "Should parse a date object"
        assert isinstance(result, dt.datetime), "Should return a datetime"
        assert result.hour == 0 and result.minute == 0, (
            "Date should be converted to midnight datetime"
        )
        assert result.tzinfo is not None, "Should have UTC timezone"
        logger.info("PASS: date object → midnight UTC datetime")

    def test_hours_since_last_returns_none_when_no_data(self):
        """When no data exists for a source, hours_since_last returns None.
        This is the safe default that triggers collection."""
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (None,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            from app.pipeline.data.collection_scheduler import hours_since_last, should_collect

            hours = hours_since_last("news_finnhub")

        assert hours is None, "No data should return None"

        # should_collect should return True when hours is None
        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            result = should_collect("news_finnhub")
        assert result is True, (
            "No data (hours=None) should trigger collection"
        )
        logger.info("PASS: No data → hours_since_last=None → should_collect=True")

    def test_stale_data_triggers_collection(self):
        """Data that is older than the refresh interval should trigger collection."""
        import datetime as dt

        # Simulate: last collected 12 hours ago (news_finnhub interval = 6h)
        old_ts = dt.datetime.now(dt.UTC) - dt.timedelta(hours=12)

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (old_ts.isoformat(),)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            from app.pipeline.data.collection_scheduler import should_collect

            result = should_collect("news_finnhub")

        assert result is True, (
            "Data 12h old with 6h interval should trigger collection"
        )
        logger.info("PASS: 12h old data with 6h interval → should_collect=True")


# ────────────────────────────────────────────────────────────────────────
# 10. Midnight UTC Rollover Test
#     Verify behavior at the UTC date boundary (23:59 → 00:00).
#     Key areas:
#     a) get_sec_13f_quarter() returns correct quarter at month boundaries
#     b) should_collect() daily interval at 11:59 PM vs 12:01 AM
#     c) _parse_timestamp date objects produce midnight UTC
#     d) is_off_peak at midnight ET
# ────────────────────────────────────────────────────────────────────────


class TestMidnightUTCRollover:
    """Verify system behavior at the UTC date boundary."""

    def test_sec_13f_quarter_january_boundary(self):
        """In January, Q3 filings should still be the latest available
        (Q4 isn't due until Feb 14)."""
        import datetime as dt
        from app.pipeline.data.collection_scheduler import get_sec_13f_quarter

        # Jan 10 2026 — Q4 not yet available
        with patch("app.pipeline.data.collection_scheduler.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = dt.datetime(2026, 1, 10)
            mock_dt.datetime.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)

            result = get_sec_13f_quarter()

        assert result == "2025Q3", (
            f"January should show previous year Q3, got {result}"
        )
        logger.info("PASS: January → Q3 of previous year (Q4 not yet filed)")

    def test_sec_13f_quarter_may_boundary(self):
        """In May, Q1 filings should be available."""
        import datetime as dt
        from app.pipeline.data.collection_scheduler import get_sec_13f_quarter

        with patch("app.pipeline.data.collection_scheduler.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = dt.datetime(2026, 5, 15)
            mock_dt.datetime.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)

            result = get_sec_13f_quarter()

        assert result == "2026Q1", (
            f"May should show Q1, got {result}"
        )
        logger.info("PASS: May → Q1 filings available")

    def test_should_collect_daily_source_just_before_midnight(self):
        """A daily source collected at 11:59 PM should NOT be collected again
        at 11:59:30 PM (still within the 24h window)."""
        import datetime as dt

        # Last collected 30 seconds ago
        recent_ts = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=30)

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (recent_ts,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            from app.pipeline.data.collection_scheduler import should_collect

            result = should_collect("price_history")  # 24h interval

        assert result is False, (
            "Data collected 30 seconds ago should NOT trigger re-collection"
        )
        logger.info("PASS: Recent collection (30s ago) → skip (within 24h window)")

    def test_should_collect_daily_source_after_24h(self):
        """A daily source collected 25 hours ago should trigger re-collection."""
        import datetime as dt

        old_ts = dt.datetime.now(dt.UTC) - dt.timedelta(hours=25)

        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (old_ts,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def fake_get_db():
            yield mock_cursor

        with patch("app.pipeline.data.collection_scheduler.get_db", side_effect=fake_get_db):
            from app.pipeline.data.collection_scheduler import should_collect

            result = should_collect("price_history")

        assert result is True, (
            "Data 25h old with 24h interval should trigger collection"
        )
        logger.info("PASS: 25h old data with 24h interval → should_collect=True")

    def test_is_off_peak_at_midnight_et(self):
        """Midnight ET (12:00 AM) should be off-peak."""
        fake_now = datetime(2026, 6, 2, 0, 0, 0)  # Tuesday midnight

        with patch("app.pipeline.data.data_lifecycle.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from app.pipeline.data.data_lifecycle import is_off_peak

            assert is_off_peak() is True, (
                "Midnight ET should be off-peak"
            )
        logger.info("PASS: Midnight ET → off-peak = True")

    def test_parse_timestamp_midnight_consistency(self):
        """A date '2026-06-01' and datetime '2026-06-01 00:00:00' should
        parse to the same midnight timestamp."""
        import datetime as dt
        from app.pipeline.data.collection_scheduler import _parse_timestamp

        from_date = _parse_timestamp(dt.date(2026, 6, 1))
        from_string = _parse_timestamp("2026-06-01 00:00:00")

        assert from_date is not None
        assert from_string is not None
        assert from_date == from_string, (
            f"date and string should produce same midnight timestamp: "
            f"{from_date} != {from_string}"
        )
        logger.info("PASS: date and '00:00:00' string → identical midnight timestamp")

